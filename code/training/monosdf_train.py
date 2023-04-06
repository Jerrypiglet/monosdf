import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
import sys
import torch
from tqdm import tqdm
import numpy as np
import math
import time
import copy
from pathlib import Path

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
from utils.plots import gamma2_th, get_surface_sliding

from tools.clean_funcs import remove_ckpt, remove_plots

import torch.distributed as dist

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S")
    return d1

class MonoSDFTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.opt = kwargs['opt']
        self.if_cluster = self.opt.if_cluster
        self.log_every_iter = 100 if self.if_cluster else 10

        conf = ConfigFactory.parse_file(kwargs['conf'])
        if kwargs['conf_add'] == '':
            self.conf = conf
        else:
            conf_add = ConfigFactory.parse_file(kwargs['conf_add'])
            self.conf = ConfigTree.merge_configs(conf, conf_add)
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.if_distributed = kwargs['if_distributed']
        self.if_gt_plotted = {'TRAIN': False, 'VAL': False}

        self.exp_name = self.opt.prefix + self.conf.get_string('train.expname')
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != '' else self.conf.get_string('dataset.scan_id', default='')
        if scan_id != '':
            self.exp_name = self.exp_name + '_{0}'.format(scan_id)
        self.exp_name += self.opt.append
        datetime_str = self.opt.datetime_str_input if self.opt.datetime_str_input != '' else get_datetime()
        if 'DATE' in self.exp_name:
            self.exp_name = self.exp_name.replace('DATE', datetime_str) # e.g. '20230129-162337-K-kitchen_HDR_grids_trainval_tmp'
        else:
            if not self.opt.resume:
                self.exp_name = datetime_str + '-' + self.exp_name
        print('=====self.exp_name', self.exp_name)

        self.load_from_task = ''
        if self.opt.resume:
            ckpts_root = os.path.join('../', kwargs['exps_folder_name'])
            if self.opt.load_from != '':
                self.load_from_task = self.opt.load_from
            else:
                self.load_from_task = self.exp_name
            
            exps = [_ for _ in os.listdir(ckpts_root) if _.startswith(self.load_from_task)]
            if len(exps) == 0:
                print('No exact match for [%s] found among:'%self.load_from_task, os.listdir(ckpts_root))
                raise RuntimeError
            elif len(exps) > 1:
                print('More than one matches for [%s] found; resuming from 1st one:'%self.load_from_task)
                print(exps)
            self.load_from_task = exps[0]

        # if self.opt.resume and self.opt.load_from == '':
        #     if os.path.exists(os.path.join('../',kwargs['exps_folder_name'], self.exp_name)):
        #         datetime_strs = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.exp_name))
        #         if (len(datetime_strs)) == 0:
        #             resume = False
        #             datetime_str = None
        #         else:
        #             datetime_str = sorted(datetime_strs)[-1]
        #             resume = True
        #     else:
        #         resume = False
        #         datetime_str = None
        # else:
        #     datetime_str = kwargs['datetime_str']
        #     resume = self.opt.resume

        # if self.opt.cancel_train: assert resume

        if self.GPU_INDEX == 0:
            if self.if_cluster:
                self.expdir = os.path.join(self.exps_folder_name, self.exp_name)
            else:
                utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
                self.expdir = os.path.join('../', self.exps_folder_name, self.exp_name)
            utils.mkdir_ifnotexists(self.expdir)
            # if resume:
            #     datetime_str = datetime_str
            # else:
            #     if self.opt.datetime_str != '':
            #         datetime_str = self.opt.datetime_str
            #     else:
            #         # datetime_str = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            #         datetime_str = get_datetime()

            self.expdir = self.expdir.replace('DATE', self.expdir)

            utils.mkdir_ifnotexists(os.path.join(self.expdir))

            self.plots_dir = os.path.join(self.expdir, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != '':
            dataset_conf['scan_id'] = kwargs['scan_id']
        if self.if_cluster:
            dataset_conf['data_dir'] = '/ruidata/monosdf/data/' + dataset_conf['data_dir']

        if not self.opt.cancel_train:
            self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(split='train', dataset_name='train', **dataset_conf)
        self.if_pixel_train = self.conf.get_config('dataset').get('if_pixel', False)
        self.if_hdr = self.conf.get_config('dataset').get('if_hdr', False)

        self.val_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(split='val', if_overfit_train=self.opt.if_overfit_train, dataset_name='val', **dataset_conf)
        assert self.val_dataset.if_pixel == False
        shuffle_val = False

        dataset_conf_vis_train = copy.deepcopy(dataset_conf); dataset_conf_vis_train.pop('val_frame_num'), dataset_conf_vis_train.pop('if_pixel')
        self.vis_train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(split='train', frame_num_override=8, if_pixel=False, dataset_name='vis_train', **dataset_conf_vis_train)
        assert self.vis_train_dataset.if_pixel == False

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=1500000)
        if not self.opt.cancel_train:
            self.ds_len = len(self.train_dataset) if not self.if_pixel_train else self.train_dataset.n_images
            print('Finish loading data. Dataset size: {0}'.format(self.ds_len))
            if ('scan' in scan_id and (int(scan_id[4:]) < 24 and int(scan_id[4:]) > 0)) or (not 'scan' in scan_id): # BlendedMVS, running for 200k iterations
                # if not self.if_pixel_train:
                self.nepochs = int(self.max_total_iters / self.ds_len)
            print('RUNNING FOR {0}'.format(self.nepochs))
            assert self.nepochs > 0

        if not self.opt.cancel_train:
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.batch_size if not self.if_pixel_train else self.conf.get_int('train.num_pixels'),
                                                                shuffle=True if not self.if_pixel_train else False,
                                                                collate_fn=self.train_dataset.collate_fn,
                                                                num_workers=1 if self.if_cluster else 8)

        self.vis_val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=shuffle_val,
                                                           collate_fn=self.val_dataset.collate_fn
                                                           )

        self.vis_train_dataloader = torch.utils.data.DataLoader(self.vis_train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=shuffle_val,
                                                           collate_fn=self.val_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model, if_hdr=self.if_hdr)
        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        
        if not self.opt.cancel_train:
            self.lr = self.conf.get_float('train.learning_rate')
            self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
            
            if self.Grid_MLP:
                self.optimizer = torch.optim.Adam([
                    {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                        'lr': self.lr * self.lr_factor_for_grid},
                    {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                        list(self.model.rendering_network.parameters()),
                        'lr': self.lr},
                    {'name': 'density', 'params': list(self.model.density.parameters()),
                        'lr': self.lr},
                ], betas=(0.9, 0.99), eps=1e-15)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
            # Exponential learning rate scheduler
            decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
            decay_steps = self.nepochs * self.ds_len
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        if self.if_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        self.iter_step = 0

        if self.load_from_task != '':
            old_checkpnts_dir = str(Path('../exps') / self.load_from_task / 'checkpoints')
            assert Path(old_checkpnts_dir).exists(), old_checkpnts_dir
            # else:
            #     old_checkpnts_dir = os.path.join(self.expdir, datetime_str, 'checkpoints')
            
            ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
            saved_model_state = torch.load(ckpt_path)
            print('======> Loaded from:', ckpt_path, 'Distributed:', self.if_distributed)
            if not self.if_distributed:
                # for k, v in saved_model_state["model_state_dict"].items():
                saved_model_state["model_state_dict"] = {k.replace('module.', ''): v for k, v in saved_model_state["model_state_dict"].items()}

            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']
            self.iter_step = saved_model_state['iter_step']
            # self.iter_step = 117900

            if not self.opt.cancel_train:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
                self.optimizer.load_state_dict(data["optimizer_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels_im = self.train_dataset.total_pixels_im if not self.opt.cancel_train else self.val_dataset.total_pixels_im
        self.img_res = self.train_dataset.img_res if not self.opt.cancel_train else self.val_dataset.img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()

        if not self.opt.cancel_train:
            self.n_batches = len(self.train_dataloader)
        if self.opt.cancel_train:
            self.nepochs = self.start_epoch

    def save_checkpoints(self, epoch, iter_step):
        torch.save(
            {"epoch": epoch, 'iter_step': iter_step, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, 'iter_step': iter_step, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, 'iter_step': iter_step, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, 'iter_step': iter_step, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, 'iter_step': iter_step, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, 'iter_step': iter_step, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        remove_ckpt(self.checkpoints_path, 3)

    def run(self):
        print("training exp [%s]..."%self.exp_name)
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))
            print('writing logs to ->', os.path.join(self.plots_dir, 'logs'))

        # self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):
            remove_plots(self.plots_dir, 3)

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0 and not self.opt.cancel_train:
                self.save_checkpoints(epoch, self.iter_step)

            '''
            VAL
            '''

            if_plot = (epoch % self.plot_freq == 0) or (epoch < self.plot_freq and epoch % (self.plot_freq//3) == 0)
            
            if self.GPU_INDEX == 0 and self.do_vis and if_plot:
                self.model.eval()
                # self.train_dataset.change_sampling_idx(-1)
                implicit_network = self.model.module.implicit_network if self.if_distributed else self.model.implicit_network
                
                # exporting mesh from SDF
                if not self.opt.cancel_mesh:
                    # mesh_path = '{0}/{1}_epoch{2}.ply'.format(self.plots_dir, self.plots_dir.split('/')[-3], epoch)
                    mesh_path = '{0}/{1}.ply'.format(self.plots_dir, self.plots_dir.split('/')[-2])
                    resolution = 1024 if self.opt.cancel_train else self.plot_conf['resolution']
                    print('- Exporting mesh to %s... (res %d)'%(mesh_path, resolution))
                    with torch.no_grad():
                        mesh = get_surface_sliding(path=self.plots_dir, 
                                    epoch=epoch, 
                                    sdf=lambda x: implicit_network(x)['sdf'].squeeze(1), 
                                    resolution=resolution, 
                                    grid_boundary=self.plot_conf.get('grid_boundary'), 
                                    level=0.001 if self.opt.cancel_train else 0.,  
                                    return_mesh=True,  
                                    center=self.val_dataset.center,
                                    scale=self.val_dataset.scale,
                                    )
                    print('-> Exported.')
                    # utils.mkdir_ifnotexists(self.plots_dir)
                    mesh.export(mesh_path, 'ply')

                # indices, model_input, ground_truth = next(iter(self.vis_val_dataloader))
                if not self.opt.cancel_eval:
                    for vis_split, dataloader in zip(['VAL', 'TRAIN'], [self.vis_val_dataloader, self.vis_train_dataloader]):
                    # for vis_split, dataloader in zip(['TRAIN'], [self.vis_train_dataloader]):
                        print('== Evaluating epoch %d %svis_dataloader (%d batches)...'%(epoch, vis_split, len(dataloader)))
                        for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(dataloader)):
                            model_input["intrinsics"] = model_input["intrinsics"].cuda()
                            model_input["uv"] = model_input["uv"].cuda()
                            model_input['pose'] = model_input['pose'].cuda()

                            split = utils.split_input(model_input, self.total_pixels_im, n_pixels=self.split_n_pixels)
                            res = []
                            for s in tqdm(split):
                                out = self.model(s, indices)
                                d = {'rgb_values': out['rgb_values'].detach(),
                                    'normal_map': out['normal_map'].detach(),
                                    'depth_values': out['depth_values'].detach(), 
                                }
                                    # 'sdf': out['sdf'].detach()}
                                if 'rgb_un_values' in out:
                                    d['rgb_un_values'] = out['rgb_un_values'].detach()
                                res.append(d)

                            batch_size = ground_truth['rgb'].shape[0]
                            model_outputs = utils.merge_output(res, self.total_pixels_im, batch_size)
                            plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['mask'])

                            # loss_output = self.loss(model_outputs, {k: v[0] for k, v in ground_truth.items()}, if_pixel_input=self.if_pixel_train)

                            plt.plot(implicit_network,
                                    indices,
                                    plot_data,
                                    self.plots_dir,
                                    epoch,
                                    self.img_res,
                                    if_hdr=self.if_hdr, 
                                    PREFIX=vis_split, 
                                    if_tensorboard=True, writer=self.writer, tid=self.iter_step, batch_id=data_index, if_gt_plotted=self.if_gt_plotted,
                                    **self.plot_conf
                                    )
        
                        self.if_gt_plotted[vis_split] = True

                self.model.train()

            '''
            TRAIN
            '''
            if self.opt.cancel_train:
                continue

            self.train_dataset.change_sampling_idx(self.num_pixels)

            print('== Training epoch %d (up to %d) with train_dataloader (%d samples after random sampling; %d training batches)...'%(epoch, self.nepochs, len(self.train_dataset.sampling_idx), len(self.train_dataloader)))
            n_batches = len(self.train_dataloader)

            # tic = time.time()
            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.train_dataloader)):
                '''
                indices: image idxes
                '''

                '''
                debug: test dataloading and summary writing ([!!!] which might be slow on the cluster)
                '''
                # self.iter_step += 1           
                # self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                # self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                # self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                # self.scheduler.step()
                # print(self.iter_step, time.time() - tic)
                # tic = time.time()
                # continue

                model_input = {k: v.cuda() if type(v)==torch.Tensor else v for k, v in model_input.items()}
                # model_input["intrinsics"] = model_input["intrinsics"].cuda()
                # model_input["uv"] = model_input["uv"].cuda()
                # model_input['pose'] = model_input['pose'].cuda()
                
                self.optimizer.zero_grad()
                
                model_outputs = self.model(model_input, indices, if_pixel_input=self.if_pixel_train)

                loss_output = self.loss(model_outputs, ground_truth, if_pixel_input=self.if_pixel_train)
                loss = loss_output['loss']
                loss.backward()
                self.optimizer.step()
                
                if self.if_hdr:
                    psnr = rend_util.get_psnr(gamma2_th(model_outputs['rgb_values']),
                                            gamma2_th(ground_truth['rgb'].cuda().reshape(-1,3))
                                            )
                else:
                    psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                            ground_truth['rgb'].cuda().reshape(-1,3))
                
                self.iter_step += 1                
                
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else -1, 
                if self.GPU_INDEX == 0:
                    print(
                        '{0} [{1}] ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, psnr = {7}, bete={8}, alpha={9}; [{10}]'
                            .format(self.exp_name, 
                                    epoch, data_index, n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item() if self.if_distributed else self.model.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item() if self.if_distributed else 1. / self.model.density.get_beta().item(), 
                                    CUDA_VISIBLE_DEVICES, 
                                    ))
                    if self.iter_step % self.log_every_iter == 0:
                        self.writer.add_scalar('Loss/loss', loss.item(), self.iter_step)
                        self.writer.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), self.iter_step)
                        self.writer.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), self.iter_step)
                        self.writer.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), self.iter_step)
                        self.writer.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), self.iter_step)
                        self.writer.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), self.iter_step)
                        self.writer.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), self.iter_step)
                        
                        self.writer.add_scalar('Statistics/beta', self.model.module.density.get_beta().item() if self.if_distributed else self.model.density.get_beta().item(), self.iter_step)
                        self.writer.add_scalar('Statistics/alpha', 1. / self.model.module.density.get_beta().item() if self.if_distributed else 1. / self.model.density.get_beta().item(), self.iter_step)
                        self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                        self.writer.add_scalar('Statistics/epoch', epoch, self.iter_step)
                        
                        if self.Grid_MLP:
                            self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                            self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                            self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
    
                if not self.if_pixel_train:
                    self.train_dataset.change_sampling_idx(self.num_pixels)
    
                self.scheduler.step()

            print('== Training epoch %d with train_dataloader... DONE'%epoch)

        if self.GPU_INDEX == 0 and not self.opt.cancel_train:
            self.save_checkpoints(epoch, self.iter_step)

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, mask):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)

        mask = mask.reshape(1, 1, self.img_res[0], self.img_res[1])

        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'mask': mask,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()
