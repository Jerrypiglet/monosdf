import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import math
import time

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
from utils.plots import gamma2_th, get_surface_sliding

import torch.distributed as dist

class MonoSDFTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.opt = kwargs['opt']
        self.if_cluster = self.opt.if_cluster

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.if_distributed = kwargs['if_distributed']

        self.expname = self.opt.expname_pre + self.conf.get_string('train.expname') + kwargs['expname']

        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != '' else self.conf.get_string('dataset.scan_id', default='')
        if scan_id != '':
            self.expname = self.expname + '_{0}'.format(scan_id)

        if self.opt.resume != '':
            self.expname = self.opt.resume
            kwargs['is_continue'] = True

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            if self.if_cluster:
                self.expdir = os.path.join(self.exps_folder_name, self.expname)
            else:
                utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
                self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            if is_continue:
                self.timestamp = timestamp
            else:
                self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != '':
            dataset_conf['scan_id'] = kwargs['scan_id']
        if self.if_cluster:
            dataset_conf['data_dir'] = '/ruidata/monosdf/data/' + dataset_conf['data_dir']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(split='train', **dataset_conf)
        self.if_pixel_train = self.conf.get_config('dataset').get('if_pixel', False)
        self.if_hdr = self.conf.get_config('dataset').get('if_hdr', False)

        val_frame_num = dataset_conf.get('val_frame_num', -1)
        val_frame_idx_input = dataset_conf.get('val_frame_idx_input', [])
        train_frame_idx_input = dataset_conf.get('train_frame_idx_input', [])
        if val_frame_num == -1 and train_frame_idx_input == [] and val_frame_idx_input == []:
            self.val_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(split='train', **dataset_conf)
            shuffle_val = True
        else:
            self.val_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(split='val', **dataset_conf)
            shuffle_val = False

        assert self.val_dataset.if_pixel == False

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=1000000)
        self.ds_len = len(self.train_dataset) if not self.if_pixel_train else self.train_dataset.n_images
        print('Finish loading data. Dataset size: {0}'.format(self.ds_len))
        if ('scan' in scan_id and (int(scan_id[4:]) < 24 and int(scan_id[4:]) > 0)) or (not 'scan' in scan_id): # BlendedMVS, running for 200k iterations
            # if not self.if_pixel_train:
            self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))
        assert self.nepochs > 0

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size if not self.if_pixel_train else self.conf.get_int('train.num_pixels'),
                                                            shuffle=True if not self.if_pixel_train else False,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=16 if self.if_cluster else 8)
        self.plot_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=shuffle_val,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model, if_hdr=self.if_hdr)
        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

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

        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')
            
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

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels_im = self.train_dataset.total_pixels_im
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()

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

    def run(self):
        print("training...")
        if self.GPU_INDEX == 0 :
            self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))
            print('writing logs to ->', os.path.join(self.plots_dir, 'logs'))

        # self.iter_step = 0
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch, self.iter_step)

            '''
            VAL
            '''

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0:
                self.model.eval()
                self.train_dataset.change_sampling_idx(-1)
                implicit_network = self.model.module.implicit_network if self.if_distributed else self.model.implicit_network
                
                #for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                print('== Evaluating epoch %d plot_dataloader (%d batches)...'%(epoch, len(self.plot_dataloader)))

                # exporting mesh from SDF
                mesh_path = '{0}/{1}_epoch{2}.ply'.format(self.plots_dir, self.plots_dir.split('/')[-3], epoch)
                print('- Exporting mesh to %s...'%mesh_path)
                with torch.no_grad():
                    mesh = get_surface_sliding(path=self.plots_dir, 
                                epoch=epoch, 
                                sdf=lambda x: implicit_network(x)[:, 0], 
                                resolution=self.plot_conf.get('resolution'), 
                                grid_boundary=self.plot_conf.get('grid_boundary'), 
                                level=0,  
                                return_mesh=True,  
                                )
                print('-> Exported.')
                # utils.mkdir_ifnotexists(self.plots_dir)
                mesh.export(mesh_path, 'ply')

                # indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.plot_dataloader)):
                    print(model_input['image_path'])
                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input['pose'] = model_input['pose'].cuda()
                    
                    split = utils.split_input(model_input, self.total_pixels_im, n_pixels=self.split_n_pixels)
                    res = []
                    for s in tqdm(split):
                        out = self.model(s, indices)
                        d = {'rgb_values': out['rgb_values'].detach(),
                            'normal_map': out['normal_map'].detach(),
                            'depth_values': out['depth_values'].detach()}
                        if 'rgb_un_values' in out:
                            d['rgb_un_values'] = out['rgb_un_values'].detach()
                        res.append(d)

                    # print('-- Plotting...')
                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels_im, batch_size)
                    plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])
                    # print('-- plt.plot...')

                    plt.plot(implicit_network,
                            indices,
                            plot_data,
                            self.plots_dir,
                            epoch,
                            self.img_res,
                            if_hdr=self.if_hdr, 
                            if_tensorboard=True, writer=self.writer, tid=self.iter_step, batch_id=data_index, 
                            **self.plot_conf
                            )

                self.model.train()

            '''
            TRAIN
            '''

            self.train_dataset.change_sampling_idx(self.num_pixels)

            print('== Training epoch %d (up to %d) with train_dataloader (%d samples after random sampling; %d training batches)...'%(epoch, self.nepochs, len(self.train_dataset.sampling_idx), len(self.train_dataloader)))
            n_batches = len(self.train_dataloader)

            tic = time.time()
            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.train_dataloader)):
                '''
                indices: image idxes
                '''
                self.iter_step += 1           
                self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                self.scheduler.step()
                print(self.iter_step, time.time() - tic)
                tic = time.time()
                continue

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
                        '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9}, alpha={10}; [{11}]'
                            .format(self.expname, 
                                    self.timestamp, epoch, data_index, n_batches, loss.item(),
                                    loss_output['rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    psnr.item(),
                                    self.model.module.density.get_beta().item() if self.if_distributed else self.model.density.get_beta().item(),
                                    1. / self.model.module.density.get_beta().item() if self.if_distributed else 1. / self.model.density.get_beta().item(), 
                                    CUDA_VISIBLE_DEVICES, 
                                    ))
                    
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


        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch, self.iter_step)

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
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
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
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
