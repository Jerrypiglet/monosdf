import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from utils.utils_misc import yellow, white_blue, green, green_text, yellow_text
import utils.general as utils
from utils import rend_util
from glob import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import random
from pathlib import Path

# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                data_dir,
                img_res,
                if_hdr=False, # if load HDR images (e.g. OpenRooms, kitchen)
                if_pixel=False, # if return batch of random pixels
                if_gt_data=True, 
                if_overfit_train=False, 
                center_crop_type='xxxx',
                use_mask=True,
                num_views=-1, 
                split='train',
                val_frame_num = -1, 
                train_frame_idx_input = [], 
                val_frame_idx_input = [], 
                frame_num_override = -1, 
                has_splits = False, 
                if_sample_frames = True, 
                dataset_name = '', 
                ):

        # self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))
        # if not Path(self.instance_dir).exists():
            # print('>>>>', self.instance_dir, 'split does not exist; switching to....')
        print(green('>>>>>>>>>> [SceneDatasetDN-%s] split: %s'%(dataset_name, split)))
        print(train_frame_idx_input, val_frame_idx_input, val_frame_num, if_sample_frames)
        
        self.instance_dir = os.path.join('../data', data_dir)
        self.scene_dir = self.instance_dir
        assert Path(self.instance_dir).exists(), "Data directory does not exist: %s"%self.instance_dir
        if has_splits:
            self.instance_dir = os.path.join(self.instance_dir, split)

        self.total_pixels_im = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        self.split = split
        assert self.split in ['train', 'val']
        self.val_frame_num = val_frame_num
        self.val_frame_idx_input = val_frame_idx_input
        self.train_frame_idx_input = train_frame_idx_input

        self.if_hdr = if_hdr
        self.if_pixel = if_pixel and self.split == 'train'
        self.if_overfit_train = if_overfit_train
        
        assert os.path.exists(self.instance_dir), "Data directory is empty: %s"%self.instance_dir

        self.sampling_idx = None
        self.dataset_name = dataset_name
        self.frame_num_override = frame_num_override
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        self.if_gt_data = if_gt_data
        
        if self.if_gt_data:
            self.image_paths = glob_data(os.path.join('{0}/{1}'.format(self.instance_dir, 'Image'), "*.png" if not self.if_hdr else "*.exr"))
        else:
            self.image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png" if not self.if_hdr else "Image/*.exr"))
            # self.image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "Image/*.png" if not self.if_hdr else "Image/*.exr"))
        assert len(self.image_paths) > 0, "No images found in %s"%self.instance_dir
        self.image_paths.sort()
        # self.image_paths = self.image_paths[:15] # for faster debugging only
        self.filenames = [Path(_).stem.replace('_rgb', '') for _ in self.image_paths] # ['000000', '000001', '000002', ...]

        # check: 0-based; incremental order; no missing frames
        assert [_[:3] for _ in self.filenames] == ['%03d'%_ for _ in range(len(self.filenames))]
        assert self.filenames[0] == '000_0001'

        if self.if_gt_data:
            # depth_paths = glob_data(os.path.join('{0}/{1}'.format(self.instance_dir, 'depth'), "*.npy"))
            # normal_paths = glob_data(os.path.join('{0}/{1}'.format(self.instance_dir, 'normal'), "*.npy"))
            depth_paths = [str(Path(self.instance_dir) / 'depth' / ('%s.npy'%_)) for _ in self.filenames]
            normal_paths = [str(Path(self.instance_dir) / 'normal' / ('%s.npy'%_)) for _ in self.filenames]
        else:
            # depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
            # normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
            depth_paths = [str(Path(self.instance_dir) / ('%s_depth.npy'%_)) for _ in self.filenames]
            normal_paths = [str(Path(self.instance_dir) / ('%s_normal.npy'%_)) for _ in self.filenames]
        
        assert [Path(_).exists() for _ in depth_paths]
        assert [Path(_).exists() for _ in normal_paths]
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            # if self.if_gt_data:
            #     # mask_paths = glob_data(os.path.join('{0}/{1}'.format(self.instance_dir, 'mask'), "*.npy"))
            #     mask_paths = [str(Path(self.instance_dir) / 'mask' / ('%s.npy'%_)) for _ in self.filenames]
            # else:
            #     # mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
            #     # mask_paths = [str(Path(self.instance_dir) / ('%s_mask.npy'%_)) for _ in self.filenames]
            mask_paths = [str(Path(self.instance_dir) / 'ImMask' / ('%s.png'%_)) for _ in self.filenames]
            assert [Path(_).exists() for _ in mask_paths]
        else:
            mask_paths = None

        self.n_images = len(self.image_paths)
        assert self.n_images > 0
        self.frame_idx_list = list(range(self.n_images))

        self.if_sample_frames = False
        if not (val_frame_num == -1 and val_frame_idx_input == []) and if_sample_frames:
        # if if_sample_frames:
            '''
            sample train/val frames according to val_frame_idx_input and val_frame_num
            '''
            assert self.num_views == -1, 'should set num_views to -1 for openrooms/kitchen where val frames are sampled.'
            self.sample_frames()
            self.if_sample_frames = True
        else:
            print('>>> [SceneDatasetDN-%s]'%(self.dataset_name), yellow('Not sampling frames.'))
        
        # if self.frame_num_override != -1:
        #     import random
        #     # self.frame_idx_list = self.frame_idx_list[:frame_num]
        #     self.frame_idx_list = random.sample(self.frame_idx_list, self.frame_num_override)
        #     # self.n_images = min(self.n_images, len(self.frame_idx_list))
        #     # self.image_paths = [self.image_paths[_] for _ in self.frame_idx_list]
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        print(yellow_text('Loaded cameras.npz from %s'%(self.cam_file)))
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        
        self.scale_mat_file = '{0}/scale_mat.npy'.format(self.scene_dir)
        print(yellow_text('Loaded scale_mat.npy from %s'%(self.scale_mat_file)))
        scale_mat = np.load(self.scale_mat_file, allow_pickle=True).item()
        self.center, self.scale = scale_mat['center'], scale_mat['scale']
        print('center, scale', self.center, self.scale)

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in self.image_paths:
            if self.if_hdr:
                rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                assert rgb is not None
                rgb = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_BGR2RGB).reshape(-1, 3)
            else:
                rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(3, -1).transpose(1, 0) # (N, 3)
            assert not np.any(np.isnan(rgb))
            assert not np.any(np.isinf(rgb))
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            if self.if_gt_data:
                depth = depth * self.scale # manually normalize GT depth (which is un-normalized)
                
            depth[np.isnan(depth)] = 1./1000.
            depth[np.isinf(depth)] = 1./1000.
            assert not np.any(np.isnan(depth))
            assert not np.any(np.isinf(depth))
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1. # [0, 1] -> [-1, 1]; OpenCV coords
            assert not np.any(np.isnan(normal))
            assert not np.any(np.isinf(normal))
            self.normal_images.append(torch.from_numpy(normal).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                if Path(path).suffix == '.npy':
                    mask = np.load(path)
                elif Path(path).suffix == '.png':
                    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    if mask.dtype == np.uint8:
                        mask = mask.astype(np.float32) / 255.
                    assert np.amax(mask) <= 1.
                    assert np.amin(mask) >= 0.
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())
        
        # checking loaded depth/normals...
        for idx, (mask, normal, depth) in enumerate(zip(self.mask_images, self.normal_images, self.depth_images)):
            # print(mask.shape, torch.sum(mask)/mask.shape[0])
            assert (mask==1.).shape[0] > 0
            _mask = (mask==1.).reshape(-1)
            # _depth = depth[_mask]
            _normal = normal[_mask]
            if torch.amax(torch.abs(torch.linalg.norm(_normal, axis=-1)-1)) > 1e-3:
                import ipdb; ipdb.set_trace()

        # get global uv
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        self.uv = uv.reshape(2, -1).transpose(1, 0) # (HW, 2)

        if self.if_pixel:
            self.convert_to_pixels()

        print(green_text('<<<<<<<<<< [SceneDatasetDN-%s-%d]'%(self.dataset_name, self.__len__())), green_text(str(self.frame_idx_list[:15])))
        # if frame_num != -1:
        #     import ipdb; ipdb.set_trace()

    def convert_to_pixels(self):
        # intrinsics = input["intrinsics"]
        # uv = input["uv"]
        # pose = input["pose"]
        # for intrinsics, pose in zip(self.intrinsics_all, self.pose_all): # (4, 4), (4, 4)
        pose_all_tensor = torch.stack(self.pose_all) # (N, 4, 4)
        intrinsics_all_tensor = torch.stack(self.intrinsics_all) # (N, 4, 4)
        _N = pose_all_tensor.shape[0]
        uv_all_tensor = self.uv.unsqueeze(0).expand(_N, -1, -1) # (N, HW, 2)
        _HW = uv_all_tensor.shape[1]

        # generate rays for pixels **in each split**
        ray_dirs, cam_loc = rend_util.get_camera_params(uv_all_tensor, pose_all_tensor, intrinsics_all_tensor) # (N, HW, 3), (N, 3)
        self.ray_dirs = ray_dirs[self.frame_idx_list].reshape(-1, 3) # (NHW, 3)
        self.ray_cam_loc = cam_loc.unsqueeze(1).expand(-1, _HW, -1)[self.frame_idx_list].view(-1, 3) # (N, HW, 3) -> (N'HW, 3)

        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv_all_tensor, torch.eye(4)[None].expand(_N, -1, -1), intrinsics_all_tensor) # (N, HW, 3)
        self.ray_dirs_tmp = ray_dirs_tmp[self.frame_idx_list].reshape(-1, 3) # (N'HW, 3)
        
        del uv_all_tensor
        del self.uv

        self.ray_rgb = torch.stack(self.rgb_images)[self.frame_idx_list].view(-1, 3) # (N, HW, 3) -> (N'HW, 3)
        del self.rgb_images
        self.ray_depth = torch.stack(self.depth_images)[self.frame_idx_list].view(-1, 1) # (N, HW, 3) -> (N'HW, 1)
        del self.depth_images
        self.ray_mask = torch.stack(self.mask_images)[self.frame_idx_list].view(-1, 1) # (N, HW, 3) -> (N'HW, 1)
        del self.mask_images
        self.ray_normal = torch.stack(self.normal_images)[self.frame_idx_list].view(-1, 3) # (N, HW, 3) -> (N'HW, 3)
        del self.normal_images


        # self.ray_frame_idx = torch.arange(0, _N, dtype=torch.int32).unsqueeze(-1).expand(-1, _HW).flatten()
        self.ray_frame_idx = np.repeat(np.arange(_N, dtype=np.int32).reshape(-1, 1), _HW, 1).flatten()
        self.ray_pose = pose_all_tensor.unsqueeze(1).expand(-1, _HW, -1, -1)[self.frame_idx_list].view(-1, 4, 4) # (N, HW, 4, 4) -> (N'HW, 4, 4)

        self.total_pixels = self.ray_rgb.shape[0]
        self.sampling_idx = list(range(self.total_pixels))

    def sample_frames(self):
        '''
        frame_idx in [0, ..., total_frame_num-1]
        '''
        self.frame_idx_list = list(range(self.n_images))
        frame_list = list(range(self.n_images))
        self.val_frame_idx_list = []
        self.train_frame_idx_list = []
        if self.val_frame_idx_input != []:
            if self.val_frame_num == -1:
                self.val_frame_num = len(self.val_frame_idx_input)
            else:
                if len(self.val_frame_idx_input) >= self.val_frame_num:
                    self.val_frame_idx_input = self.val_frame_idx_input[:self.val_frame_num]
            frame_list = list(set(frame_list) - set(self.val_frame_idx_input))
            self.val_frame_idx_list += self.val_frame_idx_input

        self.train_frame_num = self.n_images - self.val_frame_num
        if self.train_frame_idx_input != []:
            if len(self.train_frame_idx_input) >= self.train_frame_num:
                self.train_frame_idx_input = self.train_frame_idx_input[:self.train_frame_num]
            frame_list = list(set(frame_list) - set(self.train_frame_idx_input))
            self.train_frame_idx_list += self.train_frame_idx_input

        if len(self.val_frame_idx_list) < self.val_frame_num:
            val_frame_num_to_sample = self.val_frame_num - len(self.val_frame_idx_list)
            # val_frame_idx_list = [frame_list[_] for _ in list(range(0, len(frame_list), self.n_images//max(val_frame_num_to_sample-1, 2)))]
            val_frame_idx_list = [frame_list[_] for _ in random.sample(range(len(frame_list)), val_frame_num_to_sample)]
            assert len(val_frame_idx_list) == val_frame_num_to_sample, '%d != %d!'%(len(val_frame_idx_list), val_frame_num_to_sample)
            self.val_frame_idx_list += val_frame_idx_list
            frame_list = list(set(frame_list) - set(val_frame_idx_list))
        self.train_frame_idx_list += frame_list

        assert len(self.train_frame_idx_list) + len(self.val_frame_idx_list) == self.n_images
        self.train_frame_num = len(self.train_frame_idx_list)
        self.val_frame_num = len(self.val_frame_idx_list)

        self.frame_idx_list = self.train_frame_idx_list if self.split=='train' else self.val_frame_idx_list

        if self.if_overfit_train:
            assert len(self.train_frame_idx_list) >= self.val_frame_num
            _ = max(len(self.train_frame_idx_input), min(self.val_frame_num, len(self.train_frame_idx_list)))
            self.frame_idx_list = self.train_frame_idx_list[:_]
            print('[SceneDatasetDN-%s (name: %s)] -> sample_frames(): OVERFIT_TRAIN: %d val from train split'%(self.dataset_name, self.split, len(self.val_frame_idx_list)))
            # print(self.frame_idx_list)
        else:
            print('[SceneDatasetDN-%s (name %s)] -> sample_frames(): %d train, %d val'%(self.dataset_name, self.split, self.train_frame_num, self.val_frame_num))
            
    def __len__(self):
        # if self.split == 'train':
        #     return self.n_images
        # else:
        #     return 1
        if self.if_pixel:
            return len(self.sampling_idx)
        else:
            if self.if_sample_frames: # or self.frame_num_override != -1:
                return len(self.frame_idx_list)
                # if self.split == 'train':
                #     return self.train_frame_num
                # elif self.split == 'val':
                #     return self.val_frame_num
            else:
                return self.n_images

    def __getitem__(self, idx):
        
        if self.if_pixel:
            # idx becomes ray idx
            # tic = time.time()
            assert not self.num_views >= 0, 'not supported for openrooms/kitchen'
            # assert self.sampling_idx is not None, 'should be -1 for train mode, if you are in pixel mode'
            _idx = self.sampling_idx[idx]

            sample = {
                # "uv": self.uv,
                # "intrinsics": self.intrinsics_all[_idx],
                # "pose": self.pose_all[_idx], 
                # 'image_path': self.image_paths[_idx], 
                'ray_dirs': self.ray_dirs[_idx], 
                'ray_dirs_tmp': self.ray_dirs_tmp[_idx], 
                'ray_cam_loc': self.ray_cam_loc[_idx], 
                'ray_pose': self.ray_pose[_idx], 
            }
            
            ground_truth = {
                "rgb": self.ray_rgb[_idx],
                "depth": self.ray_depth[_idx],
                "mask": self.ray_mask[_idx],
                "normal": self.ray_normal[_idx]
            }
            # print(int(self.ray_frame_idx[_idx]), time.time() - tic)
            return int(self.ray_frame_idx[_idx]), sample, ground_truth

        else:
            _idx = self.frame_idx_list[idx]
            print(idx, _idx, len(self.intrinsics_all))

            if self.num_views >= 0:
                image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
                _idx = image_ids[random.randint(0, self.num_views - 1)]
            
            sample = {
                "uv": self.uv,
                "intrinsics": self.intrinsics_all[_idx],
                "pose": self.pose_all[_idx], 
                'image_path': self.image_paths[_idx], 
            }
            
            ground_truth = {
                "rgb": self.rgb_images[_idx],
                "depth": self.depth_images[_idx],
                "mask": self.mask_images[_idx],
                "normal": self.normal_images[_idx]
            }

            if self.sampling_idx is not None:
                ground_truth["rgb"] = self.rgb_images[_idx][self.sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[_idx]
                ground_truth["normal"] = self.normal_images[_idx][self.sampling_idx, :]
                ground_truth["depth"] = self.depth_images[_idx][self.sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[_idx]
                ground_truth["mask"] = self.mask_images[_idx][self.sampling_idx, :]
                ground_truth["full_mask"] = self.mask_images[_idx]
            
                sample["uv"] = self.uv[self.sampling_idx, :]

            return idx, sample, ground_truth

    def collate_fn(self, batch_list, if_pixel=False):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        # tic = time.time()
        batch_list = zip(*batch_list)

        all_parsed = []
        for _, entry in enumerate(batch_list):
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    if isinstance(entry[0][k], int) or isinstance(entry[0][k], str):
                        ret[k] = [obj[k] for obj in entry]
                    else:
                        # if if_pixel:
                            # ret[k] = torch.cat([obj[k] for obj in entry], axis=0) # cat rays
                        # else:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    # raise RuntimeError('Invalid data in batch: %s-%s'%str(entry[0]))
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return_tuple = tuple(all_parsed)

        # print('-->.', all_parsed[0], time.time() - tic)
        return return_tuple

    default_collate = torch.utils.data.dataloader.default_collate

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
            # self.sampling_ratio = 1.
        else:
            if self.if_pixel:
                total_sampling_size = int(float(sampling_size) / float(self.total_pixels_im) * self.total_pixels)
                self.sampling_idx = torch.randperm(self.total_pixels)[:total_sampling_size]
            else:
                self.sampling_idx = torch.randperm(self.total_pixels_im)[:sampling_size]
            # self.sampling_ratio = float(sampling_size) / float(self.total_pixels_im)

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
