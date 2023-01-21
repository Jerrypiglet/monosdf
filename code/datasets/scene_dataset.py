import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import random
from pathlib import Path

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id: str='scan0',
                 num_views=-1,  
                 ):

        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"
        
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        self.image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(self.image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in self.image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        # used a fake depth image and normal image
        self.depth_images = []
        self.normal_images = []

        for path in self.image_paths:
            depth = np.ones_like(rgb[:, :1])
            self.depth_images.append(torch.from_numpy(depth).float())
            normal = np.ones_like(rgb)
            self.normal_images.append(torch.from_numpy(normal).float())
            
    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
            
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "normal": self.normal_images[idx],
        }
        
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["mask"] = torch.ones_like(self.depth_images[idx][self.sampling_idx, :])
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']


# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                data_dir,
                img_res,
                scan_id: str='scan0',
                if_hdr=False, # if load HDR images (e.g. OpenRooms, kitchen)
                if_gt_data=True, 
                center_crop_type='xxxx',
                use_mask=False,
                num_views=-1, 
                split='train',
                val_frame_num = -1, 
                val_frame_idx_input = [], 
                train_frame_idx_input = []
                ):

        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        self.split = split
        assert self.split in ['train', 'val']
        self.val_frame_num = val_frame_num
        self.val_frame_idx_input = val_frame_idx_input
        self.train_frame_idx_input = train_frame_idx_input

        self.if_hdr = if_hdr
        
        assert os.path.exists(self.instance_dir), "Data directory is empty: %s"%self.instance_dir

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths

        # self.if_gt_data = False
        # self.if_gt_data = True

        self.if_gt_data = if_gt_data
        
        if self.if_gt_data:
            self.image_paths = glob_data(os.path.join('{0}/{1}'.format(self.instance_dir, 'image'), "*.png" if not self.if_hdr else "*.exr"))
        else:
            self.image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png" if not self.if_hdr else "image/*.exr"))
        self.image_paths.sort()
        self.filenames = [Path(_).stem.replace('_rgb', '') for _ in self.image_paths] # ['000000', '000001', '000002', ...]

        # check: 0-based; incremental order; no missing frames
        assert self.filenames == ['%06d'%_ for _ in range(len(self.filenames))]
        assert self.filenames[0] == '000000'

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
            if self.if_gt_data:
                # mask_paths = glob_data(os.path.join('{0}/{1}'.format(self.instance_dir, 'mask'), "*.npy"))
                mask_paths = [str(Path(self.instance_dir) / 'mask' / ('%s.npy'%_)) for _ in self.filenames]
            else:
                # mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
                mask_paths = [str(Path(self.instance_dir) / ('%s_mask.npy'%_)) for _ in self.filenames]
            assert [Path(_).exists() for _ in mask_paths]
        else:
            mask_paths = None

        self.n_images = len(self.image_paths)
        assert self.n_images > 0
        self.frame_idx_list = list(range(self.n_images))

        self.if_sample_frames = False
        if not (val_frame_num == -1 and val_frame_idx_input == []):
            '''
            sample train/val frames according to val_frame_idx_input and val_frame_num
            '''
            assert self.num_views == -1, 'should set num_views to -1 for openrooms/kitchen where val frames are sampled.'
            self.sample_frames()
            self.if_sample_frames = True
        else:
            '''
            no splits; train and val splits are the same (e.g. default scannet setting)
            '''
            pass
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

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
            depth[np.isnan(depth)] = 1./1000.
            depth[np.isinf(depth)] = 1./1000.
            assert not np.any(np.isnan(depth))
            assert not np.any(np.isinf(depth))
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
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
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

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

        print('[SceneDatasetDN-%s] -> sample_frames(): %d train, %d val'%(self.split, self.train_frame_num, self.val_frame_num))
        print(self.val_frame_idx_list)

        # if self.if_overfit_train:
        #     assert len(self.train_frame_idx_list) >= self.val_frame_num
        #     self.val_frame_idx_list = self.train_frame_idx_list[:self.val_frame_num]

    def __len__(self):
        # if self.split == 'train':
        #     return self.n_images
        # else:
        #     return 1
        if self.if_sample_frames:
            if self.split == 'train':
                return self.train_frame_num
            elif self.split == 'val':
                return self.val_frame_num
        else:
            return self.n_images

    def __getitem__(self, idx):
        _idx = self.frame_idx_list[idx]

        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            _idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
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
         
            sample["uv"] = uv[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    if isinstance(entry[0][k], int) or isinstance(entry[0][k], str):
                        ret[k] = [obj[k] for obj in entry]
                    else:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    # raise RuntimeError('Invalid data in batch: %s-%s'%str(entry[0]))
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    default_collate = torch.utils.data.dataloader.default_collate

    # def collate_fn(batch):
    #     """
    #     Data collater.

    #     Assumes each instance is a dict.    
    #     Applies different collation rules for each field.
    #     Args:
    #         batches: List of loaded elements via Dataset.__getitem__
    #     """
    #     collated_batch = {}
    #     for key in batch[0]:
    #         # if key in []:
    #         #     collated_batch[key] = [elem[key] for elem in batch]
    #         # else:
    #         try:
    #             collated_batch[key] = default_collate([elem[key] for elem in batch])
    #         except:
    #             print('[!!!!] Type error in collate_fn: ', key)

    #     return collated_batch

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
