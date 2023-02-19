'''
(monosdf-py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/preprocess$ python batch_extract.py --gpu_ids 0 1 2 4 5 6 7 --gpu_total 7
'''
from pathlib import Path
import pickle
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
import numpy as np
import subprocess
import multiprocessing
from multiprocessing import Pool
import sys
sys.path.insert(0, '../code')
from utils.utils_misc import str2bool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_total', type=int, default=2, help='total num of gpus available')
parser.add_argument('--gpu_ids', nargs='+', help='list of gpu ids available; set to -1 to use gpus: [0, ..., gpu_total-1]', required=False, default=[-1])
parser.add_argument('--workers_total', type=int, default=-1, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
parser.add_argument('--debug', action='store_true', help='not rendering; just showing missing files')

opt = parser.parse_args()

def render(_):
    cmd, gpu_id = _
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(cmd, gpu_id)
    command = 'python extract_monocular_cues.py --task %s --img_path ../data/%s/Image --output_path ../data/%s --omnidata_path /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch --pretrained_models /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch/pretrained_models/'%(cmd[0], cmd[1], cmd[1])
    print(command)
    if not opt.debug:
        subprocess.call(command.split(' '))

import os
if opt.workers_total == -1:
    opt.workers_total = opt.gpu_total

tic = time.time()
# print('==== executing %d commands...'%len(cmd_list))
# p = Pool(processes=opt.workers_total, initializer=init, initargs=(child_env,))
p = Pool(processes=opt.workers_total)
# gpu_ids = [2, 3, 4, 5, 6, 7]
gpu_ids = opt.gpu_ids
assert len(gpu_ids) >= opt.gpu_total

# total_frames = 202 if opt.split == 'train' else 10 # kitchen
# total_frames = 14 if opt.split == 'train' else 8 # OR-mini
# total_frames = 345 if opt.split == 'train' else 0 # OR-mini
# frame_idx_list = list(range(total_frames))

# splits = np.array_split(frame_idx_list, opt.workers_total)
# job_list = [(int(_[0]), int(_[-1])+1) for _ in splits]

scene_list = [
    # 'public_re_0203-main_xml1-scene0552_00_rescaledSDR/scan1', 
    # 'public_re_0203-main_xml-scene0002_00_rescaledSDR/scan1', 
    # 'public_re_0203-mainDiffMat_xml1-scene0608_01_rescaledSDR/scan1', 
    # 'public_re_0203-mainDiffMat_xml-scene0603_00_rescaledSDR/scan1', 
    # 'public_re_0203-mainDiffMat_xml-scene0603_00_rescaledSDR_darker/scan1', 
    # 'public_re_0203-main_xml-scene0005_00_rescaledSDR_darker/scan1', 
    # 'livingroom/trainval', 
    # 'livingroom0/trainval', 
    
    # 'Matterport3D/qoiz87JEwZ2_main', 
    # 'Matterport3D/qoiz87JEwZ2', 
    # 'Matterport3D/qoiz87JEwZ2_darker', 
    # 'Matterport3D/2t7WUuJeko7_main', 
    # 'Matterport3D/2t7WUuJeko7', 
    # 'Matterport3D/mJXqzFtmKg4_main', 
    # 'Matterport3D/mJXqzFtmKg4', 
    'Matterport3D//17DRP5sb8fy_single', 
    ]
job_list = []
for scene in scene_list:
    for task in ['depth', 'normal']:
        job_list.append((task, scene))

cmd_list = [(_cmd, gpu_ids[_idx%opt.gpu_total]) for _idx, _cmd in enumerate(job_list)]
list(tqdm(p.imap_unordered(render, cmd_list), total=len(cmd_list)))
p.close()
p.join()
print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))

