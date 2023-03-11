# adapted from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model")
parser.set_defaults(omnidata_path='/home/yuzh/Projects/omnidata/omnidata_tools/torch/')

parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models")
parser.set_defaults(pretrained_models='/home/yuzh/Projects/omnidata/omnidata_tools/torch/pretrained_models/')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")

parser.add_argument('--pad_H', type=int, help="", default=-1)
parser.add_argument('--pad_W', type=int, help="", default=-1)

parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = args.pretrained_models 
omnidata_path = args.omnidata_path

sys.path.append(args.omnidata_path)
print(sys.path)
from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

trans_topil = transforms.ToPILImage()
os.system(f"mkdir -p {args.output_path}")
assert torch.cuda.is_available()
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# image_size = 384
# image_size = 320
image_size = None # no resizing and cropping

'''
get target task and model

following https://github.com/EPFL-VILAB/omnidata/blob/92dd37d26e5f51109e5a96ad1991b3f74165e323/omnidata_tools/torch/demo.py
'''

if args.task == 'normal':
    
    ## Version 1 model
    # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
    # model = UNet(in_channels=3, out_channels=3)
    # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

    # if 'state_dict' in checkpoint:
    #     state_dict = {}
    #     for k, v in checkpoint['state_dict'].items():
    #         state_dict[k.replace('model.', '')] = v
    # else:
    #     state_dict = checkpoint
    
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    transform_list = []
    if args.pad_H > 0 and args.pad_W > 0:
        transform_list += [transforms.Pad((0, 0, args.pad_W, args.pad_H), fill=0, padding_mode='constant')]
    if image_size is None:
        transform_list += [get_transform('rgb', image_size=None)]
    else:
        transform_list += [transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            get_transform('rgb', image_size=None), 
                                            ]
    trans_totensor = transforms.Compose(transform_list)

elif args.task == 'depth':
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    transform_list = []
    if args.pad_H > 0 and args.pad_W > 0:
        transform_list += [transforms.Pad((0, 0, args.pad_W, args.pad_H), fill=0, padding_mode='constant')]
    if image_size is None:
        transform_list += [transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)]
    else:
        transform_list += [transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5), 
                                            ]
    trans_totensor = transforms.Compose(transform_list)

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

if image_size is not None:
    trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(image_size),
                                    ])

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        print(torch.amax(img_tensor), torch.amin(img_tensor), img_tensor.shape, '++++++')

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        if image_size is None:
            if args.pad_H > 0 and args.pad_W > 0:
                img = img.crop((0, 0, img.size[0]-args.pad_W, img.size[1]-args.pad_H))
            img.save(rgb_path)
        else:
            trans_rgb(img).save(rgb_path)


        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)
        
        print('--OUTPUT--', output.shape)

        if args.task == 'depth':
            #output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0, 1)
            output = output.detach().cpu().numpy()[0]
            if args.pad_H > 0 and args.pad_W > 0:
                output = output[:(output.shape[0]-args.pad_H), :(output.shape[1]-args.pad_W)]
                print('->', output.shape)
            np.save(save_path.replace('.png', '.npy'), output)
            
            #output = 1 - output
#             output = standardize_depth_map(output)
            plt.imsave(save_path, output.squeeze(),cmap='viridis')

        else:
            output = output.detach().cpu().numpy()[0]
            if args.pad_H > 0 and args.pad_W > 0:
                # print('-->', output.shape)
                output = output[:, :(output.shape[1]-args.pad_H), :(output.shape[2]-args.pad_W)]
                print('-->', output.shape, output.dtype, np.amax(output), np.amin(output))
            np.save(save_path.replace('.png', '.npy'), output)
            output = torch.from_numpy(output)
            trans_topil(output).save(save_path)
            print('--normal-->', torch.amax(output), torch.amin(output))
            
        print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in tqdm(glob.glob(args.img_path+'/*.png')):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!"+str(img_path))
    sys.exit()
