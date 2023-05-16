<!--Generate the TOC via: -->
<!-- (bash) ../gh-md-toc --insert README_indoor_synthetic.md-->
<!--See https://github.com/ekalinin/github-markdown-toc#readme-->

<!--ts-->
# Customized MonoSDF repo for FIPT

This is a work in progress, for preview of the [FITP project](https://jerrypiglet.github.io/fipt-ucsd/). Check later for more updates. Also check https://github.com/autonomousvision/monosdf for original repo.

Tested with pytorch=1.8.0

<!-- Distributed training: 

``` bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 --master_port 47769  ...
``` -->

## Data preparation
### Dump from rui-indoorinv-data loaders

Check [rui-indoorinv-data](https://github.com/Jerrypiglet/rui-indoorinv-data/blob/fipt/README_indoor_synthetic.md#export-to-monosdf-format) repo for instructions of exporting scenes to MonoSDF format. Copy the dumped scene files to `data/` folder, e.g.

<!-- - data/indoor_synthetic/kitchen_mi
  - scale_mat.npy # scale and offset to normalize all camera poses
  - cameras.npz
  - Image
    - 000_0001.exr
    - 000_0001.png
  - ImMask
    - 000_0001.png
  - K_list.txt
  - MiDepth/MiNormalGlobal/MiNormalGlobal_OVERLAY
    - # depth by rasterizing with GT geometry and poses
    - 000_0001.npy
    - 000_0001.png -->

<!-- https://tree.nathanfriend.io -->
``` 
.
└── data/indoor_synthetic/kitchen_mi/
    ├── scale_mat.npy # scale and offset to normalize all camera poses
    ├── cameras.npz
    ├── Image/
    │   ├── 000_0001.exr
    │   └── 000_0001.png
    ├── ImMask/
    │   └── 000_0001.png
    ├── K_list.txt
    └── MiDepth/MiNormalGlobal/MiNormalGlobal_OVERLAY/
        ├── # depth by rasterizing with GT geometry and poses
        ├── 000_0001.npy
        └── 000_0001.png
```

### Extract estimated geometry from omnidata

`` use newest torch and omnidata installation (conda env: monosdf-py38); otherwise fails``

<!-- `NEW: batch extract:' -->

Add the scene name to `scene_list = [...`. Then run: 

``` bash
(monosdf-py38) monosdf/preprocess$ python batch_extract.py --gpu_ids 0 1 2 --gpu_total 3
```

<!-- `extract for single scene:'

``` bash
(monosdf-py38)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Set {TASK} to depth, normal in 2 runs

``` bash
(monosdf-py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/preprocess$ python extract_monocular_cues.py --task {TASK} --img_path ../data/kitchen/trainval/image --output_path ../data/kitchen/trainval --omnidata_path /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch --pretrained_models /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch/pretrained_models/

(monosdf-py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/preprocess$ python extract_monocular_cues.py --task {TASK} --img_path ../data/public_re_3_v3pose_2048-main_xml-scene0008_00_morerescaledSDR/scan1/image --output_path ../data/public_re_3_v3pose_2048-main_xml-scene0008_00_morerescaledSDR/scan1 --omnidata_path /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch --pretrained_models /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch/pretrained_models/
``` -->

## Training
  
``` bash
(monosdf-py38) monosdf/code$ CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/mi.conf --conf_add confs/mi_kitchen.conf --append _tmp --prefix DATE-
```

Launch Tensorboard at `exps` to monitor logs and visualizations.