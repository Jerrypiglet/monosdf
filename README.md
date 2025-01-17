<p align="center">

  <h1 align="center">MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction</h1>
  <p align="center">
    <a href="https://niujinshuchong.github.io/">Zehao Yu</a>
    ·
    <a href="https://pengsongyou.github.io/">Songyou Peng</a>
    ·
    <a href="https://m-niemeyer.github.io/">Michael Niemeyer</a>
    ·
    <a href="https://tsattler.github.io/">Torsten Sattler</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>

  </p>
  <h2 align="center">NeurIPS 2022</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2206.00665">Paper</a> | <a href="https://niujinshuchong.github.io/monosdf/">Project Page</a> | <a href="https://autonomousvision.github.io/sdfstudio/">SDFStudio</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We demonstrate that state-of-the-art depth and normal cues extracted from monocular images are complementary to reconstruction cues and hence significantly improve the performance of implicit surface reconstruction methods. 
</p>
<br>

# [Obsolete]

pytorch=1.8.0

Distributed training: 

``` bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 --master_port 47769  ...
```

## data preparation
### [1/2] dump from OpenRooms_RAW_loader

``` bash
(or-py310) ruizhu@ubuntu:~/Documents/Projects/OpenRooms_RAW_loader$ python convert_mitsubaScene3D_to_monosdf.py 
(or-py310) ruizhu@ubuntu:~/Documents/Projects/OpenRooms_RAW_loader$ python convert_openroomsScene3D_to_monosdf.py 
```

### [2/2] extract estimated geometry from omnidata
`` use newest torch and omnidata installation (conda env: monosdf-py38); otherwise fails``

`NEW: batch extract:'

``` bash
(monosdf-py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/preprocess$ python batch_extract.py --gpu_ids 0 1 2 4 5 6 7 --gpu_total 7
```

`extract for single scene:'

``` bash
(monosdf-py38)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Set {TASK} to depth, normal in 2 runs

``` bash
(monosdf-py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/preprocess$ python extract_monocular_cues.py --task {TASK} --img_path ../data/kitchen/trainval/image --output_path ../data/kitchen/trainval --omnidata_path /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch --pretrained_models /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch/pretrained_models/

(monosdf-py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/preprocess$ python extract_monocular_cues.py --task {TASK} --img_path ../data/public_re_3_v3pose_2048-main_xml-scene0008_00_morerescaledSDR/scan1/image --output_path ../data/public_re_3_v3pose_2048-main_xml-scene0008_00_morerescaledSDR/scan1 --omnidata_path /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch --pretrained_models /home/ruizhu/Documents/Projects/omnidata/omnidata_tools/torch/pretrained_models/

```

## [scannet]

``` bash
(py38) ruizhu@ubuntu:~/Documents/Projects/monosdf/code$ 

conda activate py38

python training/exp_runner.py --conf confs/scannet_mlp.conf --scan_id scan1

python evaluation/eval.py --conf confs/scannet_mlp.conf --scan_id scan1 --resolution 512 --eval_rendering --evals_folder ../pretrained_results --checkpoint ../pretrained_models/scannet_mlp/scan1.pth
```

## [kitchen-train]

scans:
- 'scan1': train split; 202 frames
- 'scan2': val split; 10 frames
- 'trainval': train+val split; 212 frames

**(GT geometry; PNG input)**

``` bash
python training/exp_runner.py --conf confs/kitchen_trainval_mlp.conf --scan_id trainval

python evaluation/eval.py --conf confs/kitchen_mlp.conf --scan_id trainval --resolution 512 --eval_rendering --evals_folder ../eval_results/kitchen_train_png_gt --checkpoint ../exps/kitchen_gt_train_mlp_1/2023_01_18_00_01_24/checkpoints/ModelParameters/latest.pth
```

**(GT geometry; HDR input)**

``` bash
python training/exp_runner.py --conf confs/kitchen_hdr_gt_mlp.conf --scan_id trainval

python evaluation/eval.py --conf confs/kitchen_hdr_gt_mlp.conf --scan_id trainval --resolution 512 --eval_rendering --evals_folder ../eval_results/kitchen_train_HDR_GT --checkpoint ../exps/kitchen_HDR_GT_train_mlp_1/2023_01_18_22_58_00/checkpoints/ModelParameters/latest.pth
```

**(EST geometry; HDR input)**

``` bash
python training/exp_runner.py --conf confs/kitchen_hdr_est_mlp.conf --scan_id trainval
python training/exp_runner.py --conf confs/kitchen_hdr_est_mlp.conf --scan_id trainval --expname _gamma2_L2loss
[+] python training/exp_runner.py --conf confs/kitchen_hdr_est_mlp.conf --scan_id trainval --expname _gamma2_L2loss_4xreg_lr1e-4_decay25
[+] python training/exp_runner.py --conf confs/kitchen_HDR_grids.conf --scan_id trainval --expname _gamma2
[++] python training/exp_runner.py --conf confs/kitchen_HDR_grids.conf --scan_id trainval --expname _gamma2_randomPixel_L2loss_4xreg_lr5e-4_decay25
[++] python training/exp_runner.py --conf confs/kitchen_hdr_est_mlp.conf --scan_id trainval --expname _gamma2_randomPixel_L2loss_4xreg_lr5e-4_decay25

python evaluation/eval.py --conf confs/kitchen_hdr_est_mlp.conf --scan_id trainval --resolution 512 --eval_rendering --evals_folder ../eval_results/kitchen_train_HDR_EST --checkpoint ../exps/kitchen_HDR_EST_train_mlp_1/2023_01_18_22_10_44/checkpoints/ModelParameters/latest.pth
```

### other options
- **gamma in rgb loss**: ``loss{if_gamma_loss = True``

## [openrooms]

**(GT geometry; PNG input)**

``` bash
python training/exp_runner.py --conf confs/openrooms_mlp.conf --scan_id scan1

python evaluation/eval.py --conf confs/openrooms_mlp.conf --scan_id scan1 --resolution 512 --eval_rendering --evals_folder ../eval_results/openrooms_png_gt --checkpoint ../exps/public_re_3_v3pose_2048-main_xml-scene0008_00_more_gt_train_mlp_1/2023_01_18_01_30_24/checkpoints/ModelParameters/latest.pth
```

**(GT geometry; HDR input)**

Config:

``model -> rendering_network -> if_hdr = True``
``dataset -> if_hdr = True``

``` bash
python training/exp_runner.py --conf confs/openrooms_hdr_gt_mlp.conf --scan_id scan1

python evaluation/eval.py --conf confs/openrooms_hdr_gt_mlp.conf --scan_id scan1 --resolution 512 --eval_rendering --evals_folder ../eval_results/openrooms_HDR_GT --checkpoint ../exps/public_re_3_v3pose_2048-main_xml-scene0008_00_more_HDR_GT_train_mlp_1/2023_01_18_21_58_00/checkpoints/ModelParameters/latest.pth
```

**(EST geometry; HDR input)**

``` bash
python training/exp_runner.py --conf confs/openrooms_hdr_est_mlp.conf --scan_id scan1

[+] python training/exp_runner.py --conf confs/openrooms_hdr_est_mlp.conf --scan_id scan1 --expname _gamma2_L2loss_4xreg_lr1e-4_decay25

python evaluation/eval.py --conf confs/openrooms_hdr_est_mlp.conf --scan_id scan1 --resolution 512 --eval_rendering --evals_folder ../eval_results/openrooms_HDR_EST --checkpoint ../exps/public_re_3_v3pose_2048-main_xml-scene0008_00_more_HDR_EST_train_mlp_1/2023_01_18_21_57_14/checkpoints/ModelParameters/latest.pth
```

## EVAL

``` bash
CUDA_VISIBLE_DEVICES=0 python training/exp_runner.py --conf confs/kitchen_HDR_grids.conf --scan_id trainval --expname _EVALTRAIN2023_01_23_21_23_38 --resume_folder kitchen_HDR_grids_gamma2_randomPixel_fixedDepthHDR_trainval/2023_01_23_21_23_38 --is_continue --if_overfit_train --cancel_train
```

## cleaning up
### remove exps
- add task names to `clean_up_tasks.txt` (e.g. kitchen_HDR_gridstmpppppp_trainval)
- `python clean_up_tasks.py`
### remove nautilus tasks
- add task names to `clean_up_tasks.txt` (e.g. zz-torch-job-gpu20230125-005454-qzhdp or 20230125-005454)
- `./cluster_control$ python rui_tool.py delete --all`

## TODO
- [] change training to handle rays instead of batchsize=1: change to random batch of rays
- [x] add datetime to taskname from rui_tool; instead of add when launching
- [] better eval commands
- [] override options in cmd
# Update
MonoSDF is integrated to [SDFStudio](https://github.com/autonomousvision/sdfstudio), where monocular depth and normal cues can be applied to [UniSurf](https://github.com/autonomousvision/unisurf/tree/main/model) and [NeuS](https://github.com/Totoro97/NeuS/tree/main/models). Please check it out.

# Setup

## Installation
Clone the repository and create an anaconda environment called monosdf using
```
git clone git@github.com:autonomousvision/monosdf.git
cd monosdf

conda create -y -n monosdf python=3.8
conda activate monosdf

conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt
```
The hash encoder will be compiled on the fly when running the code.

## Dataset
For downloading the preprocessed data, run the following script. The data for the DTU, Replica, Tanks and Temples is adapted from [VolSDF](https://github.com/lioryariv/volsdf), [Nice-SLAM](https://github.com/cvg/nice-slam), and [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet), respectively.
```
bash scripts/download_dataset.sh
```
# Training

Run the following command to train monosdf:
```
cd ./code
python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf CONFIG  --scan_id SCAN_ID
```
where CONFIG is the config file in `code/confs`, and SCAN_ID is the id of the scene to reconstruct.

We provide example commands for training DTU, ScanNet, and Replica dataset as follows:
```
# DTU scan65
python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/dtu_mlp_3views.conf  --scan_id scan65

# ScanNet scan 1 (scene_0050_00)
python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_mlp.conf --scan_id scan1

# Replica scan 1 (room0)
python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/replica_mlp.conf --scan_id scan1
```

We created individual config file on Tanks and Temples dataset so you don't need to set the scan_id. Run training on the courtroom scene as:
```
python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/tnt_mlp_1.conf
```

We also generated high resolution monocular cues on the courtroom scene and it's better to train with more gpus. First download the dataset
```
bash scripts/download_highres_TNT.sh
```

Then run training with 8 gpus:
```
CUDA_VISIBLE_DEVICES1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/tnt_highres_grids_courtroom.conf
```
Of course, you can also train on all other scenes with multi-gpus.

# Evaluations

## DTU
First, download the ground truth DTU point clouds:
```
bash scripts/download_dtu_ground_truth.sh
```
then you can evaluate the quality of extracted meshes (take scan 65 for example):
```
python evaluate_single_scene.py --input_mesh scan65_mesh.ply --scan_id scan65 --output_dir dtu_scan65
```

We also provide script for evaluating all DTU scenes:
```
python evaluate.py
```
Evaluation results will be saved to ```evaluation/DTU.csv``` by default, please check the script for more details.

## Replica
Evaluate on one scene (take scan 1 room0 for example)
```
cd replica_eval
python evaluate_single_scene.py --input_mesh replica_scan1_mesh.ply --scan_id scan1 --output_dir replica_scan1
```

We also provided script for evaluating all Replica scenes:
```
cd replica_eval
python evaluate.py
```
please check the script for more details.

## ScanNet
```
cd scannet_eval
python evaluate.py
```
please check the script for more details.

## Tanks and Temples
You need to submit the reconstruction results to the [official evaluation server](https://www.tanksandtemples.org), please follow their guidance. We also provide an example of our submission [here](https://drive.google.com/file/d/1Cr-UVTaAgDk52qhVd880Dd8uF74CzpcB/view?usp=sharing) for reference.

# Custom dataset
We provide an example of how to train monosdf on custom data (Apartment scene from nice-slam). First, download the dataset and run the script to subsample training images, normalize camera poses, and etc.
```
bash scripts/download_apartment.sh 
cd preprocess
python nice_slam_apartment_to_monosdf.py
```

Then, we can extract monocular depths and normals (please install [omnidata model](https://github.com/EPFL-VILAB/omnidata) before running the command):
```
python extract_monocular_cues.py --task depth --img_path ../data/Apartment/scan1/image --output_path ../data/Apartment/scan1 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
python extract_monocular_cues.py --task normal --img_path ../data/Apartment/scan1/image --output_path ../data/Apartment/scan1 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
```

Finally, we train monosdf as
```
python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/nice_slam_grids.conf
```

# Pretrained Models
First download the pretrained models with
```
bash scripts/download_pretrained.sh
```
Then you can run inference with (DTU for example)
```
cd code
python evaluation/eval.py --conf confs/dtu_mlp_3views.conf --checkpoint ../pretrained_models/dtu_3views_mlp/scan65.pth --scan_id scan65 --resolution 512 --eval_rendering --evals_folder ../pretrained_results
```

You can also run the following script to extract all the meshes:
```
python scripts/extract_all_meshes_from_pretrained_models.py
```

# High-resolution Cues
Here we privode script to generate high-resolution cues, and training with high-resolution cues. Please refer to our supplementary for more details.

First you need to download the Tanks and Temples dataset from [here](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and unzip it to ```data/tanksandtemples```. Then you can run the script to create overlapped patches 
```
cd preprocess
python generate_high_res_map.py --mode create_patches
```

and run the Omnidata model to predict monocular cues for each patch 
```
python extract_monocular_cues.py --task depth --img_path ./highres_tmp/scan1/image/ --output_path ./highres_tmp/scan1 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
python extract_monocular_cues.py --task depth --img_path ./highres_tmp/scan1/image/ --output_path ./highres_tmp/scan1 --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
```
This step will take a long time (~2 hours) since there are many patches and the model only use a batch size of 1. 

Then run the script again to merge the output of Omnidata.
```
python generate_high_res_map.py --mode merge_patches
```

Now you can train the model with
```
CUDA_VISIBLE_DEVICES1,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/tnt_highres_grids_courtroom.conf
```

Please note that the script for generating high-resolution cues only works for the Tanks and Temples dataset. You need to adapt it if you want to apply to other dataset.

# Acknowledgements
This project is built upon [VolSDF](https://github.com/lioryariv/volsdf). We use pretrained [Omnidata](https://omnidata.vision) for monocular depth and normal extraction. Cuda implementation of Multi-Resolution hash encoding is based on [torch-ngp](https://github.com/ashawkey/torch-ngp). Evaluation scripts for DTU, Replica, and ScanNet are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python), [Nice-SLAM](https://github.com/cvg/nice-slam) and [manhattan-sdf](https://github.com/zju3dv/manhattan_sdf) respectively. We thank all the authors for their great work and repos. 


# Citation
If you find our code or paper useful, please cite
```bibtex
@article{Yu2022MonoSDF,
  author    = {Yu, Zehao and Peng, Songyou and Niemeyer, Michael and Sattler, Torsten and Geiger, Andreas},
  title     = {MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction},
  journal   = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
}
```
