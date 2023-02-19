'''
(base) root@rui-deployment-mono2-bb6bf9cdb-7pm9w:/ruidata/monosdf/exps# scp -r mm1:/home/ruizhu/Documents/Projects/monosdf/transfer_* . && python transfer_exps.py --task 20230207-225908
'''
import os
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False, action="store_true", help='')
parser.add_argument('--task', default='', type=str, help='')
opt = parser.parse_args()

list_path = 'transfer_exps.txt'
src_path = '/ruidata/monosdf/exps'
dest_path = 'Documents/Projects/monosdf/exps'

if opt.task == '':
    with open(list_path) as f:
        task_list = f.read().splitlines() 
    task_list = [x.strip() for x in task_list]
else:
    task_list = [opt.task]

exps_all = [_ for _ in os.listdir(src_path)]
for task in task_list:
    exps = [_ for _ in exps_all if _.startswith(task)]
    if len(exps) == 0:
        print('No exact match for [%s] found among:'%task, os.listdir(src_path))
        raise RuntimeError
    elif len(exps) > 1:
        print('More than one matches for [%s] found; resuming from 1st one:'%task)
        print(exps)
    copy_from_task = exps[0]

    pth_path = Path(src_path) / copy_from_task / 'checkpoints/ModelParameters/latest.pth'
    assert pth_path.exists(), str(pth_path)

    ckpt_path_dest = Path(dest_path) / copy_from_task / 'checkpoints/ModelParameters'
    mkdir_cmd = 'ssh mm1 \'mkdir -p %s\''%str(ckpt_path_dest)
    os.system(mkdir_cmd)

    pth_path_dest = Path(dest_path) / copy_from_task / 'checkpoints/ModelParameters/latest.pth'
    copy_pth_cmd = 'rsync %s mm1:%s'%(str(pth_path), str(pth_path_dest))
    os.system(copy_pth_cmd)

    conf_path = Path(src_path) / copy_from_task / 'runconf.conf'
    assert conf_path.exists(), str(conf_path)
    conf_path_dest = Path(dest_path) / copy_from_task / 'runconf.conf'
    copy_conf_cmd = 'rsync %s mm1:%s'%(str(conf_path), str(pth_path_dest))
    os.system(copy_conf_cmd)

    plots_path_dest = Path(dest_path) / copy_from_task / 'plots'
    mkdir_cmd = 'ssh mm1 \'mkdir -p %s\''%str(plots_path_dest)
    os.system(mkdir_cmd)

    ply_path_src = Path(src_path) / copy_from_task / 'plots/*.ply'
    copy_ply_cmd = 'scp -r %s mm1:%s'%(str(ply_path_src), str(plots_path_dest))
    os.system(copy_ply_cmd)




