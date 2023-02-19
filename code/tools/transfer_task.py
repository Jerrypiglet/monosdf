'''
cluster:

root@tb-rui-3-769d446887-69wnc:/ruidata/monosdf# scp mm1:/home/ruizhu/Documents/Projects/monosdf/code/tools/transfer_task.py . && python transfer_task.py --exps 20230206-233133
'''

from pathlib import Path
import argparse
from utils.utils_misc import red
import os

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action="store_true", help='')
    parser.add_argument('--exps', nargs='+', help='list of exps to transfer', required=True, default=[])
    parser.add_argument('--exps_root', default='/ruidata/monosdf/exps', type=str, help='')
    parser.add_argument('--exps_root_local', default='/home/ruizhu/Documents/Projects/monosdf/exps', type=str, help='')
    opt = parser.parse_args()

    exps_root = '/ruidata/monosdf/exps'
    if opt.exps_root:
        exps_root = opt.exps_root
    exp_paths = Path(exps_root).iterdir()
    exps_root_local = '/home/ruizhu/Documents/Projects/monosdf/exps'
    if opt.exps_root_local:
        exps_root_local = opt.exps_root_local

    # task_list = [_ for _ in exp_paths if _.stem.startswith(opt.task_name)]
    valid_exp_list = []
    for exp_path in exp_paths:
        for exp in opt.exps:
            if exp_path.stem.startswith(exp):
                valid_exp_list.append(exp_path)
    if len(valid_exp_list) == 0:
        print(red('No task found: '+opt.task_name))
        print(valid_exp_list)
        exit()
    
    print('Tranfering %d tasks...'%len(valid_exp_list))

    file_list = [
        'runconf.conf', 
        'checkpoints/ModelParameters/latest.pth', 
        'checkpoints/OptimizerParameters/latest.pth', 
        'checkpoints/SchedulerParameters/latest.pth', 
        'plots/exps.ply', 
    ]

    for valid_exp in valid_exp_list:
        exp = valid_exp.stem
        print(valid_exp)
        for file in file_list:
            src_file = Path(valid_exp) / file
            assert src_file.exists(), src_file
            dest_file = Path(exps_root_local) / valid_exp.stem / file
            cmd1 = 'ssh mm1 mkdir -p %s'%str(dest_file.parent)
            cmd2 = 'scp -r %s mm1:%s'%(str(src_file), str(dest_file))
            print(cmd1)
            if not opt.debug:
                os.system(cmd1)
            print(cmd2)
            if not opt.debug:
                os.system(cmd2)

