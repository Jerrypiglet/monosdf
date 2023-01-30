from pathlib import Path
from tqdm import tqdm
from icecream import ic
import shutil

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False, action="store_true", help='')
opt = parser.parse_args()

list_path = 'clean_up_tasks.txt'
folders = ['exps']

with open(list_path) as f:
    task_list = f.read().splitlines() 
task_list = [x.strip() for x in task_list]
task_list_parent = [_.split('/')[0] for _ in task_list]
task_list_parent_has_sub = [_.split('/')[0] for _ in task_list if len(_.split('/'))>1]
task_list_dates = [_.replace('zz-torch-job-gpu', '').replace('zz-iris-job-gpu', '')[:-6] for _ in task_list if _.startswith('zz-torch-job-gpu') or _.startswith('zz-iris-job-gpu')] + [_ for _ in task_list if _.startswith('202')]

# opt.debug = True
# opt.debug = False

for folder in folders:
    log_paths = Path(folder).iterdir()
    for log_path in log_paths:
        log_name = log_path.name
        # if log_name != '20230110-023750-rad_kitchen_190-10_specT': continue
        if_remove = False
        if log_name.startswith('tmp') or log_name.endswith('-tmp') or log_name.endswith('-tmp_new') or log_name.split('_')[-2].endswith('tmp') or any(log_name.startswith(_) for _ in task_list_dates):
            if not opt.debug:
                shutil.rmtree(log_path, ignore_errors=True)
                print('Removed '+str(log_path))
            else:
                print('Remove '+str(log_path), '?')
            continue
        if log_name in task_list_parent or '-'.join(log_name.split('-')[:2]) in task_list_parent:
            if log_name in task_list_parent_has_sub and folder in ['exps']:
                log_paths_sub = (Path(folder)/log_name).iterdir()
                for log_path_sub in log_paths_sub:
                    log_name_sub = log_path_sub.name
                    if log_name+'/'+log_name_sub in task_list:
                        if not opt.debug:
                            shutil.rmtree(str(Path(log_path)/log_name_sub), ignore_errors=True)
                            print('Removed '+str(Path(log_path)/log_name_sub))
                        else:
                            print('Remove '+str(Path(log_path)/log_name_sub), '?')
                        continue
            else:
                if not opt.debug:
                    shutil.rmtree(log_path, ignore_errors=True)
                    print('Removed '+str(log_path))
                else:
                    print('Remove '+str(log_path), '?')
                continue
