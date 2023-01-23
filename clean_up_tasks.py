from pathlib import Path
from tqdm import tqdm
from icecream import ic
import shutil

list_path = 'clean_up_tasks.txt'
folders = ['exps']

with open(list_path) as f:
    task_list = f.read().splitlines() 
task_list = [x.strip() for x in task_list]
task_list_parent = [_.split('/')[0] for _ in task_list]
task_list_parent_has_sub = [_.split('/')[0] for _ in task_list if len(_.split('/'))>1]

for folder in folders:
    log_paths = Path(folder).iterdir()
    for log_path in log_paths:
        log_name = log_path.name
        # if log_name != '20230110-023750-rad_kitchen_190-10_specT': continue
        # print(log_name)
        if_remove = False
        if log_name.startswith('tmp') or log_name.endswith('-tmp') or log_name.endswith('-tmp_new'):
            shutil.rmtree(log_path, ignore_errors=True)
            print('Removed '+str(log_path))
            continue
        if log_name in task_list_parent or '-'.join(log_name.split('-')[:2]) in task_list_parent:
            if log_name in task_list_parent_has_sub and folder in ['log']:
                log_paths_sub = (Path(folder)/log_name).iterdir()
                for log_path_sub in log_paths_sub:
                    log_name_sub = log_path_sub.name
                    if log_name+'/'+log_name_sub in task_list:
                        shutil.rmtree(str(Path(log_path)/log_name_sub), ignore_errors=True)
                        print('Removed '+str(Path(log_path)/log_name_sub))
                        continue
            else:
                shutil.rmtree(log_path, ignore_errors=True)
                print('Removed '+str(log_path))
                continue
