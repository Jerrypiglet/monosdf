'''
cluster: 
    (base) root@rui-deployment-mono2-bb6bf9cdb-4fzj5:/ruidata/monosdf# scp -r mm1:/home/ruizhu/Documents/Projects/monosdf/code/tools/clean_funcs.py .
    python clean_funcs.py --debug --exps_root exps
'''

from pathlib import Path
import argparse
from utils.utils_misc import red

def remove_ckpt(checkpoints_path, N_ckpts, debug=False):
    checkpoints_path = Path(checkpoints_path)
    if not checkpoints_path.exists():
        print(red('checkpoints_path does not exist: '+str(checkpoints_path)))
        return

    for ckpt_folder in ['OptimizerParameters', 'ModelParameters', 'SchedulerParameters']:
        ckpt_path = checkpoints_path / ckpt_folder
        if ckpt_path.exists():
            all_stems = [int(ckpt_file.stem) for ckpt_file in ckpt_path.iterdir() if ckpt_file.stem!='latest']
            all_stems.sort()
            for stem in all_stems[::-1][N_ckpts:]:
                ckpt_file = ckpt_path / (str(stem)+'.pth')
                if not debug:
                    if ckpt_file.exists():
                        ckpt_file.unlink()
                print('Removed ckpt file: '+str(ckpt_file))
        else:
            print(red('ckpt_path does not exist: '+str(ckpt_path)))


def remove_plots(plots_path, N_plots, debug=False):
    plots_path = Path(plots_path)
    if not plots_path.exists():
        print(red('plots_path does not exist: '+str(plots_path)))
        return

    ply_file_list = [plot_file for plot_file in plots_path.iterdir() if (plot_file.suffix.endswith('ply') and 'epoch' in plot_file.stem)]
    try:
        all_epochs = [int(ply_file.stem.split('epoch')[1]) for ply_file in ply_file_list]
    except IndexError:
        import ipdb; ipdb.set_trace()
    all_epochs.sort()

    ply_file_list_delete = []
    for epoch in all_epochs[::-1][N_plots:]:
        ply_file_list_delete += [ply_file for ply_file in ply_file_list if int(ply_file.stem.split('epoch')[1])==epoch]
    for ply_file in ply_file_list_delete:
        if not debug:
            if ply_file.exists():
                ply_file.unlink()
        print('Removed ply file: '+str(ply_file))

    png_file_list = [plot_file for plot_file in plots_path.iterdir() if plot_file.suffix.endswith('png')]
    png_file_list_delete = []
    all_epochs = [int(png_file.stem.split('epoch')[1]) if 'epoch' in png_file.stem else int(png_file.stem.split('_')[1]) for png_file in png_file_list]
    for epoch in all_epochs[::-1][N_plots:]:
        png_file_list_delete += [png_file for png_file in png_file_list if int(png_file.stem.split('_')[1])==epoch]
    for png_file in png_file_list_delete:
        if not debug:
            if png_file.exists():
                png_file.unlink()
        print('Removed png file: '+str(png_file))

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action="store_true", help='')
    parser.add_argument('--N_ckpts', default=3, type=int, help='keep last N ckpts')
    parser.add_argument('--N_plots', default=3, type=int, help='keep last N plots (each from one epoch)')
    parser.add_argument('--exps_root', default='', type=str, help='')
    opt = parser.parse_args()

    exps_root = '/home/ruizhu/Documents/Projects/monosdf/exps'
    if opt.exps_root:
        exps_root = opt.exps_root
    exp_paths = Path(exps_root).iterdir()
    run_paths_list = []

    for exp_path in exp_paths:
        
        for run_path in Path(exp_path).iterdir():
            if run_path.is_file():
                continue
            if run_path.stem in ['checkpoints', 'plots']:
                run_paths_list.append(exp_path)
                continue
            run_paths_list.append(run_path)
    
    run_paths_list = list(set(run_paths_list))

    for run_path in run_paths_list:
        # print(run_path.stem, run_path)

        # checkpoints
        checkpoints_path = Path(run_path) / 'checkpoints'
        remove_ckpt(checkpoints_path, opt.N_ckpts, opt.debug)

        # plots
        plots_path = Path(run_path) / 'plots'
        remove_plots(plots_path, opt.N_plots, opt.debug)