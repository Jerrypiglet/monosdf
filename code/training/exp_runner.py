import sys

# sys.path.append('../code')
import os, sys, inspect
from pathlib import Path
pwd_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, pwd_path)
sys.path.insert(0, str(Path(pwd_path).parent.absolute()))

import argparse
import torch

import os
from training.monosdf_train import MonoSDFTrainRunner
import datetime
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--append', type=str, default='')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument("--load_from", type=str, default='', help='load from previos ckpt')
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--resume', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run; otherwise start a new run (which can be restored from a previous run indicated via --load_from')
    parser.add_argument('--if_overfit_train', default=False, action="store_true", help='If set, change val dataset to train split')

    # parser.add_argument('--timestamp', default='latest', type=str,
    #                     help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=str, default='', help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument('--cancel_train', default=False, action="store_true", help='If set, cancel training')
    
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel', default=0)
    parser.add_argument("--datetime_str_input", type=str, required=False, help='', default='')
    parser.add_argument('--is_distributed', default=False, action="store_true", help='')
    parser.add_argument('--if_cluster', default=False, action="store_true", help='')

    opt = parser.parse_args()

    '''
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    '''
    gpu = opt.local_rank


    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    if_distributed = world_size > 1 and opt.is_distributed
    if if_distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
        torch.distributed.barrier()


    trainrunner = MonoSDFTrainRunner(opt=opt, conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    # expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    # is_continue=opt.is_continue,
                                    if_distributed=if_distributed, 
                                    # timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis
                                    )

    trainrunner.run()
