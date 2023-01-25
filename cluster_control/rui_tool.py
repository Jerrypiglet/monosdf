#!/usr/bin/env python

# Example:
## Create:
# python rui_tool.py create -f rui_torch_job_create.yaml -s 'python -m torch.distributed.launch --master_port 5324 --nproc_per_node=4 train_combine_v3_RCNNOnly_bbox.py --num_layers 3 --pointnet_camH --pointnet_camH_refine --pointnet_personH_refine --loss_last_layer --accu_model --task_name DATE_pod_BASELINEv4_detachcamParamsExceptCamHinFit_lossLastLayer_NEWDataV5_SmallerPersonBins_YcLargeBins5_DETACHinput_plateau750_cascadeL3-V0INPUT-SmallerPERSONBins1-190_lr1e-5_w360-10_human175STD15W05 --config maskrcnn/coco_config_small_synBN1108.yaml --weight_SUN360=10. SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 16 SOLVER.PERSON_WEIGHT 0.05 SOLVER.BASE_LR 1e-5 MODEL.HUMAN.MEAN 1.75 MODEL.HUMAN.STD 0.15 MODEL.RCNN_WEIGHT_BACKBONE 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug MODEL.RCNN_WEIGHT_CLS_HEAD 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug'
## Delete:
# python rui_tool.py delete 'z-torch-job-4-20200129' --all
## Sync:
# python rui_tool.py sync sum

import argparse
# from datetime import date
from datetime import datetime
import yaml
import subprocess
import os, sys
import pprint
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Kubectl Helper')
    subparsers = parser.add_subparsers(dest='command', help='commands')
    
    delete_parser = subparsers.add_parser('delete', help='Delete a batch of jobs')
    delete_parser.add_argument('-p', '--pattern', type=str, help='The pattern to delete', default='')
    delete_parser.add_argument('-f', '--file', type=str, help='file containing list of tast names to be deleted', default='../clean_up_tasks.txt')
    delete_parser.add_argument('-a', '--all', action='store_true', help='If delete all (should be true)')
    delete_parser.add_argument('-n', '--namespace', type=str, help='namespace')
    delete_parser.add_argument('--debug', action='store_true', help='if debugging')

    sync_parser = subparsers.add_parser('sync', help='sync to rclone storage')
    sync_parser.add_argument('option', type=str, help='The pattern to delete')
    sync_parser.add_argument('-a', '--all', action='store_true', help='If delete all (should be true)')
    sync_parser.add_argument('-n', '--namespace', type=str, help='namespace')
    sync_parser.add_argument('--debug', action='store_true', help='if debugging')

    tb_parser = subparsers.add_parser('tb', help='Delete a batch of jobs')
    # tb_parser.add_argument('-n', '--namespace', type=str, help='namespace')
    tb_parser.add_argument('--debug', action='store_true', help='if debugging')
    tb_parser.add_argument('-f', '--file', type=str, help='Path to template file', default='')
    tb_parser.add_argument('--cpur', type=int, help='request of CPUs', default=1)
    tb_parser.add_argument('--cpul', type=int, help='limit of CPUs', default=4)
    tb_parser.add_argument('--memr', type=int, help='request of memory in Gi', default=12)
    tb_parser.add_argument('--meml', type=int, help='limit of memory in Gi', default=20)
    tb_parser.add_argument('--namespace', type=str, help='namespace of the job', default='mc-lab')


    # tb_parser.add_argument('--logs_path', type=str, help='python path in pod', default='/root/miniconda3/bin/python')

    create_parser = subparsers.add_parser('create', help='Create a batch of jobs')
    create_parser.add_argument('-f', '--file', type=str, help='POD''ath to template file')
    create_parser.add_argument('-s', '--string', type=str, help='Input command')
    create_parser.add_argument('-d', '--deploy', action='store_true', help='deploy the code')
    create_parser.add_argument('-z', '--zip', action='store_true', help='deploy the code')
    create_parser.add_argument('--resume', type=str, help='resume_from: e.g. 20201129-232627', default='NoCkpt')
    create_parser.add_argument('--command_str_pre', type=str, help='', default='export PATH=/usr/local/cuda/bin:$PATH; ')
    create_parser.add_argument('--deploy_src', type=str, help='deploy to target path', default='~/Documents/Projects/monosdf/code/')
    create_parser.add_argument('--deploy_s3', type=str, help='deploy s3 container', default='s3mm1:train/train')
    create_parser.add_argument('--deploy_tar', type=str, help='deploy to target path', default='/ruidata/monosdf/job_list')
    create_parser.add_argument('--service', type=str, default='ffsend', help='upload services', choices={"transfer", "ffsend", 'mm1', 'rclone'})
    # create_parser.add_argument('--deploy_train_path', type=str, help='deploy to target path', default='/ruidata/monosdf/code')
    create_parser.add_argument('--python_path', type=str, help='python path in pod', default='python')
    create_parser.add_argument('--pip_path', type=str, help='python path in pod', default='pip')
    create_parser.add_argument('--gpus', type=int, help='nubmer of GPUs', default=1)
    create_parser.add_argument('--cpur', type=int, help='request of CPUs', default=8)
    create_parser.add_argument('--cpul', type=int, help='limit of CPUs', default=12)
    create_parser.add_argument('--memr', type=int, help='request of memory in Gi', default=12)
    create_parser.add_argument('--meml', type=int, help='limit of memory in Gi', default=20)
    create_parser.add_argument('--namespace', type=str, help='namespace of the job', default='mc-lab')
    create_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    create_parser.add_argument('-r', '--num-replicas', type=int, help='Number of replicas')
    # create_parser.add_argument('-n', '--namespace', type=str, help='namespace')
    create_parser.add_argument('vals', help='Values to replace', nargs=argparse.REMAINDER)
    create_parser.add_argument('--debug', action='store_true', help='if debugging')

    args = parser.parse_args()
    return args


def iterate_dict(input_dict, var_replace_list=None):
    if not isinstance(input_dict, dict):
        if isinstance(input_dict, list):
            return [iterate_dict(x, var_replace_list=var_replace_list) for x in input_dict]
        else:
            if var_replace_list is not None:
                for var in var_replace_list:
                    # if input_dict == var:
                    #     return var_replace_list[var]
                    # print('------', str(input_dict), var)
                    if var in str(input_dict):
                        print(var, input_dict, '------>', input_dict.replace(str(var), str(var_replace_list[var])))
                        return input_dict.replace(str(var), str(var_replace_list[var]))
            return input_dict
    
    new_dict = {}
    for key in input_dict:
        new_dict.update({key: iterate_dict(input_dict[key], var_replace_list=var_replace_list)})

    return new_dict

def replace_vars(args):
    var_mapping = {'gpus': '#GPUS', 'cpur': '#CPUR', 'cpul': '#CPUL', 'memr': '#MEMR', 'meml': '#MEML', 'namespace': '#NAMESPACE'}
    var_replace_list = {}
    for var in args:
        if var in var_mapping:
            var_replace_list.update({var_mapping[var]: args[var]})
    return var_replace_list

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S")
    return d1

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return loaded

def dump_yaml(yaml_path, yaml_content):
    with open(yaml_path, 'w') as stream:
        try:
            yaml.dump(yaml_content, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)

def run_command(command, namespace=None):
    if namespace is not None:
        command += ' --namespace='+namespace
    ret = subprocess.check_output(command, shell=True)
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret

def run_command_generic(command, debug=False):
    #This command could have multiple commands separated by a new line \n
    # some_command = "export PATH=$PATH://server.sample.mo/app/bin \n customupload abc.txt"

    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()  

    #This makes the wait possible
    p_status = p.wait()

    #This will give you the output of the command being executed
    print("Command output: " + output.decode('utf-8'))
    if debug:
        print('-', output, '-')
        # import ipdb; ipdb.set_trace()

    return output.decode('utf-8').strip()

def get_pods(pattern, namespace=None):
    command = 'kubectl get pods -o custom-columns=:.metadata.name,:.status.succeeded'
    if namespace is not None:
        command += ' --namespace='+namespace
    ret = run_command(command)
    pods = list(filter(None, ret.splitlines()))[1:]
    pods = [pod.split() for pod in pods]
    pods = list(filter(lambda x: re.match(pattern, x[0]), pods))
    pods = [pod[0] for pod in pods]
    if len(pods) == 1:
        pods = str(pods[0])
    # pprint.pprint(pods)
    print(pods)
    return pods

def create_job_from_yaml(yaml_filename):
    # https://stackoverflow.com/questions/4760215/running-shell-command-and-capturing-the-output
    result = subprocess.run('kubectl create -f %s'%yaml_filename, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    stdout = result.stdout.decode('utf-8')
    print('>>>>>>>>>>>> kubectl create %s result:'%yaml_filename)
    print(stdout)

def deploy_to_transfer(args):
    exclude_cmds = '-x \'logs/*\' -x \'Summary_vis/*\' -x \'Checkpoint/*\' -x \'tmp_build/*\' -x \'*.pyc\' -x \'*.ipynb\' -x \'wandb/*\''
    from pathlib import Path
    tmp_file = Path(args.deploy_src) / 'tmp.zip'
    if tmp_file.exists():
        tmp_file.unlink()
        
    if args.service == 'transfer':
        deploy_command = 'cd %s && zip -r tmp.zip * %s && curl --upload-file ./tmp.zip https://transfer.sh/tmp.zip && rm *.zip && cd -'%(args.deploy_src, exclude_cmds)
        print('>>>>>>>>>>>> deploying with...: %s'%deploy_command)
        # os.system(deploy_command)
        return_commands = run_command_generic(deploy_command)
        return_url = 'https' + return_commands.partition('https')[-1].partition('tmp.zip')[0] + 'tmp.zip'
        return_url = return_url.replace('transfer.sh', 'transfer.sh/get')
    elif args.service == 'ffsend': # https://github.com/timvisee/ffsend#linux-all-distributions
        # deploy_command = 'cd %s && zip -r tmp.zip * %s && ffsend upload ./tmp.zip --expiry 24h && rm *.zip && cd -'%(args.deploy_src, exclude_cmds)
        deploy_command = 'cd %s && zip -r tmp.zip * %s'%(args.deploy_src, exclude_cmds)
        print('>>>>>>>>>>>> deploying with...: %s'%deploy_command)
        return_commands = run_command_generic(deploy_command)
        from pathlib import Path
        # print(Path(args.deploy_src) / 'tmp.zip')
        # assert (Path(args.deploy_src) / 'tmp.zip').exists()
        # print('+++', run_command_generic('ffsend upload %s --expiry 24h'%str(Path(args.deploy_src) / 'tmp.zip')))
        # return_url = 'https' + return_commands.partition('https')[-1].partition('/home')[0]
        send_command = 'ffsend upload --downloads 1 --expiry-time 30m %s'%str(Path(args.deploy_src) / 'tmp.zip')
        print(send_command)
        return_url = run_command_generic(send_command, debug=True)
        return_url = return_url.strip()
        # return_url = return_url.replace('transfer.sh', 'transfer.sh/get')
    elif args.service in ['mm1']:
        return_url = ''
    elif args.service in ['rclone']:
        deploy_src ='~/Documents/Projects/DPTSSN/train/'
        deploy_s3 ='s3mm1:train/train_ngc_2022/train_%s'%datetime_str
        deploy_command = 'cd %s; zip -r tmp.zip * %s; '%(args.deploy_src, deploy_src)
        deploy_command += 'rclone mkdir %s && rclone copy -vv --progress --fast-list --checkers=64 --transfers=64 tmp.zip %s/'%(deploy_s3, deploy_s3)
        print('>>>>>>>>>>>> deploying with rclone ...: %s'%deploy_command)
        return_commands = run_command_generic(deploy_command)
        return_url = deploy_s3 + '/tmp.zip'

    print('>>>>>>>>>>>> Uploaded to: %s'%return_url)

    return return_url

def create(args):
    if args.resume != 'NoCkpt':
        datetime_str = args.resume
        tmp_yaml_filaname = 'tasks/%s/tmp_%s.yaml'%(datetime_str, datetime_str)
        print('============ Resuming from YAML file: %s'%tmp_yaml_filaname)
        yaml_content = load_yaml(tmp_yaml_filaname)
        os.system('kubectl delete job '+yaml_content['metadata']['name'])
        print('============ Task removed: %s'%yaml_content['metadata']['name'])
        yaml_content['metadata']['name'] += '-re'
        command_str = yaml_content['spec']['template']['spec']['containers'][0]['args'][0]
        s_split = command_str.split(' ')
        start_index = s_split.index('rclone')
        for i in range(5):
            s_split.pop(start_index)
        insert_index = s_split.index('--if_cluster')
        s_split.insert(insert_index+1, '--reset_latest_ckpt')
        s_split.insert(insert_index+1, '--resume resume')
        command_str = ' '.join(s_split)
        command_str = command_str.replace('&& &&', '&&')
        yaml_content['spec']['template']['spec']['containers'][0]['args'][0] = command_str

        tmp_yaml_filaname = tmp_yaml_filaname.replace('.yaml', '-RE.yaml')
        dump_yaml(tmp_yaml_filaname, yaml_content)
        print('============ YAML file dumped to %s'%tmp_yaml_filaname)
        print(command_str)

    else:
        command_str = args.string
        datetime_str = get_datetime()
        command_str = command_str.replace('DATE', datetime_str)
        print('------------ Command string:')
        print(command_str)

        yaml_content = load_yaml(args.file)
        var_replace_list = replace_vars(vars(args))
        yaml_content = iterate_dict(yaml_content, var_replace_list=var_replace_list)
        print('------------ yaml_content:')
        print(yaml_content)

        command_str = command_str.replace('python', args.python_path)
        command_str += ' --datetime_str %s'%datetime_str
        # if args.deploy:
        #     pass

            # args.deploy_s3 += '-%s'%datetime_str
            # if args.zip:
            #     command_str = 'mkdir %s && rclone --progress copy %s/tmp.zip %s/ && cd %s && unzip tmp.zip && '%(args.deploy_tar, args.deploy_s3, args.deploy_tar, args.deploy_tar) + command_str
            # else:
            #     command_str = 'rclone --progress copy %s %s && cd %s && '%(args.deploy_s3, args.deploy_tar, args.deploy_tar) + command_str
        # else:
        if args.deploy:
            args.deploy_tar += '/code-%s'%datetime_str
            # deploy_to_s3(args)
            transfer_url = deploy_to_transfer(args)
            if args.service == 'transfer':
                download_command = 'wget %s; unzip tmp.zip; '%transfer_url
            elif args.service == 'ffsend': # first, download/install ffsend binary
                download_command = '/ruidata/ffsend download %s; apt install unzip; unzip tmp.zip; '%transfer_url
            elif args.service == 'mm1': # first, download/install ffsend binary
                download_command = 'scp ruizhu@hyperion.ucsd.edu:/home/ruizhu/Documents/Projects/DPTSSN/train/tmp.zip .; unzip tmp.zip; '
            elif args.service == 'rclone': # first, download/install ffsend binary
                # download_command = 'curl https://rclone.org/install.sh | bash; mkdir ~/.config/rclone/; cp /code/rclone/* ~/.config/rclone/; rclone copy --progress --fast-list --checkers=64 --transfers=64 %s/ .'%deploy_s3
                download_command = 'curl https://rclone.org/install.sh | bash; mkdir ~/.config/rclone/; cp /code/rclone/* ~/.config/rclone/; '
                deploy_s3 ='s3mm1:train/train_ngc_2022/train_%s'%datetime_str
                download_command += 'rclone copy --progress --fast-list --checkers=64 --transfers=64 %s/tmp.zip .; unzip tmp.zip; '%deploy_s3
        else:
            download_command = ''

        # command_str = args.command_str_pre
        command_str = args.command_str_pre + 'mkdir %s; cd %s; '%(args.deploy_tar, args.deploy_tar) + download_command + command_str
        command_str = command_str.replace('pip', args.pip_path)
        # command_str = 'tensorboard --logdir . --port 6010'
            
        yaml_content['spec']['template']['spec']['containers'][0]['args'][0] += command_str
        yaml_content['metadata']['name'] += datetime_str
        tmp_yaml_filaname = 'yamls/tmp_%s.yaml'%datetime_str
        dump_yaml(tmp_yaml_filaname, yaml_content)
        print('============ YAML file dumped to %s'%tmp_yaml_filaname)

    create_job_from_yaml(tmp_yaml_filaname)

    if args.resume == 'NoCkpt':
        task_dir = './tasks/%s'%datetime_str
        os.mkdir(task_dir)
        os.system('cp %s %s/'%(tmp_yaml_filaname, task_dir))
        text_file = open(task_dir + "/command.txt", "w")
        n = text_file.write(command_str)
        text_file.close()
        print('yaml and command file saved to %s'%task_dir)

    # os.remove(tmp_yaml_filaname)
    # print('========= REMOVED YAML file %s'%tmp_yaml_filaname)

    pod_name = get_pods(yaml_content['metadata']['name'])
    
    if pod_name and args.resume == 'NoCkpt':
        with open("all_commands.txt", "a+") as f:
            f.write("%s-%s\n" % (pod_name, datetime_str))
            f.write("%s\n" % command_str)

def delete(args, delete_all=False, answer=None):
    if args.namespace:
        namespace = args.namespace
    else:
        namespace = None
    ret = run_command('kubectl get jobs -o custom-columns=:.metadata.name,:.status.succeeded', namespace)
    jobs = list(filter(None, ret.splitlines()))[1:]
    jobs = [job.split() for job in jobs]
    if args.debug:
        print('Got jobs: ', jobs, pattern)

    pattern = args.pattern
    list_path = args.file
    assert pattern != '' or list_path != ''

    if list_path != '':
        with open(list_path) as f:
            task_list = f.read().splitlines() 
        task_list = [x.strip() for x in task_list]
        task_list = [_ for _ in task_list if _.startswith('zz-torch-job-gpu')]
        task_list = [_[:-6] for _ in task_list]
        jobs = [_ for _ in jobs if _[0] in task_list]
    elif pattern != '':
        print('Trying to delete pattern %s...'%pattern)
        if args.all or delete_all:
            jobs = list(filter(lambda x: re.match(pattern, x[0]), jobs))
        else:
            jobs = list(filter(lambda x: re.match(pattern, x[0]) and x[1] == '1', jobs))
    # pprint.pprint(jobs)

    # if debug:
    print('Filtered jobs:', jobs)
    if len(jobs) == 0:
        return
    while True:
        if answer is None:
            answer = input('Do you want to delete those jobs?[y/n]')
        if answer == 'y':
            job_names = [x[0] for x in jobs]
            ret = run_command('kubectl delete jobs ' + ' '.join(job_names), namespace)
            if isinstance(ret, bytes):
                ret = ret.decode()
            print(ret)
            break
        elif answer == 'n':
            break

def sync(args):
    option = args.option
    # assert option in ['sum', 'vis', 'ckpt'], 'Sync options must be in (sum, vis, ckpt)!'
    # option_to_pattern_dict = {'sum': 'z-job-syncsum', 'vis': 'z-job-syncvis', 'ckpt': 'z-job-syncckpt'}
    # option_to_yaml_dict = {'sum': 'rui_torch_job_syncSum.yaml', 'vis': 'rui_torch_job_syncVis.yaml', 'ckpt': 'rui_torch_job_syncCkpt.yaml'}

    assert option in ['exps'], 'Sync options must be in (sum, vis, ckpt)!'
    option_to_pattern_dict = {'exps': 'z-job-syncsum'}
    option_to_yaml_dict = {'sum': 'rui_torch_job_syncSum.yaml', 'vis': 'rui_torch_job_syncVis.yaml', 'ckpt': 'rui_torch_job_syncCkpt.yaml'}

    pattern = option_to_pattern_dict[option]
    delete(args, pattern=pattern, answer='y', delete_all=True)

    create_job_from_yaml(option_to_yaml_dict[option])

def tb(args):
    yaml_content = load_yaml(args.file)
    var_replace_list = replace_vars(vars(args))
    yaml_content = iterate_dict(yaml_content, var_replace_list=var_replace_list)
    print('------------ yaml_content:')
    print(yaml_content)
    tmp_yaml_filaname = 'tb_job.yaml'
    dump_yaml(tmp_yaml_filaname, yaml_content)
    print('============ YAML file dumped to %s'%tmp_yaml_filaname)

    create_job_from_yaml(tmp_yaml_filaname)

def main():
    args = parse_args()

    if args.command == 'delete':
        delete(args)
    elif args.command == 'create':
        create(args)
    elif args.command == 'sync':
        sync(args)
    elif args.command == 'tb':
        tb(args)

if __name__ == "__main__":
    main()