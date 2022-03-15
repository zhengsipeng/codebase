import os
import time
import re
import sys
import argparse

print("#######################################DIR CHECK!#########################################")
platform = 'aml'
parser = argparse.ArgumentParser('K8S training script', add_help=False)
parser.add_argument('--blob_mount_dir', type=str,default="", help="The working directory.")
# parser.add_argument('--git_key', type=str,default="", help="The git key.")
parser.add_argument('--command', type=str,default="./tools/train.py", help="The command to run.")
parser.add_argument('--branch',default="main",type=str,help="github branch")
args = parser.parse_args()


def ompi_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK'))

git_key = os.environ.get('GITKEY')

def get_master_ip():
    regexp = "[\s\S]*export[\s]*DLTS_SD_worker0_IP=([0-9.]+)[\s|s]*"

    with open("/dlts-runtime/env/init.env", 'r') as f:
        line = f.read()
    print("^^^^^^^^", line)
    match = re.match(regexp, line)
    if match:
        ip = str(match.group(1))
        print("master node ip is " + ip)
        return ip

# os.system("ifconfig")
if os.environ.get('AZ_BATCH_MASTER_NODE', None) is None and platform != 'aml':
    os.environ.setdefault("AZ_BATCH_MASTER_NODE", get_master_ip())

# print("---------------------------", os.environ['AZ_BATCH_MASTER_NODE'])


def ompi_local_size():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE'))

work_dir = os.environ['HOME']
os.chdir(work_dir)
print(work_dir)
# print("current dir is {}:".format(os.getcwd()))
print("ompi_rank: {}, ompi_local_size: {}".format(ompi_rank(), ompi_local_size()))

if ompi_rank() % ompi_local_size() == 0:
    os.system("ls -l")
    # print("*************os.environ: {}*****************".format(os.environ))
    os.system("nvidia-smi")
    # print("*************os.enviro: {}\n*****************".format(os.environ))
    if os.path.isfile(os.path.join(os.environ['HOME'], 'gitdone.txt')):
        print("gitdone file exists!!!!!!!!")
        os.system("rm $HOME/gitdone.txt")

    if os.path.isfile(os.path.join(os.environ['HOME'], 'done.txt')):
        print("done file exists!!!!!!!!")
        os.system("rm $HOME/done.txt")

    if os.path.isdir('codebase'):
        print("codebase exist!!!!!!")
        # print("current dir {} !!!!!!!!!".format(os.getcwd()))
        os.system("ls -l")
        os.system("rm -rf codebase")

    os.system("git clone -b {} https://{}@github.com/zhengsipeng/codebase.git".format(args.branch, git_key))
    os.system('echo done > $HOME/gitdone.txt')
    print("********* finish git clone {}**************".format(ompi_rank()))
else:
    while not os.path.exists(os.path.join(os.environ['HOME'], 'gitdone.txt')):
        print("wait for git clone")
        time.sleep(10)

os.chdir('./codebase')


os.system("pwd")
os.system("ls -l")
time.sleep(30)

print("################################INSTALL ADDITIONAL REQURIEMENTS! ####################################")
os.system('pip install lmdb')

print("################################START TO RUN COMMAND! ####################################")

# os.system(". setup.sh")
# os.system('/bin/bash -c ". setup.sh"')
os.environ['PYTHONPATH']=os.getcwd()
print(os.environ['PYTHONPATH'])

print("start command: ", args.command + f" --blob_mount_dir {args.blob_mount_dir}")

os.system(args.command + f" --blob_mount_dir {args.blob_mount_dir}")

# os.system("horovodrun -np 8 python src/pretrain/run_pretrain.py \
#     --config src/configs/pretrain_indomain_base_resnet50_mlm_itm.json \
#     --output_dir {}".format(save_dir))


if ompi_rank() % ompi_local_size() == 0:
    os.chdir(os.environ['HOME'])
    os.system("rm -rf LFVideo")
    os.system("rm done.txt")
    os.system("rm gitdone.txt")