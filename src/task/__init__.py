import os
import random
import argparse
import deepspeed
import torch
import torch.distributed as dist
from mmcv import Config
from src.utils.logger import LOGGER, add_log_to_file
from src.utils.dist import master_process
from src.utils.misc import mkdirp, set_random_seed
from src.optimization.optimizer import build_optimizer_parameters
from src.optimization.lr_scheduler import build_scheduler
from src.utils.load import load_model_weights_with_mismatch



def set_parser_config():
    parser = argparse.ArgumentParser('Set parser', add_help=False)
    parser.add_argument('--config', default='./src/configs/pretrain/hoi_pretrain.yaml')
    parser.add_argument('--blob_mount_dir', default=".", type=str)  # /blob_mount
    parser.add_argument('--deepspeed_sparse_attention', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--fp16', action='store_true', help='enable fp16')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    config = Config.fromfile(args.config)
    
    LOGGER.info(config)
    LOGGER.info(args)

    if not master_process(args):
        LOGGER.disabled = True
    if master_process(args):
        mkdirp(os.path.join(config.TRAINING.save_dir,"log"))
        add_log_to_file(os.path.join(config.TRAINING.save_dir,"log/log.txt"))

    return args, config