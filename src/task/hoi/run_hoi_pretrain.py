'''
import datetime
import json
import time
from pathlib import Path
import numpy as np
import datasets
from util.dictionary import build_dictionary
import util.misc as utils
from dataset import build_dataloader, build_concat_dataloader, get_coco_api_from_dataset
from engine import build_evaluator, train_one_epoch, pretrain_one_epoch
# from models import build_model
from model_zoo import build_all_model
from util.resume_model import resume_model
from util.optimizer import prep_optimizer
from timm.utils import NativeScaler
from functools import partial
from torch.cuda.amp import autocast as autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")
'''
import os
import random
import argparse
import deepspeed
import torch
import torch.distributed as dist
from mmcv import Config
from src.models.pretrain.HOIPretrainModel import HOIPretrainModel
from src.utils.logger import LOGGER, add_log_to_file
from src.utils.misc import mkdirp, set_random_seed
from src.utils.dist import master_process
from src.optimization.optimizer import build_optimizer_parameters
from src.optimization.lr_scheduler import build_scheduler
from src.dataset.dataloader import build_dataloader
from src.trainer.trainer_hoi_pretrain import Trainer
import pdb


def main():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config', default='./src/configs/pretrain/hoi_pretrain.yaml')
    parser.add_argument('--deepspeed_sparse_attention',action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--fp16', action='store_true', help='enable fp16')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--distributed',action='store_true')
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
    
    model =  HOIPretrainModel(args, config)
    
    if config.WEIGHTS.model_weight != '':
        LOGGER.info(f"Loading model weights from {config.WEIGHTS.model_weight}")
        load_model_weights_with_mismatch(model, os.path.join(config.WEIGHTS.model_weight))

    parameter_group = build_optimizer_parameters(config, model)
    #dbname = config.DATA.DATASET_train.type
    # init deepspeed
    if args.distributed:
        model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                             model=model,
                                                             model_parameters=parameter_group)
    LOGGER.info(f'Training with {dist.get_world_size()} gpus')
    if args.fp16:
        LOGGER.info('Enable fp16 Training')
        fp16 = model_engine.fp16_enabled()

    dataset_trains, dataset_vals, dataloader_trains, dataloader_vals = build_dataloader(args, config)

    dataloader_train = dataloader_trains['PretrainDataset']
    steps_per_epoch = len(dataloader_train)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)
    
    #criterion = build_loss_func(config)
    criterion = None
    trainer = Trainer(args, config, model_engine, optimizer, scheduler, criterion, 
                      dataloader_trains, dataloader_vals['VisualGenome-val'])

    LOGGER.info('start first evaluate')
    
    # if config.stage==2:
    #     trainer.evaluate(dataloader_vals['PreTrainDataset-val'], stage=2)
    # trainer.evaluate(dataloader_vals['ActNetRetDataset-val'], stage=1)


    trainer.train(args.resume)


if __name__ == '__main__':
    deepspeed.init_distributed()
    main()

