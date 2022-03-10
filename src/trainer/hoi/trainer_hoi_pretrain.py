import os
import argparse
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import deepspeed
from tensorboardX import SummaryWriter

from src.utils.dist import master_process
from src.utils.logger import LOGGER


class Trainer():
    def __init__(self, args, config, model, optimizer, scheduler, criterion, 
                    dataloader_train, rt_dataloader_val=None, mlm_dataloader_val=None):
        self.config = config
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = config.TRAINING.total_epochs
        self.dataloader_train = dataset_train
        self.rt_dataloader_val = rt_dataloader_val
        self.mlm_dataloader_val = mlm_dataloader_val

        self.args = args

        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion

        if master_process(self.args):
            self.summary_writer = SummaryWriter(log_dir=os.path.join(args.blob_mount_dir,config.TRAINING.save_dir,'tb_log'))

    def _checkpoint(self, PATH, ckpt_id, epoch, global_step):
        """Utility function for checkpointing model + optimizer dictionaries
        The main purpose for this is to be able to resume training from that instant again
        """
        checkpoint_state_dict = {
            'epoch': epoch,
            'global_step': global_step,
        }
        save_dir = os.path.join(PATH, 'checkpoint')
        success = self.model.save_checkpoint(save_dir, ckpt_id, checkpoint_state_dict)
        status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(save_dir, ckpt_id)
        if success:
            LOGGER.info(f"Success {status_msg}")
        else:
            LOGGER.warning(f"Failure {status_msg}")

    def _save_model(self, PATH, epoch, step):    
        save_dir = os.path.join(PATH, 'saved_model', 'epoch_{0:03d}_step_{1:05d}'.format(epoch, step))
        self.model.save_fp16_model(save_dir)

    def _resume(self, PATH, tag=None):
        save_dir = os.path.join(PATH, "checkpoint")
        LOGGER.info(f"resume from {save_dir}")