import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import torch
import deepspeed
from torch.utils.data import DataLoader
import torch.distributed as dist
from src.trainer.basic_trainer import BasicTrainer
from src.utils.logger import LOGGER


class Trainer(BasicTrainer):
    def __init__(self, args, config, model, optimizer, scheduler, criterion, 
                dataloader_train, dataloader_val):
        super(Trainer, self).__init__(args, config, model, optimizer, scheduler, criterion, 
                dataloader_train, dataloader_val)
        
    def train(self, resume):
        self.model.train()
        if resume:
            self._resume(self.config.TRAINING.save_dir)
            LOGGER.info(f'resume from {self.start_epoch}, global step {self.global_step}')
            steps_trained_in_current_epoch = self.global_step % len(self.dataloader_train)
        else:
            steps_trained_in_current_epoch = 0
        LOGGER.info(f'begin training from {self.start_epoch}')

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            if self.args.distributed:
                self.dataloader_train.sampler.set_epoch(epoch)
            
            step = 0
            for batch in self.dataloader_train:
                # Skip past any already trained steps if resuming training
                if step < steps_trained_in_current_epoch:
                    step += 1
                    continue
                batch_anno_idxs = batch['batch_anno_idxs'].to(self.local_rank)
                batch_word_vectors = batch['batch_word_vectors'].to(self.local_rank)
                batch_txt_mask = batch['batch_txt_mask'].to(self.local_rank)
                batch_map_gt = batch['batch_map_gt'].to(self.local_rank)
                batch_vis_input = batch['batch_vis_input'].to(self.local_rank)
                batch_duration = batch['batch_duration'].to(self.local_rank)
                if self.args.fp16:
                    batch_word_vectors = batch_word_vectors.half()
                    batch_vis_input = batch_vis_input.half()
                
                output = self.model(batch_word_vectors, batch_txt_mask, batch_vis_input)
                import pdb; pdb.set_trace()