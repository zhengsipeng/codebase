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
from src.models.tsg.eval import eval_predictions
from src.trainer.basic_trainer import BasicTrainer
from src.utils.dist import concat_all_gather
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
                batch_vis_input = batch['batch_vis_input'].to(self.local_rank).float()
                batch_duration = batch['batch_duration'].to(self.local_rank)
                if self.args.fp16:
                    batch_word_vectors = batch_word_vectors.half()
                    batch_vis_input = batch_vis_input.half()
                
                prediction, map_mask = self.model(batch_word_vectors, batch_txt_mask, batch_vis_input)
                loss, joint_prob = self.criterion(prediction, map_mask, batch_map_gt)
                self.model.backward(loss)
                if self.config.TRAINING.CLIP_GRAD:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.model.step()
                self.global_step += 1
                self.scheduler.step_update(self.global_step)
                
                lr = self.scheduler._get_lr(self.global_step)[0] if self.config.TRAINING.LR_SCHEDULER.NAME != "ReduceLROnPlateau" \
                        else self.optimizer.state_dict()["param_groups"][0]["lr"]
                #import pdb; pdb.set_trace()
                self.report_step_metrics(lr, loss.item())

                #if self.global_step % self.config.TRAINING.eval_step == 0:
                #    self.evaluate(self.dataloader_val, stage=2)
                if self.global_step % self.config.TRAINING.checkpoint_step == 0:
                    self._checkpoint(self.config.TRAINING.save_dir, self.global_step, epoch, self.global_step)
                if self.global_step % self.config.TRAINING.save_step == 0:
                    self._save_model(self.config.TRAINING.save_dir, epoch, step)

                break
            
            self.start_epoch = epoch
            LOGGER.info(epoch)
            self.evaluate(self.dataloader_val, stage=2)

    def evaluate(self, dataloader_val, stage=2):
        LOGGER.info("start evaluate")
        self.model.eval()
        st = time.time()

        all_prediction = []
        all_anno_idxs = []
        for step, batch in enumerate(dataloader_val):
            batch_anno_idxs = batch['batch_anno_idxs'].to(self.local_rank)
            batch_word_vectors = batch['batch_word_vectors'].to(self.local_rank)
            batch_txt_mask = batch['batch_txt_mask'].to(self.local_rank)
            batch_map_gt = batch['batch_map_gt'].to(self.local_rank)
            batch_vis_input = batch['batch_vis_input'].to(self.local_rank).float()
            batch_duration = batch['batch_duration'].to(self.local_rank)
            if self.args.fp16:
                batch_word_vectors = batch_word_vectors.half()
                batch_vis_input = batch_vis_input.half()

            prediction, map_mask = self.model(batch_word_vectors, batch_txt_mask, batch_vis_input)
            loss, joint_prob = self.criterion(prediction, map_mask, batch_map_gt)

            joint_prob = concat_all_gather(joint_prob)
            batch_anno_idxs = concat_all_gather(batch_anno_idxs)
            #min_idx = min(batch_anno_idxs)
            all_prediction.extend(joint_prob)
            all_anno_idxs.extend(batch_anno_idxs)
            break
        all_prediction = torch.cat(all_prediction, dim=0)
        all_anno_idxs = torch.stack(all_anno_idxs, dim=0)
        all_prediction = all_prediction[all_anno_idxs]

        annotations = dataloader_val.dataset.annotations
        eval_result, miou = eval_predictions(all_prediction.cpu().numpy(), annotations, self.config.TEST, verbose=False)
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
