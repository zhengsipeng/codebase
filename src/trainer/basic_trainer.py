import os
from pathlib import Path
from src.utils.logger import LOGGER
from src.utils.dist import master_process
from tensorboardX import SummaryWriter


class BasicTrainer():
    def __init__(self, args, config, model, optimizer, scheduler, criterion, 
                dataloader_train, dataloader_val):
        self.config = config
        self.local_rank = model.local_rank
        self.global_step = 0
        self.start_epoch = 0
        self.total_epochs = config.TRAINING.EPOCHS
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.args = args

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        if master_process(self.args):
            self.summary_writer = SummaryWriter(
                log_dir=os.path.join(args.blob_mount_dir, config.TRAINING.save_dir, 'tb_log'))

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
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 
                                'saved_model', 
                                'epoch_{0:03d}_step_{1:05d}'.format(epoch, step))
        self.model.save_fp16_model(save_dir)
    
    def _resume(self,PATH, tag=None):
        save_dir = os.path.join(self.args.blob_mount_dir,PATH, 'checkpoint')
        LOGGER.info(f"resume from {save_dir}")
        _, checkpoint_state_dict = self.model.load_checkpoint(save_dir)
        self.start_epoch = checkpoint_state_dict['epoch']
        self.global_step = checkpoint_state_dict['global_step']
        del checkpoint_state_dict
    
    def report_step_metrics(self, lr, loss):
        ##### Record the LR against global_step on tensorboard #####
        if master_process(self.args):
            self.summary_writer.add_scalar(f'Train/lr', lr, self.global_step)
            self.summary_writer.add_scalar(f'Train/train_loss', loss, self.global_step)
        ##### Recording  done. #####
        if self.global_step % self.config.TRAINING.print_step == 0:
            LOGGER.info('training_progress: step={}, loss={}, lr={}'.
                format(self.global_step, loss, lr))
    
    