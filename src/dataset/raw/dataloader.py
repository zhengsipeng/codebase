import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import ConcatDataset
from transformers import RobertaTokenizerFast
from src.dataset.init_transform import init_transform_dict
from src.dataset.pretrain_dataset import PretrainDataset
from src.utils.logger import LOGGER
import src.dataset.transforms as T
import pdb


class VideoCollector(object):
    """is_train is kept here if we want to remove
    the randomness during validation of MLM accuracy.
    In that case, instantiate two VideoCollector"""
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def collate_batch(self, batch):
        video_frames = default_collate([d["video_frames"] for d in batch])
        video_chunk = default_collate([d["video_chunk"] for d in batch])
        text_chunk = default_collate([d["text_chunk"] for d in batch])
        attention_mask = default_collate([d["attention_mask"] for d in batch])
        segment = default_collate([d["segment"] for d in batch])
        
        # group data
        text_ids = default_collate([d["text_ids"] for d in batch]) # (B, L)
        if self.mlm:
            text_ids, mlm_labels = mask_batch_text_tokens(
                text_ids, self.tokenizer)  # make mlm data
        else:
            text_ids, mlm_labels = text_ids, None
     
        if 'loc_label' in batch[0]:
            labels = default_collate([d["loc_label"] for d in batch])
        else:
            labels = None
        
        if 'next_text_ids' in batch[0]:
            next_text_ids = default_collate([d["next_text_ids"] for d in batch])
            next_attention_mask = default_collate([d["next_attention_mask"] for d in batch])
        else:
            next_text_ids, next_attention_mask = None, None

        if 'dense_video_frames' in batch[0]:
            dense_video_frames = default_collate([d["dense_video_frames"] for d in batch])
            dense_video_index = default_collate([d["dense_video_index"] for d in batch])
        else:
            dense_video_frames,dense_video_index = None, None

        return dict(
                video_frames=video_frames, # C, N, H, W
                video_chunk=video_chunk, # N, num_clips
                text_ids=text_ids, # Seq_len
                mlm_labels=mlm_labels,
                text_chunk=text_chunk,
                attention_mask=attention_mask,
                segment=segment,
                labels=labels,
                next_text_ids=next_text_ids,
                next_attention_mask=next_attention_mask,
                dense_video_frames=dense_video_frames,
                dense_video_index=dense_video_index
                )


def build_dataset(args, config, tokenizer, split='train'):
    transform=init_transform_dict(config)[split]
    
    dataset_dicts = config.DATA.DATASET_train if split=='train' else config.DATA.DATASET_val
    if isinstance(dataset_dicts, dict):
        dataset_dicts = [dataset_dicts]
    datasets = {}
    for dataset_dict in dataset_dicts:
        name = dataset_dict['name']
        metadata_dir=os.path.join(dataset_dict['metadata_dir'])
        image_path=os.path.join(dataset_dict['image_path'])
        
        dataset = globals()[dataset_dict['type']](config,
                                name,
                                metadata_dir,
                                image_path,
                                tokenizer=tokenizer,
                                transform=transform)
  
        LOGGER.info(f'build dataset: {name}, {len(dataset)}')

        datasets[name] = dataset

    return datasets


def build_dataloader(args, config):
    # build tokenizer
    if "roberta" in config.TextEncoder.model:
        from transformers import RobertaModel
        tokenizer = RobertaTokenizerFast.from_pretrained(config.TextEncoder.model)
    else:
        raise NotImplementedError

    dataset_trains = build_dataset(args, config, tokenizer, split='train')
    dataset_vals = build_dataset(args, config, tokenizer, split='val')
    
    if len(dataset_trains) > 1:
        dataset_trains["PretrainDataset"] = ConcatDataset([v for k, v in dataset_trains.items()])
   
    data_collator = VideoCollector(tokenizer, 
                                     mlm=config.TRAINING.use_mlm,
                                     mlm_probability=0.15)

    sampler_train, sampler_val = None, None

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        LOGGER.info(f'using dist training, build sampler')
        
    data_loader_trains = {}
    for k, dataset_train in dataset_trains.items():
        if args.distributed:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE_per_gpu,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            collate_fn = data_collator.collate_batch,
            drop_last=True,
        )

        data_loader_trains[k] = data_loader_train

    data_loader_vals = {}
    for k, dataset_val in dataset_vals.items():
        if args.distributed:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                    dataset_val, shuffle=False)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE_per_gpu,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            collate_fn = data_collator.collate_batch,
            drop_last=True
        )
        data_loader_vals[k] = data_loader_val

    LOGGER.info(f'build dataloader done!')
    LOGGER.info(f'dataloader_train: {len(data_loader_train)}')

    for k, v in data_loader_vals.items():
        LOGGER.info(f'data_loader_val {k}: {len(v)}')

    return dataset_trains, dataset_vals, data_loader_trains, data_loader_vals

