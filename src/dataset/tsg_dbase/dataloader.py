import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from src.utils.logger import LOGGER
from torch.utils.data.dataloader import default_collate
from .actnet import ActNetDataset
#from .charades_dataset import CharadesDataset
import pdb


def collate_fn(batch):
    batch_word_vectors = [b["word_vectors"] for b in batch]
    batch_txt_mask = [b["txt_mask"] for b in batch]
    batch_vis_input = [b["visual_input"] for b in batch]
    batch_word_vectors = nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True)
    batch_txt_mask = nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True)
    batch_vis_input = nn.utils.rnn.pad_sequence(batch_vis_input, batch_first=True)

    batch_map_gt = default_collate([b['map_gt'] for b in batch])
    batch_anno_idxs = default_collate([b['anno_idx'] for b in batch])
    batch_duration = default_collate([b['duration'] for b in batch])

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][0,:num_clips,:num_clips] = map_gt

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_word_vectors': batch_word_vectors,
        'batch_txt_mask': batch_txt_mask,
        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': batch_vis_input,
        'batch_duration': batch_duration,
    }

    return batch_data


def build_dataset(args, config, split='train'):
    dataset_dict = config.DATA["DATASET_"+split]  # train,val,trainval,test
    name = dataset_dict['name']
    metadata_path=os.path.join(dataset_dict['metadata_path'])
    video_path=os.path.join(args.blob_mount_dir, dataset_dict['video_path'])
    #import pdb;pdb.set_trace()
    dataset = globals()[dataset_dict['type']](config, split, metadata_path, video_path)
    LOGGER.info(f'build dataset: {name}, {len(dataset)}')
    return dataset


def build_dataloader(args, config):
    dataset_train = build_dataset(args, config, split='train')
    dataset_val = build_dataset(args, config, split='test')

    sampler_train, sampler_val = None, None

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        LOGGER.info(f'using dist training, build sampler')
        
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
                
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE_per_gpu,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn = collate_fn,
        drop_last=True,
    )
    
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE_per_gpu,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        collate_fn = collate_fn,
        drop_last=True
    )

    LOGGER.info(f'build dataloader done!')

    return dataset_train, dataset_val, dataloader_train, dataloader_val

