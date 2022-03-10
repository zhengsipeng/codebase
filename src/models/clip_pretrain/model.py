from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from model_zoo.hoi_pretrained_model import clip



class PretrainModel(nn.Module):
    def __init__(self, args):
        super(PretrainModel, self).__init__()

        self.clip_model, _ = clip.load(args.backbone, device=args.device)
        self.clip_model.float()
        self.dropout = nn.Dropout(args.dropout)
        self.device = torch.device(args.device)
        self.args = args
        self.init_weights()
        self.cnt = 0

    def init_weights(self,):
        for name, p in self.named_parameters():
            do_not_init = ['backbone', 'bert_model', 'clip_model', 'clip_teacher']
            init_flag = True
            if p.dim() <= 1:
                init_flag = False
            for do_not_init_name in do_not_init:
                if do_not_init_name in name:
                    init_flag = False
                    break
            if init_flag:
                nn.init.xavier_uniform_(p)
    
    def copy_cross_weights(self,):
        state_dict = self.state_dict().copy()
        for key, val in state_dict.items():
            if key == "clip_model.positional_embedding":
                state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                continue
            if key.find("clip_model.transformer.resblocks") == 0:
                num_layer = int(key.split(".")[3])
                # cut from beginning
                if num_layer < self.args.num_layer:
                    state_dict[key.replace("clip_model", "cross")] = val.clone()
                    continue
        self.load_state_dict(state_dict)
    
    def forward(self):
        