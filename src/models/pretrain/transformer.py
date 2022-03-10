# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from model_zoo.pix2seq.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from model_zoo.pix2seq.transformer import _get_activation_fn
from heapq import heappush, heappop
import pdb


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, drop_path=0.1,
                 activation="relu", normalize_before=False, pass_pos_and_query=True,
                 args=None):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, drop_path, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, drop_path, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()

        self.num_vocal = args.dictionary.num_vocal
        self.num_bins = args.dictionary.num_bins
        self.num_max_objs = args.num_max_objs

        self.vocal_classifier = nn.Linear(d_model, self.num_vocal)
        self.det_embed = nn.Embedding(1, d_model)  # cls token
        # In raw pix2seq, the vocab size is self.num_vocal-2 which doesn't include "end" and "noise"
        #self.vocal_embed = nn.Embedding(self.num_vocal - 2, d_model)  # valcabulary
        self.vocal_embed = nn.Embedding(self.num_vocal, d_model)
        
        self.pred_eos = args.pred_eos
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
    
        self.num_classes = args.dictionary.num_classes
        self.manual_mask = args.manual_mask
        if self.manual_mask:
            self.logit_masks = self._get_logit_masks()
        
        self.sampler = args.sampler
        self.beam_width = args.beam_width
        if args.query_pos:
            self.query_pos = nn.Embedding(self.num_max_objs * 5 + 1, d_model)
        else:
            self.query_pos = None

        if args.classifier_norm:
            self.classifier_norm = nn.LayerNorm(d_model)
        else:
            self.classifier_norm = None

        self.drop_cls = args.drop_cls
        self.eval_p = args.eval_p
        self.eos_bias = args.eos_bias

        if args.use_cls_token:
            self.V_CLS = nn.Embedding(1, d_model)
            self.T_CLS = nn.Embedding(1, d_model)
        else:
            self.V_CLS, self.T_CLS = None, None
        print(f"Build with return intermediate dec: {args.return_intermediate_dec}, query_pos: {args.query_pos}, drop_cls: {args.drop_cls}")
        print("Eval with", self.eval_p)

    def _get_logit_masks(self):
        mask1 = torch.cat([torch.ones(self.num_bins+1), torch.zeros(self.num_classes+2)])  # pos mask
        mask2 = torch.cat([torch.zeros(self.num_bins+1), torch.ones(self.num_classes), torch.zeros(2)])
        return torch.stack([mask1, mask2])  # 2, 2094

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def seq_if_objs_available(self, input_seq):
        raise NotImplementedError

    def forward(self, 
            src=None, 
            mask=None, 
            pos_embed=None, 
            text_memory_resized=None, 
            text_attention_mask=None,
            encoded_text=None
        ):
        """
        Args:
            src: shape[B, C, H, W]
            input_seq: shape[B, 500, C] for training and shape[B, 1, C] for inference
            mask: shape[B, H, W]
            pos_embed: shape[B, C, H, W]
        """
        # flatten NxCxHxW to HWxNxC
        
        bs = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        device = src.device

        if self.V_CLS is not None:
            # We add a CLS token to the image
            V_CLS = self.V_CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
            # Add the CLS token to the incoming features
            src = torch.cat((V_CLS, src))
            # Adding zeros as the first token in the sequence to be compatible with the CLS token
            pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))
            # Adding one mask item to the beginning of the mask to be compatible with CLS token
            cls_pad = torch.zeros(bs, 1).bool().to(device)
            mask = torch.cat((cls_pad, mask), dim=1)
        if self.T_CLS is not None:
            T_CLS = self.T_CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
            text_memory_resized = torch.cat((T_CLS, text_memory_resized))
            cls_pad = torch.zeros(bs, 1).bool().to(device)
            text_attention_mask = torch.cat((cls_pad, text_attention_mask), dim=1)

        # Concat on the sequence dimension
        visual_text_src = torch.cat([src, text_memory_resized], dim=0)
        # For mask, sequence dimension is second
        mask = torch.cat([mask, text_attention_mask], dim=1)
        # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
        
        img_text_memory = self.encoder(visual_text_src, src_key_padding_mask=mask, pos=pos_embed)
        img_memory = img_text_memory[:len(src)]
        text_memory = img_text_memory[-len(text_memory_resized):]
        assert img_memory.shape[1] == text_memory.shape[1]

        memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory,
                "img_memory": img_memory,
                "text_pooled_op": encoded_text.pooler_output if self.T_CLS is not None else None,
                "img_pooled_op": img_memory[0] if self.V_CLS is not None else None,  # Return the CLS token
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
            }
        return memory_cache



def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        drop_path=args.drop_path,
        normalize_before=args.pre_norm,
        args=args
    )

