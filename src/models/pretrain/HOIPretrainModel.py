import math
import torch
import torch.nn as nn
import itertools
#from .transformer import build_transformer
from transformers import RobertaTokenizerFast
from src.models.module.image import build_image_encoder
from src.models.module.language import build_text_encoder
from src.models.module import build_cross_encoder
import pdb


class HOIPretrainModel(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(self, args, config):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_loss: If true, perform image-text contrastive learning
            contrastive_align_loss: If true, perform box - token contrastive learning
            qa_dataset: If not None, train a QA head for the target dataset (CLEVR or GQA)
            split_qa_heads: If true, use several head for each question type
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        backbone,
        transformer,
        d_model=512,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_loss=False,
        contrastive_align_loss=False,
        split_qa_heads=True,
        predict_final=False,
        freeze_text_encoder=False,
        text_encoder_type="roberta-base",
        """
        super().__init__()
        self.img_encoder = build_image_encoder(config)
        self.text_encoder = build_text_encoder(config)
        self.cross_encoder = build_cross_encoder(config)
        self.hidden_dim = config.Model.hidden_dim
        self.expander_dropout = 0.1
        self.resizer = FeatureResizer(input_feat_size=config.Model.hidden_dim, 
                                      output_feat_size=self.hidden_dim,
                                      dropout=self.expander_dropout)

        self.num_obj_classes = config.HOI.pretrain_obj_classes
        self.num_verb_classes = config.HOI.pretrain_verb_classes

        
        '''
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        
        config = self.text_encoder.config
        '''

    def get_bbox_masks(self, bboxes, labels, H, W, device):
        bsz = len(bboxes)
        mask_dicts = []
        for i in range(bsz):
            box_dict = dict()
            _bboxes = bboxes[i]
            _labels = labels[i]
            num_box = 0
            for j in _labels.cpu():
                clsid = j.item()
                if clsid not in box_dict:
                    box_dict[clsid] = []
                box_dict[clsid].append(_bboxes[num_box])
                num_box += 1
            # 1: [x,y,w,h]; 2: ...
            mask_dict = dict()
            # mask sure there are person's bbox and there aren't background's bbox
            assert 80 not in box_dict and 0 in box_dict
            for clsid, boxes in box_dict.items():
                # we generate mask map for each class in the image; class 80 denotes "background"
                ft_map = torch.ones((H, W)).to(device)
                for box in boxes:
                    x1, y1 = (box[0] - box[2]/2).cpu().item(), (box[1] - box[3]/2).cpu().item()
                    x2, y2 = (box[0] + box[2]/2).cpu().item(), (box[1] + box[3]/2).cpu().item()
                    x1, y1 = max(0, math.ceil(x1*W-1)), max(0, math.ceil(y1*H-1))
                    x2, y2 = min(W, int(x2*W-1)), min(H, int(y2*H-1))
                    if x1 == x2:
                        if x1 == 0:
                            x2 += 1
                        else:
                            x1 -= 1
                    if y1 == y2:
                        if y1 == 0:
                            y2 += 1
                        else:
                            y1 -= 1

                    x_ids, y_ids = [x_id for x_id in range(x1, x2+1)], [y_id for y_id in range(y1, y2+1)]
                    xy_ids = list(itertools.product(x_ids, y_ids))
                    x_ids = torch.as_tensor([xy[0] for xy in xy_ids])
                    y_ids = torch.as_tensor([xy[1] for xy in xy_ids])
                    img_mask = torch.ones((H, W))
                    img_mask = img_mask.index_put((x_ids, y_ids), torch.zeros((x2-x1+1, y2-y1+1)).flatten())
                    ft_map = ft_map * img_mask.to(device)
                mask_dict[clsid] = ft_map
            
            # background mask, exclude person's bbox
            mask_dict[80] = torch.ones((H, W)).to(device) * mask_dict[0]
            mask_dicts.append(mask_dict)

        return mask_dicts

    def get_cls_bbox_fts(self, src, bbox_mask_dicts):
        cls_ft_dicts = []
        for i, bbox_mask_dict in enumerate(bbox_mask_dicts):
            cls_ft_dict = dict()
            for clsid, cls_mask in bbox_mask_dict.items():
                cls_ft = src[i] * cls_mask
                cls_ft = cls_ft.flatten(1).mean(1)  # 2048
                cls_ft_dict[clsid] = cls_ft
            cls_ft_dicts.append(cls_ft_dict)
        return cls_ft_dicts

    def parsing_annotation(self, targets):
        bsz = len(targets["relations"])
        # rels
        predicates = [r[1] for rels in targets["relations"] for r in rels]
        merged_obj_ids = [r[2] for rels in targets["svo_ids"] for r in rels]
        pdb.set_trace()
        merged_rels = [" ".join(rel) for rel in rels]
        merged_rel_imgids = [i for i, rels in enumerate(targets["relations"]) for r in rels]

        # build "batch rels" for text encoder input 
        rels, predicates, rel_seq, obj_ids, rel_endpoints = [], [], [], [], []
        for i in range(bsz):
            _rels = [rel for rel in targets["relations"][i]]
            rels.append(_rels)
            rel_seq.append(" ".join(_rels))
            predicates.append(r[2] for r in targets["svo_ids"][i])
            obj_ids.append(r[2] for r in targets["svo_ids"][i])
            start, end = 0, 0
            endpoints = []
            assert 1==0 ############## obj len may be > 1
            for _rel in _rels:
                end += len(_rel.split(" "))
                endpoints.append([start, end])
                start += end
            rel_endpoints.append(endpoints)

        merged_rels, merged_rel_imgids, merged_predicates, merged_obj_ids, \
            rels, predicates, rel_seq, obj_ids, rel_endpoints \
            = self.pseudo_double_example(
                merged_rels, merged_rel_imgids, merged_predicates, merged_obj_ids, \
                rels, predicates, rel_seq, obj_ids, rel_endpoints
            )

        return merged_rels, merged_rel_imgids, merged_predicates, merged_obj_ids, \
            rels, predicates, rel_seq, obj_ids, rel_endpoints

    def forward(self, image_tensor, targets, loss_type="visual_text_joint"):
        """
        image_tensor: C,W,H       
        targets: boxes, box_class, caption, relation, svo_ids, so_names
        """
        if isinstance(image_tensor, (list, torch.Tensor)):
            image_tensor = nested_tensor_from_tensor_list(image_tensor)

        features, pos = self.backbone(image_tensor)  # image encoder
        pdb.set_trace()
        
        src, mask = features[-1].decompose()
        assert mask is not None
        device = src.device
        bbox_mask_dicts = self.get_bbox_masks(targets["boxes"], targets["box_labels"], src.shape[-2], src.shape[-1], device)
        bbox_feat_dicts = self.get_cls_bbox_fts(src, bbox_mask_dicts)
        bsz = src.shape[0]

        rels, rel_imgids, predicates, obj_ids = self.parsing_annotation(targets)
        
        src = src.repeat(2, 1, 1, 1)
        bbox_mask_dicts = bbox_mask_dicts + bbox_feat_dicts
        bbox_feat_dicts = bbox_feat_dicts + bbox_feat_dicts
        num_rels = len(rels)

        # Encode the text for each relation
        tokenized = self.tokenizer.batch_encode_plus(rels, \
            padding="longest", return_tensors="pt").to(device)
        encoded_text = self.text_encoder(**tokenized)
        # Transpose memory because pytorch's attention expects sequence first                                                                                                                                                                                    
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = self.resizer(text_memory)
        
        memory_cache = self.transformer(
            self.input_proj(src),
            mask.repeat(2, 1, 1),
            pos[-1].repeat(2, 1, 1, 1),
            text_memory_resized,
            text_attention_mask,
            encoded_text
        )
        image_fts, text_fts = memory_cache["image_memory"], memory_cache["text_memory"]
        pdb.set_trace()
        # build corre_mat for contrastive loss
        corre_mat = torch.eye(num_rels)
        for i, rel_i in enumerate(rels):
            for j, rel_j in enumerate(rels):
                if i == j:
                    continue
                if obj_ids[i] == obj_ids[j] and predicates[i] == predicates[j]:
                    corre_mat[i, j] = 1
        
        # extract visual feature for each relation in the rels
        rel_fts = []
        for i, rel in enumerate(rels):
            rel_imgid = rel_imgids[i]
            h_ft = bbox_feat_dicts[rel_imgid][0]  # person
            o_ft = bbox_feat_dicts[rel_imgid][obj_ids[i]]
            ho_ft = torch.cat([h_ft, o_ft])
            rel_fts.append(ho_ft)

        if loss_type == "visual_text_joint":
            loss = self.build_hoi_contrast(rel_fts, text_memory, corre_mat)
        elif loss_type == "global_hoi_predict":
            raise NotImplementedError
        elif loss_type == "ho_patch_jigsaw":
            raise NotImplementedError
        elif loss_type == "visual_feat_reconstruct":
            raise NotImplementedError

        pdb.set_trace()

        # Concat on the sequence dimension
        src = torch.cat([src, text_memory_resized], dim=0)
        # For mask, sequence dimension is second
        mask = torch.cat([mask, text_attention_mask], dim=1)
        # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

    #def build_hoi_contrast(self):

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def build(args):
    device = torch.device(args.device)
    
    transformer = build_transformer(args)

    model = PretrainModel(backbone, transformer)
   
    weight_dict = {'loss_seq': 1}
    #num_classes = 100
    #criterion = SetCriterion(num_classes, weight_dict, args).to(device)


    return model, None, None