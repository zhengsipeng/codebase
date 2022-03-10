'''
from .pix2seq import build_pix2seq_model
from .pix2hoi import build_pix2hoi_model
from .pretrain import build_pretrain_model
#from .hoi_pretrained_model import build_hoi_pretrained


build_all_model = {
    "pix2seq": build_pix2seq_model,
    "pix2hoi": build_pix2hoi_model,
    "pretrain": build_pretrain_model
    #"clip_pretrain": build_clip_pretrained
}
'''