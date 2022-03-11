from torch import nn
import src.models.tsg.tan.frame_modules as frame_modules
import src.models.tsg.tan.prop_modules as prop_modules
import src.models.tsg.tan.map_modules as map_modules
import src.models.tsg.tan.fusion_modules as fusion_modules
import pdb 


class TAN(nn.Module):
    def __init__(self, args, config):
        super(TAN, self).__init__()
        self.cfg = config
        self.fp16 = args.fp16
        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2)) # 8,512,64
        map_h, map_mask = self.prop_layer(vis_h) # 8,512,64,64;  8,1,64,64
        # 8,33,300;  8,33,1
        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask) # 8,512,64,64
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask  # 8,1,64,64
        
        return prediction, map_mask

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask
        pdb.set_trace()
        return fused_h, prediction, map_mask


if __name__ == '__main__':
    from easydict import EasyDict
    from mmcv import Config
    import json
    args = EasyDict({'deepspeed_sparse_attention': False, 'deepspeed_config':'src/configs/ds_cfgs/ds_config.json', 'config_path':'src/configs/pretrain.yaml', 'stage':2})
    config = Config.fromfile(args.config_path)

    import pdb; pdb.set_trace()

    model = TAN(args, config)
    print(config)

    video_inp = torch.randn(16, 3, 64, 192, 320)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text = "Replace me by any text you'd like."
    text_inp = tokenizer(text, return_tensors='pt')

    video_feat, text_feat = model(video_inp, text_inp, args.stage)
    
    import pdb; pdb.set_trace()