from .resnet import build_resnet_encoder


def build_image_encoder(config):
    if config.ImageEncoder.backbone == "vit":
        raise NotImplementedError
    elif config.ImageEncoder.backbone in ["resnet50", "resnet101"]:
        return build_resnet_encoder(config)
    else:
        raise NotImplementedError
