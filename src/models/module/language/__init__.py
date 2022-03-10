def build_text_encoder(config):
    if "roberta" in config.TextEncoder.model:
        from transformers import RobertaModel
        text_encoder = RobertaModel.from_pretrained(config.TextEncoder.model)
    else:
        raise NotImplementedError
    
    if config.TextEncoder.freeze_text_encoder:
        for p in text_encoder.parameters():
            p.requires_grad_(False)

    return text_encoder