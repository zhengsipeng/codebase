import src.dataset.transforms as T


def init_transform_dict(config):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    normalize = T.Compose([T.ToTensor(), T.Normalize(norm_mean, norm_std)])
    if config.Model.name == "pretrain":
        input_res=(384, 384),
        
        transform_dict = {
            'train': T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize([(384, 384)]),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, max_size=800),
                            T.RandomResize([(input_res)])
                        ]
                    )
                ),
                T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
                normalize,
            ]),
            'val': T.Compose([
                T.RandomResize([(384, 384)]),
                normalize,
            ]),
            'test': T.Compose([
                T.RandomResize([(384, 384)]),
                normalize,
            ])
        }
        
    else:
        raise NotImplementedError

    
    return transform_dict