import torch
import pdb

def resume_model(args, model_without_ddp, optimizer, lr_scheduler):
    cur_ap, max_ap = 0, 0
    if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    resume_state_dict = {}
    checkpoint_state_dict = checkpoint['model']
    model_state_dict = model_without_ddp.state_dict()
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict:
            resume_value = v
            print(f'Load {k} {tuple(v.shape)} from scratch.')
        elif v.shape != checkpoint_state_dict[k].shape: 
            checkpoint_value = checkpoint_state_dict[k]
            print(checkpoint_value.shape, resume_value.shape)
            num_dims = len(checkpoint_value.shape)
            
            if "vocal_embed" in k:
                resume_value = torch.cat([checkpoint_value, v[checkpoint_value.shape[0]:]], dim=0)
        
            elif "classifier" in k:
                if checkpoint_value.shape[0] > v.shape[0]:
                    resume_value = checkpoint_value[:v.shape[0]]
                #elif checkpoint_value.shape[0] < v.shape[0]:  # !!!!!!!!! TODO
                #    resume_value[:checkpoint_value.shape[0]] = v
            else:
                raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")
            print(f"Load {k} {tuple(v.shape)} from resume model "
                    f"{tuple(checkpoint_value.shape)}.")
        else:
            resume_value = checkpoint_state_dict[k]
        resume_state_dict[k] = resume_value

    model_without_ddp.load_state_dict(resume_state_dict)
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler.end_epoch = args.epochs
        args.start_epoch = checkpoint['epoch'] + 1
    if 'ap' in checkpoint:
        cur_ap = checkpoint['ap']
        max_ap = checkpoint['max_ap']
    
    return cur_ap, max_ap