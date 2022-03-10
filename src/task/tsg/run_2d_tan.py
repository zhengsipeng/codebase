from src.task import *
from src.optimization.lr_scheduler import build_scheduler
from src.dataset.tsg_dbase.dataloader import build_dataloader
from src.models.tsg.tan.tan import TAN
from src.optimization.loss import build_loss_func
from src.trainer.tsg.trainer_tan import Trainer
import pdb


def main():
    args, config = set_parser_config()
    model =  TAN(args, config)

    if config.WEIGHTS.model_weight != '':
        LOGGER.info(f"Loading model weights from {config.WEIGHTS.model_weight}")
        load_model_weights_with_mismatch(model, os.path.join(config.WEIGHTS.model_weight))

    parameter_group = build_optimizer_parameters(config, model)
    if args.distributed:
        model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                             model=model,
                                                             model_parameters=parameter_group)

    LOGGER.info(f'Training with {dist.get_world_size()} gpus')
    if args.fp16:
        LOGGER.info('Enable fp16 Training')
        fp16 = model_engine.fp16_enabled()
    
    dataset_train, dataset_val, dataloader_train, dataloader_val = build_dataloader(args, config)

    steps_per_epoch = len(dataloader_train)
    scheduler = build_scheduler(config, optimizer, steps_per_epoch)
    criterion = build_loss_func(config)
    trainer = Trainer(args, 
                      config, 
                      model_engine, 
                      optimizer, 
                      scheduler, 
                      criterion, 
                      dataloader_train, 
                      dataloader_val)

    trainer.train(args.resume)


if __name__ == '__main__':
    deepspeed.init_distributed()
    main()
