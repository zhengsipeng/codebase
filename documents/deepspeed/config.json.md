# Notes
```
pip install deepspeed

A full example for deepspeed training can be seen in "src/task/tsg/run_2d_tan.py" and "src/trainer/tsg/trainer_tan.py"

1GPU: python "src/task/tsg/run_2d_tan.py --deepspeed --deepspeed_config src/configs/tsg/tan/ds_cfg_tan.json --distributed --config src/configs/tsg/tan/tacos_2D-TAN-128x128-K5L8-conv.yaml --blob_mount_dir ."

Multi-GPU: "deepspeed src/task/tsg/run_2d_tan.py --deepspeed --deepspeed_config src/configs/tsg/tan/ds_cfg_tan.json --distributed --config src/configs/tsg/tan/tacos_2D-TAN-128x128-K5L8-conv.yaml --blob_mount_dir ."

More details can be seen in https://www.deepspeed.ai/docs
```



# Configuration
a config.json should be like this:
```
{

    "train_micro_batch_size_per_gpu": 4,

    "gradient_accumulation_steps": 1,  # automatic gradient accumulation to enlarge batch

    "steps_per_print": 100,

    "zero_optimization": {  #  ZeRO is to optimize the training efficiency
      "stage": 2,
      "allgather_partitions": true,
      "allgather_bucket_size": 5e8,
      "overlap_comm": false,
      "reduce_scatter": true,
      "reduce_bucket_size": 5e8,
      "contiguous_gradients" : false,
      "stage3_gather_fp16_weights_on_model_save": true
    },

    "fp16": {  # amp training
      "enabled": false,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "initial_scale_power": 32,
      "hysteresis": 2,
      "min_loss_scale": 1
    },

    "optimizer": {  
        "type": "Adam",
        "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.98],
        "eps": 1e-8,
        "weight_decay": 1e-5
        }
    },


    "sparse_attention": {
      "mode": "fixed",
      "block": 32,
      "different_layout_per_head": true,
      "num_local_blocks": 16,
      "num_global_blocks": 1,
      "attention": "bidirectional",
      "horizontal_global_attention": true,
      "num_different_global_patterns": 4
    }
    }
```

