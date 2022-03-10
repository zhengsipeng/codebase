# Installation
* **Install COCOAPI and panopticapi**
```
https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools
https://github.com/cocodataset/panopticapi.git#egg=panopticapi
```
# Pix2Seq
## Training
```
sh train.sh --model pix2seq --output_dir outputs
```

## Evaluation
```
sh train.sh --model pix2seq --output_dir outputs --resume /path/to/checkpoints --eval
```

run "export PYTHONPATH$=PYTHONPATH:." first

python src/task/tsg/run_2d_tan.py --deepspeed --deepspeed_config src/configs/ds_cfgs/ds_cfg.json --distributed --config src/configs/tsg/actnet_loc_64f.yaml --blob_mount_dir .