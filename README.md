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

