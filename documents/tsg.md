# Prepare data
```
data/tsg
-- ActivityNet
---- activitynet_v1-3.part-0*
---- train/val/test.json
-- Charades-STA
-- TACoS
---- merge_npys_to_hdf5.py
---- train/val/test.json
---- tall_c3d_features.hdf5
----
```

python src/task/tsg/run_2d_tan.py --deepspeed --deepspeed_config src/configs/tsg/tan/ds_cfg_tan.json --distributed --config src/configs/tsg/tan/2D-TAN-64x64-K9L4-conv.yaml --blob_mount_dir .

python src/task/tsg/run_2d_tan.py --deepspeed --deepspeed_config src/configs/ds_cfgs/ds_cfg.json --distributed --config src/configs/tsg/actnet_loc_64f.yaml --blob_mount_dir .