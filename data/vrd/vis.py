import json
import glob
import pickle as pkl
import os
from tqdm import tqdm

fpaths = glob.glob("/data3/zsp/data/vidor/vrd_annotation/*.pkl")

with open("vidor/action.txt", "r") as f:
    actions = [l.strip() for l in f.readlines()]

with open("vidor/obj.txt", "r") as f:
    objs = [l.strip() for l in f.readlines()]

vid2dir = dict()
all_dirs = glob.glob("/dataset/28d47491/zsp/data/vidor/annotation/training/**/*")
for _dir in tqdm(all_dirs):
    
    vid = _dir.split("/")[-1].split(".")[0]
    subdir = _dir.split("/")[-2]
    vid2dir[vid] = subdir

for fpath in fpaths:
    vid = fpath.split("/")[-1].split(".")[0]
    subdir = fpath.split("/")[-2]
    with open(fpath, "rb") as f:
        data = pkl.load(f)
    
    for fid in data.keys():
        if fid < 30:
            continue
        fdata = data[fid]
        sclasses = fdata["sclasses"]
        oclasses = fdata["oclasses"]
        vclasses = fdata["verb_classes"]
        if len(sclasses) < 10:
            continue
        subdir = vid2dir[vid]
        os.system("cp /dataset/28d47491/zsp/data/vidor/images/%s/%s/%06d.jpg vis/%s-%06d-%s.jpg"%(subdir, vid, fid, vid, fid+1, subdir))
        #print(vid, subdir, fid)
        f = open("vis/%s-%06d.txt"%(vid, fid), "w")

        for i in range(len(sclasses)):
            sname = objs[sclasses[i]]
            oname = objs[oclasses[i]]
            vname = actions[vclasses[i][0]]
           
            f.writelines("%s %s %s \n"%(sname, vname, oname))
        f.close()
        break
        #

