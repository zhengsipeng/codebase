import imp
import json
import os
from matplotlib.pyplot import annotate
from tqdm import tqdm
import glob
import pdb
import pickle as pkl
import numpy as np
from collections import defaultdict


def generate_vrd_annos(dbname):
    if dbname == "vidor_part":
        dbname, postfix = dbname.split("_")
    else:
        postfix = ""
    with open("%s/action.txt"%dbname, "r") as f:
        action_list = [l.strip() for l in f.readlines()]
    action_dict = dict(zip(action_list, range(len(action_list))))
    with open("%s/obj.txt"%dbname, "r") as f:
        obj_list  = [l.strip() for l in f.readlines()]
    obj_dict = dict(zip(obj_list, range(len(obj_list))))

    if dbname == "vidor" or dbname == "vidor_part":
        anno_files = []
        for split in ["training", "validation"]:
            anno_files += glob.glob("vidor/annotation/%s/**/*.json"%split)
    else:
        anno_files = []
        for split in ["train", "test"]:
            anno_files += glob.glob("vidvrd/annotation/%s/*.json"%split)
    for path in ["/data3/zsp/data/%s/vrd_annotation/"%dbname, "%s/vrd_annotation/"%dbname, 
                 "/dataset/28d47491/zsp/data/%s/vrd_annotation/"%dbname]:
        if not os.path.exists(path):
            os.makedirs(path)

    vrd_anno = dict()
    video_num = 0
    for anno_f in tqdm(anno_files):
        video_num += 1
        if postfix != "" and video_num > 1000:
            break

        if dbname == "vidvrd":
            vid = anno_f.split(".")[0].split("/")[-1]
        else:
            subid, vid = anno_f.split(".")[0].split("/")[-2: ]
  
        with open(anno_f, "r") as f:
            data = json.load(f)

        rels = data["relation_instances"]
        trajectories = data["trajectories"]
        traj_objs = data["subject/objects"]
        fnum = data["frame_count"]
        
        each_frame_dict = dict()
        each_frame_reldict = dict()  # dict of rel_inst (corresponding so_traj and begin/end fid) for each frame
        for rel_id, rel in enumerate(rels):
            begin, end = rel["begin_fid"], rel["end_fid"]
            stid, otid = rel["subject_tid"], rel["object_tid"]
            pclass = rel["predicate"]
            pclassid = action_dict[pclass] 
            
            for fid in range(begin, end):
                if fid not in each_frame_dict.keys():
                    each_frame_dict[fid] = {}
                    each_frame_reldict[fid] = {}
                
                if "%s-%s"%(stid, otid) not in each_frame_dict[fid].keys():
                    each_frame_dict[fid]["%s-%s"%(stid, otid)] = [pclassid]
                else:
                    if pclassid not in each_frame_dict[fid]["%s-%s"%(stid, otid)]:
                        each_frame_dict[fid]["%s-%s"%(stid, otid)].append(pclassid)
                       
        for fid, each_frame in each_frame_dict.items():
            so_traj_ids = []
            sclasses, oclasses, verb_classes = [], [], []
            sboxes, oboxes = [], []
            rel_dict = each_frame_dict[fid]
            for so, v in each_frame.items():
                stid, otid = int(so.split("-")[0]), int(so.split("-")[1])
                so_traj_ids.append([stid, otid])
                
                for traj in traj_objs:
                    if traj["tid"] == stid:
                        sclass = traj["category"]
                    if traj["tid"] == otid:
                        oclass = traj["category"]
                assert sclass and oclass
                sclassid, oclassid = obj_dict[sclass], obj_dict[oclass]
                sclasses.append(sclassid)
                oclasses.append(oclassid)
                verb_classes.append(v)
                
                boxes = trajectories[fid]

                for box in boxes:
                    if box["tid"] == stid:
                        sboxes.append([box["bbox"]["xmin"], box["bbox"]["ymin"], box["bbox"]["xmax"], box["bbox"]["ymax"]])
                    if box["tid"] == otid:
                        oboxes.append([box["bbox"]["xmin"], box["bbox"]["ymin"], box["bbox"]["xmax"], box["bbox"]["ymax"]])
               
            assert len(sboxes) == len(sclasses)
          
            if len(list(set(verb_classes[0]))) < len(verb_classes[0]):
                pdb.set_trace()
            each_frame_dict[fid] = {"sclasses": sclasses, "oclasses": oclasses, "verb_classes": verb_classes,
                                    "so_traj_ids": so_traj_ids, "sboxes": sboxes, "oboxes": oboxes}

        #for path in ["/data3/zsp/data/%s/vrd_annotation/"%dbname, "%s/vrd_annotation/"%dbname, 
        #         "/dataset/28d47491/zsp/data/%s/vrd_annotation/"%dbname]:
        for path in ["/data3/zsp/data/%s/vrd_annotation/"%dbname]:
            with open(path+"%s.pkl"%vid, "wb") as f:
                pkl.dump(each_frame_dict, f)

        vrd_anno[vid] = each_frame_dict
       
    dbname = dbname.split("_")[0]
    if postfix != "":
        postfix = "_" + postfix

    np.save("/data3/zsp/data/%s/vrd_annotation%s.npy"%(dbname, postfix), vrd_anno)
    #with open("/data3/zsp/data/%s/vrd_annotation%s.json"%(dbname, postfix), "w") as f:
    #    json.dump(vrd_anno, f)
    assert 1==0
    with open("%s/vrd_annotation%s.pkl"%(dbname, postfix), "wb") as f:
        pkl.dump(vrd_anno, f)
    with open("/data3/zsp/data/%s/vrd_annotation%s.pkl"%(dbname, postfix), "wb") as f:
        pkl.dump(vrd_anno, f)
    with open("/dataset/28d47491/zsp/data/%s/vrd_annotation%s.pkl"%(dbname, postfix), "wb") as f:
        pkl.dump(vrd_anno, f)


def get_max_numpair_frame(dbname):
    with open("/data3/zsp/data/%s/vrd_annotation.pkl"%dbname, "rb") as f:
        annos = pkl.load(f)
    max_num = 0
    num_20 = 100
    for vid in annos.keys():
        each_frame_dict = annos[vid]
        for fid, fanno in each_frame_dict.items():
            num_svo = len(fanno["verb_classes"])
            if num_svo > 100:
                num_20 += 1
                print(num_20)
            if num_svo > max_num:
                max_num = num_svo
                print("max_num", max_num)
            break
          

def filterout_anno(dbname):
    max_len = 24 if dbname == "vidvrd" else 32
    train_frame_sample = "/data3/zsp/data/%s/training_frame_sample.json"%dbname
    annotation_file = "/data3/zsp/data/%s/vrd_annotation.pkl"%dbname
    with open(train_frame_sample, "r") as f:
        data = json.load(f)
    with open(annotation_file, "rb") as f:
        annotations = pkl.load(f)

    frame_names = data["training_fids"]
    for frame_name in tqdm(frame_names):
        for i in range(max_len):
            video_id, fid = frame_name.split("-")[-2:]
            fid = int(fid)
            #pdb.set_trace()
            anno = annotations[video_id][fid]
            if len(list(anno.keys())) == 0:
                print(anno)
                print(frame_name)
                assert 1==0

    #, data["max_lens"]

if __name__ == "__main__":
    #generate_vrd_annos("vidvrd")
    generate_vrd_annos("vidor")
    #generate_vrd_annos("vidor_part")
    #get_max_numpair_frame("vidor")
    #filterout_anno("vidvrd")