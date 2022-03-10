# This script is used to generate image idxs for training
import os
import json
import glob
import numpy as np
from tqdm import tqdm


def generate_train_img_idxs(dbname, duration, max_len=32):
    if dbname == "vidor_part":
        dbname, postfix = dbname.split("_")
    else:
        postfix = ""

    if dbname == "vidor":
        anno_files = []
        for split in ["training"]:
            anno_files += glob.glob("vidor/annotation/%s/**/*.json"%split)
    else:
        anno_files = []
        for split in ["train"]:
            anno_files += glob.glob("vidvrd/annotation/%s/*.json"%split)

    print(len(anno_files))
    
    training_fids = []
    max_lens = []
    total_fnum = 0
    video_num = 0
    for anno_f in tqdm(anno_files):
        video_num += 1
        if postfix != "" and video_num > 1000:
            break

        if dbname == "vidor" or dbname == "vidor_part":
            classname, videoname = anno_f.split(".")[0].split("/")[-2:]
        else:
            videoname = anno_f.split(".")[0].split("/")[-1]
     
        with open(anno_f, "r") as f:
            data = json.load(f)
        rels = data["relation_instances"]
        fnum = data["frame_count"]
        total_fnum += fnum
        pos_fids = np.zeros(fnum)
        
        for rel in rels:
            begin_fid = rel["begin_fid"]
            end_fid = rel["end_fid"]
            for idx in range(begin_fid, end_fid):
                pos_fids[idx] = 1
        
        selected_fnum = 0
        for idx in range(0, fnum, duration):
            #if pos_fids[idx] == 0 or idx + 7 >= fnum:
            #    continue

            _max_len = max_len
            for i in range(0, max_len):
                if idx+i >= fnum or pos_fids[idx+i] == 0:
                    _max_len = i
                    break

            if _max_len < max_len:
                continue

            if dbname == "vidor" or dbname == "vidor_part":
                sample_name = classname + "-" + videoname + "-" + str(idx)
            else:
                sample_name = videoname + "-" + str(idx)
            
            training_fids.append(sample_name)
            max_lens.append(_max_len)
            selected_fnum += 1
        print(total_fnum, len(training_fids), fnum, selected_fnum)
        #print(len(training_fids))
    
    dbname = dbname.split("_")[0]

    print("Num of Sample Frames: %d"%len(training_fids))
    data = {"training_fids": training_fids, "max_lens": max_lens}

    if postfix != "":
        postfix = "_" + postfix

    if not os.path.exists("/data3/zsp/data/%s"%dbname):
        os.makedirs("/data3/zsp/data/%s"%dbname)
    with open("/data3/zsp/data/%s/training_frame_sample%s.json"%(dbname, postfix), "w") as f:
        json.dump(data, f)
    with open("%s/training_frame_sample%s.json"%(dbname, postfix), "w") as f:
        json.dump(data, f)
    with open("/dataset/28d47491/zsp/data/%s/training_frame_sample%s.json"%(dbname, postfix), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    #generate_train_img_idxs(dbname="vidvrd", duration=4, max_len=24)  # 8433
    #generate_train_img_idxs(dbname="vidor", duration=16, max_len=32)  # 367200(7145644)
    generate_train_img_idxs(dbname="vidor_part", duration=33, max_len=32)  # 367200(7145644)
    