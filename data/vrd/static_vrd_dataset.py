import glob
import json
from tqdm import tqdm

def static_rela_types():
    json_files = []

    for split in ["training", "validation"]:
        json_files += glob.glob("/data3/zsp/data/vidor/annotations/%s/**/*.json"%split)
    print(len(json_files))
    subject_list, object_list, rel_list = [], [], []
    max_obj, max_rel = 0, 0
    for jf in tqdm(json_files):
        with open(jf, "r") as f:
            data = json.load(f)
        so_list = data["subject/objects"]
        rels = data["relation_instances"]
        trajs = data["trajectories"]

        if len(rels) > max_rel:
            max_rel = len(rels)

        for each_f in trajs:
            if len(each_f) > max_obj:
                max_obj = len(each_f)
                print(max_obj, max_rel)

        for rel in rels:
            sid = rel["subject_tid"]
            oid  = rel["object_tid"]
            pclass = rel["predicate"]
            sclass, oclass = so_list[sid]["category"], so_list[oid]["category"]
            if sclass not in subject_list:
                subject_list.append(sclass)
            if oclass not in object_list:
                object_list.append(oclass)
            if (sclass, pclass, oclass) not in rel_list:
                rel_list.append((sclass, pclass, oclass))
 
        #break
    with open("vidor_subject.txt", "w") as f:
        for subject in subject_list:
            f.writelines(subject+"\n")
    with open("vidor_object.txt", "w") as f:
        for obj in object_list:
            f.writelines(obj+"\n")
    with open("vidor_relation.txt", "w") as f:
        for rel in rel_list:
            s,p,o = rel
            f.writelines(s+"-"+p+"-"+o+"\n")


def is_rel_multilabel(dbname):
    json_files = []

    if dbname == "vidor":
        for split in ["training", "validation"]:
            json_files += glob.glob("/data3/zsp/data/vidor/annotations/%s/**/*.json"%split)
    else:
        for split in ["train", "test"]:
            json_files += glob.glob("/data3/zsp/data/vidvrd/annotations/%s/*.json"%split)
    print(len(json_files))
    for jf in tqdm(json_files):
        with open(jf, "r") as f:
            data = json.load(f)
        
        so_list = data["subject/objects"]
        rels = data["relation_instances"]
        so_pair_dict = {}
      
        for rel in rels:
            sid = rel["subject_tid"]
            oid  = rel["object_tid"]
            begin_fid = rel["begin_fid"]
            end_fid = rel["end_fid"]
            sclass, oclass = so_list[sid]["category"], so_list[oid]["category"]
            pclass = rel["predicate"]

            if (sid, oid) in so_pair_dict.keys():
                unlabeled = True
                for [p, bfid, efid] in so_pair_dict[(sid, oid)]:
                    if p == pclass or bfid >= end_fid or efid <= begin_fid:
                        continue

                    print(sid, p, oid, bfid, efid)
                    print(sid, pclass, oid, begin_fid, end_fid)
                    #unlabeled = False
                    #assert 1==0
                if unlabeled:
                    so_pair_dict[(sid, oid)].append([pclass, begin_fid, end_fid])

            so_pair_dict[(sid, oid)] = [[pclass, begin_fid, end_fid]]
            

def get_anno_files(dbname):
    if dbname == "vidor":
        anno_files = []
        for split in ["training", "validation"]:
            anno_files += glob.glob("vidor/annotations/%s/**/*.json"%split)
    else:
        anno_files = []
        for split in ["train", "test"]:
            anno_files += glob.glob("vidvrd/annotations/%s/*.json"%split)
    
    print(len(anno_files))
    return anno_files


def get_minmax_action_len(dbname):
    anno_files = get_anno_files(dbname)
    
    training_fids = []
    minlen, maxlen = 3000, 0
    minnum, maxnum = 0, 0
    for anno_f in tqdm(anno_files):
        if dbname == "vidor":
            classname, videoname = anno_f.split(".")[0].split("/")[-2:]
        else:
            videoname = anno_f.split(".")[0].split("/")[-1]

        with open(anno_f, "r") as f:
            data = json.load(f)
        rels = data["relation_instances"]
        for rel in rels:
            begin_fid = rel["begin_fid"]
            end_fid = rel["end_fid"]
            duration = end_fid - begin_fid
            if duration > maxlen:
                maxlen = duration
            if duration < minlen:
                minlen = duration
            if duration < 24:
                minnum += 1
                print(minnum, maxnum)
            else:
                maxnum += 1
    print("MinMax: %d/%d"%(minlen, maxlen))


def get_minmax_wh_size(dbname):
    anno_files = get_anno_files(dbname)
    max_h, max_w = 0, 0
    for anno_f in tqdm(anno_files):
        with open(anno_f, "r") as f:
            data = json.load(f)
        width = data["width"]
        height = data["height"]
        if width > max_w:
            max_w = width
        if height > max_h:
            max_h = height
        print(width, height)
    print("Max Width and Height: %d/%d"%(max_w, max_h))


def cal_num_val_rinst(dbname):
    if dbname == "vidor":
        anno_files = glob.glob("vidor/annotation/validation/**/*.json")
    else:
        anno_files = glob.glob("vidvrd/annotation/test/*.json")
    num_rinsts = 0
    for anno_file in tqdm(anno_files):
        with open(anno_file, "r") as f:
            anno = json.load(f)
        rinsts = anno["relation_instances"]
        num_rinsts += len(rinsts)
    print(num_rinsts)


def cal_max_rinst_len():
    anno_files = glob.glob("vidor/annotation/validation/**/*.json")
    min_len = 10000
    min_num = 0
    total_num = 0
    for anno_file in tqdm(anno_files):
        with open(anno_file, "r") as f:
            anno = json.load(f)
        rinsts = anno["relation_instances"]
        for rinst in rinsts:
            total_num += 1
            begin_fid = rinst["begin_fid"]
            end_fid = rinst["end_fid"]
            duration = end_fid - begin_fid
            #print(duration)
            #if duration > 200:
            if duration < 200 and duration >= 100:
                min_num += 1
                print(duration, min_num, total_num)
                min_len = duration
    print(total_num)
    # 30142
    # <10: 967
    #10~100:13761
    #100~200:6263
    #>200:9117
if __name__ == "__main__":
    #static_rela_types()
    is_rel_multilabel('vidor')
    #get_minmax_action_len("vidor")  # vidvrd: 30/1200;  vidor: 3/5395(recommend: 16)
    #get_minmax_wh_size("vidor")  # W/H VIDOR: 640/1138; VIDVRD: 1920/1080
    #cal_num_val_rinst("vidor")
    cal_max_rinst_len()