import os
import json


def main(mode):
    jnum = 0
    perrel_num = dict()
    if mode == 'train':
        rootdir = 'annotation/training/'
    else:
        rootdir = 'annotation/validation/'
    for subset in os.listdir(rootdir):
        subpath = rootdir + subset
        for jfile in os.listdir(subpath):
            jnum += 1
            if jnum == 1653:
                continue
            print(jnum, jfile)
            jpath = os.path.join(subpath, jfile)
            with open(jpath, 'rb') as f:
                data = json.load(f)
            insts = data['relation_instances']
            for inst in insts:
                cls = inst['predicate']
                if cls not in perrel_num.keys():
                    perrel_num[cls] = 0
                perrel_num[cls] += 1
    for key, value in perrel_num.items():
        print(key, value)


if __name__ == '__main__':
    main('val')

