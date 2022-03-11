import json

def merge(dbname):
    trainval = dict()
    with open("%s/train.json"%dbname, "r") as f:
        train = json.load(f)
    print(len(train))
    with open("%s/val.json"%dbname, "r") as f:
        val = json.load(f)
    print(len(val))
    for split in [train, val]:
        for k, v in split.items():
            trainval[k] = v
    print(len(trainval))
    with open("%s/trainval.json"%dbname, "w") as f:
        json.dump(trainval, f)


if __name__ == "__main__":
    merge("ActivityNet")
    merge("TACoS")