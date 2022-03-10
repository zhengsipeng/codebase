import json
import torch
import torchvision


class PretrainDataset(torchvision.datasets.CocoDetection):
    def __init__(self, cfg, name, metadata_dir, image_dir, tokenizer,  transform=None, return_rawtext=False):
        super(PretrainDataset, self).__init__(image_dir, metadata_dir)
        self.cfg = cfg
        self.dbname = name.split("-")[0]
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.image_dir = image_dir
        self.return_rawtext = return_rawtext
        self.tokenizer = tokenizer
        self.prepare = ConvertCocoPolysToMask(return_masks=False, return_tokens=True)

        self.obj2id = json.load(open("data/pretrain/relation_annotation/merged_hoi_clss2id.json", "r"))
        self.verb2id = json.load(open("data/pretrain/relation_annotation/merged_hoi_verb2id.json", "r"))

        print("num of object classes: %d"%(len(list(self.obj2id.keys()))-1))  # 5716
        print("num of verb classes: %d"%(len(list(self.verb2id.keys()))))  # 2267

    def reassign_svo_ids(self, relations):
        svo_class_ids = []
        for i, relation in enumerate(relations):
            s, v, o = relation
            print(relation)
            sid, vid, oid = self.obj2id[s], self.verb2id[v], self.obj2id[o]
            '''
            if o == 'others':
                oid = self.obj2id["background"]
            else:
                oid = self.obj2id[o]
            '''
            svo_class_ids.append([sid, vid, oid])
        return svo_class_ids

    def __getitem__(self, idx):
        img, target = super(PretrainDataset, self).__getitem__(idx)
        image_id = self.ids[idx]

        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        relations = coco_img["relations"]
        so_class_names = coco_img["so_class_names"]
        svo_class_ids = coco_img["svo_class_ids"]

        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        
        img, target = self.prepare(img, target)

        target["so_class_names"] = so_class_names
        target["relations"] = relations
        target["svo_class_ids"] = svo_class_ids #self.reassign_svo_ids(relations)
        pdb.set_trace()
        img, target = self.transforms(img, target)

        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]

        #tokenized = self.prepare.tokenizer(caption, return_tensors="pt")

        return img, target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_tokens=False):
        self.return_masks = return_masks
        self.return_tokens = return_tokens

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
 
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target