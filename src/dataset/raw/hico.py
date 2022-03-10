from PIL import Image
from pathlib import Path
from collections import defaultdict
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision
import dataset.transforms as T
#from util.misc import draw_boxes
import pdb


class HICODetection(Dataset):
    def __init__(self, dataset, img_folder, anno_file, transforms, num_max_objs, num_max_hois, 
                    large_scale_jitter, img_set):
        self.dataset = dataset
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, "r") as f:
            annotations = json.load(f)
        self.annotations = self.filter_annotations(annotations)
        self.num_max_objs = num_max_objs
        self.num_max_hois = num_max_hois
        self.transforms = transforms
        self.transforms_unchanged = self.unchanged_transform()

        self.large_scale_jitter = large_scale_jitter

        self.valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self.valid_verb_ids = list(range(1, 118)) if self.dataset == "hico" else list(range(1, 29))
        

        if img_set == 'train' and self.dataset == "hico":
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))

    def filter_annotations(self, annotations):
        # make sure that each image has at least one positive sample
        #print("Filtering annotations...")
        _annotations = []
        for anno in annotations:
            if anno["hoi_annotation"] == 0:
                continue
            _annotations.append(anno)
        print("Number %d images are loaded"%len(_annotations))
        return annotations

    def unchanged_transform(self):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.RandomHorizontalFlip(),
            T.RandomDistortion(0.4, 0.4, 0.4, 0.5),
            normalize
        ])

    def set_rare_hois(self, anno_file):
        with open(anno_file, "r") as f:
            annotations = json.load(f)
        
        counts = defaultdict(lambda: 0)

        for img_anno in annotations:
            hois = img_anno["hoi_annotation"]
            bboxes = img_anno["annotations"]
            for hoi in hois:
                triplet = (self.valid_obj_ids.index(bboxes[hoi["subject_id"]]["category_id"]),
                           self.valid_obj_ids.index(bboxes[hoi["object_id"]]["category_id"]),
                           self.valid_verb_ids.index(hoi["category_id"])
                )
                counts[triplet] += 1

        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)
    
    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)

    def filter_target(self, target, img_anno):   
        kept_box_indices = [label[0] for label in target['labels']]
 
        target['labels'] = target['labels'][:, 1]
        
        obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
        sub_obj_pairs = []

        sub_box_ids, obj_box_ids = [], []
        verb_ids = []
        
        for hoi in img_anno['hoi_annotation']:
            #if self.dataset == "hico":
            #    flag = hoi['object_id'] not in kept_box_indices
            #else:
            flag = hoi["subject_id"] not in kept_box_indices or (hoi["object_id"] != -1 \
                    and hoi["object_id"] not in kept_box_indices)

            if hoi['subject_id'] not in kept_box_indices or flag:
                continue
    
            sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
            
            if sub_obj_pair in sub_obj_pairs:
                verb_labels[sub_obj_pairs.index(sub_obj_pair)][self.valid_verb_ids.index(hoi['category_id'])] = 1
                verb_ids[sub_obj_pairs.index(sub_obj_pair)].append(self.valid_verb_ids.index(hoi['category_id']))
            else:
                sub_box_ids.append(hoi["subject_id"])
                obj_box_ids.append(hoi["object_id"])
                sub_obj_pairs.append(sub_obj_pair)
                if self.dataset == "vcoco" and hoi["object_id"] == -1:
                    obj_labels.append(torch.tensor(len(self.valid_obj_ids)))
                else:
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                verb_label = [0 for _ in range(len(self.valid_verb_ids))]
                verb_label[self.valid_verb_ids.index(hoi['category_id'])] = 1
                sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                verb_labels.append(verb_label)
                verb_ids.append([self.valid_verb_ids.index(hoi['category_id'])])
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)
        
        if len(sub_obj_pairs) == 0:
            target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['verb_labels'] = torch.zeros((0, len(self.valid_verb_ids)), dtype=torch.float32)
            target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)

            target["sub_box_ids"] = torch.zeros((0,), dtype=torch.int64)
            target["obj_box_ids"] = torch.zeros((0,), dtype=torch.int64)
            target["verb_ids"] = torch.zeros((0,), dtype=torch.int64)
            target["verb_cum"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target['obj_labels'] = torch.stack(obj_labels)
            target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
            target['sub_boxes'] = torch.stack(sub_boxes)
            target['obj_boxes'] = torch.stack(obj_boxes)

            target["sub_box_ids"] = torch.as_tensor(sub_box_ids)
            target["obj_box_ids"] = torch.as_tensor(obj_box_ids)
            verb_num = torch.as_tensor([len(verb_ids[i]) for i in range(len(verb_ids))])
            verb_cum = torch.cumsum(verb_num, dim=0)
            target["verb_cum"] = torch.as_tensor(verb_cum)
            target["verb_ids"] = []
            for verb_id in verb_ids:
                target["verb_ids"] += verb_id
            target["verb_ids"] = torch.as_tensor(target["verb_ids"])
        
        # add negative HOI pairs
        human_box_ids = []
        for box_id in range(target["boxes"].shape[0]):
            if target["labels"][box_id] == 0:
                human_box_ids.append(box_id)
        
        hoi_pairs = [(target["sub_box_ids"][i], target["obj_box_ids"][i]) for i in range(target["sub_box_ids"].shape[0])]

        neg_pairs = []
        for hid in human_box_ids:
            for box_id in range(target["boxes"].shape[0]):
                if box_id == hid:
                    continue
                if (hid, box_id) not in hoi_pairs:
                    neg_pairs.append([hid, box_id])
        target["neg_hoi_pairs"] = torch.as_tensor(neg_pairs)
 
        return target

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
       
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_max_hois:
            img_anno['annotations'] = img_anno['annotations'][:self.num_max_hois]
        
        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Add index for confirming which boxes are kept after image transformation
        classes = [(i, self.valid_obj_ids.index(obj['category_id'])) for \
                    i, obj in enumerate(img_anno['annotations'])]

        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])

        """
        Object Detection Target Format
        {'boxes': tensor([[0.4707, 0.6620, 0.1694, 0.3454], [0.4535, 0.8317, 0.1953, 0.0579]]), 
         'labels': tensor([ 1, 35]), 'image_id': tensor([442808]), 
         'area': tensor([8533.7373,  199.0080]), 'iscrowd': tensor([0, 0]), 
         'orig_size': tensor([608, 640]), 'size': tensor([556, 585], dtype=torch.int32)} 

        HOI Target Format
        {'boxes': tensor([[0.5129, 0.4797, 0.8993, 0.8438], [0.5246, 0.4727, 0.5105, 0.3453]]), 
         'labels': tensor([ 0, 41]), 
         'area': tensor([318306.8750,  73955.3828]), 'iscrowd': tensor([0, 0]), 
         'orig_size': tensor([640, 427]), 'size': tensor([793, 529], dtype=torch.int32)
         'obj_labels': tensor([41], 'verb_labels': 1,117,
         'sub_boxes': tensor([[0.5129, 0.4797, 0.8993, 0.8438]]), 'obj_boxes': tensor([[0.5246, 0.4727, 0.5105, 0.3453]])
        }
        """
        
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]
            
            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self.transforms is not None:
                if self.large_scale_jitter:
                    img1, target1 = self.transforms(img, target)
                    img2, target2 = self.transforms(img, target)
                    
                    # make sure that num_boxes > 0
                    if target1["boxes"].shape[0] <= 1:
                        img1, target1 = self.transforms_unchanged(img, target)
                    if target2["boxes"].shape[0] <= 1:
                        img2, target2 = self.transforms_unchanged(img, target)

                    target1 = self.filter_target(target1, img_anno)
                    target2 = self.filter_target(target2, img_anno)
       
                    return img1, img2, target1, target2
                else:
                    img, target = self.transforms(img, target)
                    target = self.filter_target(target, img_anno)
                    if target["boxes"].shape[0] <= 1:
                        img, target = self.transforms_unchanged(img, target)
                  
        else:
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = torch.as_tensor(idx)

            if self.transforms is not None:
                img, target = self.transforms(img, target)
                target = self.filter_target(target, img_anno)
                if target["boxes"].shape[0] <= 1:
                    img, target = self.transforms_unchanged(img, target)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self.valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
    
        return img, target


def make_hico_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
 
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if args.large_scale_jitter:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.LargeScaleJitter(output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0),
                T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomDistortion(0.4, 0.4, 0.4, 0.5),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])
    
    if image_set == "val":
        if args.large_scale_jitter:
            return T.Compose([
                T.LargeScaleJitter(output_size=1333, aug_scale_min=1.0, aug_scale_max=1.0),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize
            ])
    
    raise ValueError(f'unknown {image_set}')


def build(img_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
        'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[img_set]
    dataset = HICODetection(args.dataset_file, img_folder, anno_file, 
                            transforms=make_hico_transforms(img_set, args),
                            num_max_objs=args.num_max_objs, num_max_hois=args.num_max_hois,
                            large_scale_jitter=args.large_scale_jitter, img_set=img_set)

    if img_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)

    return dataset

