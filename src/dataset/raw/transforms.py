# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import math
import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from src.utils.box_ops import box_xyxy_to_cxcywh, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from src.utils.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")


    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
 
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        result_img, result_target = crop(img, target, region)
        if not self.respect_boxes or len(result_target["boxes"]) == init_boxes or i == max_patience - 1:
            return result_img, result_target
        return result_img, result_target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        # h, w = image.shape[-2:]
        h, w = target["size"][0], target["size"][1]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()
        target["size"] = scaled_size

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        if "masks" in target:
            masks = target['masks']
            masks = interpolate(
                masks[:, None].float(), scaled_size, mode="nearest")[:, 0] > 0.5
            target['masks'] = masks
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ["labels", "area", "iscrowd"]

        target = target.copy()
        target["size"] = torch.tensor([h, w])

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area
            fields.append("boxes")

        if "masks" in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append("masks")

        # remove elements for which the boxes or masks that have zero area
        if "boxes" in target or "masks" in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                target[field] = target[field][keep]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])

        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            padding = out_desired_size - scaled_size
            output_image = F.pad(scaled_image, [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target




class TargetPermute(object):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, img, target):
        num_object = len(target["labels"])
        rand_idx = torch.randperm(num_object)
        for k in ["boxes", "labels", "area", "segmentation", "keypoints"]:
            if k in target.keys():
                target[k] = target[k][rand_idx]
        return img, target


class SeqBuilder(object):
    def __init__(self, args) -> None:
        self.num_bins = args.dictionary.num_bins
        self.num_classes = args.dictionary.num_classes
        self.num_vocal = args.dictionary.num_vocal
        self.max_objects = args.num_max_objs
        self.max_input_size = args.max_input_size
        self.seq_aug = args.seq_aug

    def __call__(self, img, target):
        label = target["labels"]
        box = target["boxes"]
        img_size = target["size"]
        h, w = img_size[0], img_size[1]
        scale_factor = torch.stack([w, h, w, h], dim=0)

        label_token = label.unsqueeze(1) + self.num_bins + 1
        scaled_box = box * scale_factor
        scaled_box = box_cxcywh_to_xyxy(scaled_box)
        box_tokens = (scaled_box / self.max_input_size * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
        real_tokens = torch.cat([box_tokens, label_token], dim=1)

        input_seq = real_tokens.flatten()
        end_token = torch.tensor([self.num_vocal - 2], dtype=torch.int64)
        target_seq = torch.cat([input_seq, end_token], dim=0)

        if self.seq_aug:
            num_objects = real_tokens.shape[0]
            num_noise = self.max_objects - len(label)

            # augmentation for input seq
            random_class = torch.randint(0, self.num_classes, (num_noise, 1)) + self.num_bins + 1
            random_box_x0y0 = torch.rand(num_noise, 2)
            random_box_wh = torch.rand(num_noise, 2)
            random_box_x1y1 = (random_box_x0y0 + random_box_wh).clamp(min=0, max=1)
            random_scaled_box = torch.cat([random_box_x0y0, random_box_x1y1], dim=1) * scale_factor
            random_box_tokens = (random_scaled_box / 1333 * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            random_tokens = torch.cat([random_box_tokens, random_class], dim=1)

            if num_objects > 0:
                # random scaling
                jitter_box_idx = torch.randint(0, num_objects, (num_noise,))
                jitter_class = label_token[jitter_box_idx]
                jitter_box = box[jitter_box_idx]
                jitter_box_wh = jitter_box[:, 2:].repeat(1, 2)
                jitter_box = box_cxcywh_to_xyxy(jitter_box)
                jitter_box = (torch.rand((num_noise, 4)) - 0.5) * 2 * 0.2 * jitter_box_wh + jitter_box
                scaled_jitter_box = jitter_box.clamp(min=0, max=1.0) * scale_factor
                jitter_box_tokens = (scaled_jitter_box / 1333 * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
                jitter_tokens = torch.cat([jitter_box_tokens, jitter_class], dim=1)

                fake_tokens = torch.stack([random_tokens, jitter_tokens], dim=1)
                select_idx = torch.randint(0, 2, (num_noise,))
                fake_tokens = fake_tokens[range(num_noise), select_idx]
            else:
                fake_tokens = random_tokens
            
            input_seq = torch.cat([input_seq, fake_tokens.flatten()], dim=0)
            
            # augmentation for target seq
            fake_target_tokens = torch.zeros((num_noise, 5), dtype=torch.int64)
            fake_target_tokens[:, :3] = -100
            fake_target_tokens[:, 3] = self.num_vocal - 1  # noise class
            fake_target_tokens[:, 4] = self.num_vocal - 2  # eos
            fake_target_tokens = fake_target_tokens.flatten()
            target_seq = torch.cat([target_seq, fake_target_tokens], dim=0)
        
        target["input_seq"] = input_seq
        target["target_seq"] = target_seq

        return img, target

 