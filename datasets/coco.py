# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from PIL import Image
import os
import os.path
import pickle
import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, idx_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file, idx_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
<<<<<<< HEAD
        self.idx_file = pickle.load(open(idx_file, "rb"))
        self.ids = self.idx_file.keys()
        self.ids_map = self.idx_file
=======
        self.ids = [0]
        self.ids_map = {0:[7, 8, 9, 10]}
<<<<<<< HEAD
>>>>>>> 087dfa61dce65b662e1ea35cb397a1dd996d2e83
=======
>>>>>>> 087dfa61dce65b662e1ea35cb397a1dd996d2e83

    def __getitem__(self, idx):
        img_batch, target_batch = [], []
        instances = self.ids_map[idx]
        for i in range(len(instances)):
            img_id = instances[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            path = self.coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.root, path)).convert('RGB')

            image_id = img_id
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            img_batch.append(img)
            target_batch.append(target)
        return img_batch, target_batch


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        det_keys = [obj["detection_key"] for obj in anno]

        loc_gt = []
        for obj in anno:
            if obj["obj_lat"] != None and obj["obj_lng"] != None:
                loc_gt.append([float(obj["obj_lat"]), float(obj["obj_lng"])])
            else:
                loc_gt.append([0, 0])

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        det_keys = torch.as_tensor(det_keys)
        loc_gt = torch.as_tensor(loc_gt)

        det_keys = det_keys[keep]
        loc_gt = loc_gt[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["det_key"] = det_keys
        target["loc_gt"] = loc_gt
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
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

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    if args.instance == "sv_sample":
        PATHS = {
            "train": (root / 'sv_sample_all', root / "annotations" / 'instances_sv_sample_all.json'),
            "val": (root / 'sv_sample_all', root / "annotations" / 'instances_sv_sample_all.json'),
        }
    elif args.instance == "sv_sample_multi":
        PATHS = {
            "train": (root / 'sv_sample_all', root / "annotations" / 'instances_sv_sample_multi.json'),
            "val": (root / 'sv_sample_all', root / "annotations" / 'instances_sv_sample_multi.json'),
        }

    else:
        PATHS = {
            "train": (root / str(args.instance+'_train_all'), root / "annotations" / str('instances_'+args.instance+'_train_all.json')),
            "val": (root / str(args.instance+'_val_all'), root / "annotations" / str('instances_'+args.instance+'_val_all.json')),
        }
        print(PATHS)

    img_folder, ann_file = PATHS[image_set]

    dataset = CocoDetection(img_folder, ann_file, args.coco_path+"idx/"+args.idx+"_"+image_set+"_all.p", transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
