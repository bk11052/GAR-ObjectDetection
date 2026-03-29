"""
PASCAL VOC 2007 Dataset loader for GAR.
Custom implementation with Faster R-CNN compatible (image, target) transforms.
"""

import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    """
    PASCAL VOC Detection Dataset.

    Args:
        root: path to VOCdevkit directory
        year: '2007' or '2012'
        split: 'train', 'val', 'trainval', or 'test'
        transforms: callable that takes (PIL.Image, target_dict) and returns
                    (Tensor, target_dict). Use get_voc_transforms().
    """

    def __init__(self, root, year="2007", split="trainval", transforms=None):
        self.root = root
        self.year = year
        self.split = split
        self.transforms = transforms

        voc_root = os.path.join(root, f"VOC{year}")
        split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")

        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.img_dir = os.path.join(voc_root, "JPEGImages")
        self.ann_dir = os.path.join(voc_root, "Annotations")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")

        img = Image.open(img_path).convert("RGB")
        target = self._parse_annotation(ann_path, idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _parse_annotation(self, ann_path, idx):
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []
        difficult = []

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in CLASS_TO_IDX:
                continue

            diff = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name] + 1)  # 0 = background
            difficult.append(diff)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "difficult": torch.as_tensor(difficult, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
        }
        return target

    def get_image_id(self, idx):
        return self.ids[idx]


def collate_fn(batch):
    """Custom collate for variable-size targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)


class Compose:
    """Compose transforms that operate on (image, target) pairs."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert PIL Image to tensor, pass target through."""
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip for both image and bounding boxes."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width = image.width if isinstance(image, Image.Image) else image.shape[-1]
            image = TF.hflip(image)
            boxes = target["boxes"]
            # Flip x coordinates: new_xmin = width - old_xmax
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def get_voc_transforms(train=True):
    """Returns (image, target) compatible transforms for VOC."""
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())
    return Compose(transforms)
