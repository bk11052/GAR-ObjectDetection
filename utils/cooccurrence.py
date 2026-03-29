"""
Co-occurrence matrix computation for GAR.

Computes:
  1. Object-Object co-occurrence: (O x O)
  2. Object-IndoorOutdoor co-occurrence: (O x 2)
  3. Object-Place co-occurrence: (O x 365)
  4. Object-Attribute co-occurrence: (O x 102)

Usage:
    python utils/cooccurrence.py --voc_root data/VOCdevkit --year 2007 --split trainval \
                                  --save_dir data/cooccurrence/ --device cuda:0
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.voc_dataset import VOCDataset, VOC_CLASSES

NUM_VOC_CLASSES = len(VOC_CLASSES)  # 20


def compute_obj_obj(voc_root, year, split, save_path):
    """
    Object-Object co-occurrence matrix (O x O).
    E_obj_obj[i,j] = number of images containing both class i and class j.
    Diagonal is 0 (self-loops learned adaptively in GCR).
    """
    dataset = VOCDataset(voc_root, year=year, split=split)
    matrix = np.zeros((NUM_VOC_CLASSES, NUM_VOC_CLASSES), dtype=np.float32)

    for _, target in tqdm(dataset, desc="Object-Object co-occurrence"):
        labels = target["labels"].numpy()  # 1-indexed (1~20)
        # Convert to 0-indexed
        present = set(int(l) - 1 for l in labels)
        for i in present:
            for j in present:
                if i != j:
                    matrix[i, j] += 1.0

    # Zero diagonal (self-loop handled by GCR)
    np.fill_diagonal(matrix, 0)
    np.save(save_path, matrix)
    print(f"Saved Object-Object matrix {matrix.shape} → {save_path}")
    return matrix


def compute_obj_scene(voc_root, year, split, save_dir, device="cpu",
                      scene_detector=None):
    """
    Object-Scene co-occurrence matrices using the Scene Detector.

    For each image, runs the scene detector to get:
      - indoor/outdoor label (argmax → index 0 or 1)
      - top-1 place category (argmax → index 0~364)
      - top-1 scene attribute (argmax → index 0~101)

    Then counts co-occurrence with detected objects.

    Args:
        scene_detector: SceneDetector model (must be on `device`)
    """
    from models.scene_detector import SceneDetector

    if scene_detector is None:
        print("[Cooccurrence] Initializing Scene Detector...")
        scene_detector = SceneDetector(freeze=True).to(device)
        scene_detector.eval()

    dataset = VOCDataset(voc_root, year=year, split=split)

    # Normalization for VGG-16 (ImageNet stats, also applicable for Places365)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize,
    ])

    obj_inout = np.zeros((NUM_VOC_CLASSES, 2), dtype=np.float32)
    obj_place = np.zeros((NUM_VOC_CLASSES, 365), dtype=np.float32)
    obj_attr  = np.zeros((NUM_VOC_CLASSES, 102), dtype=np.float32)

    with torch.no_grad():
        for img, target in tqdm(dataset, desc="Object-Scene co-occurrence"):
            labels = target["labels"].numpy()
            present = set(int(l) - 1 for l in labels)
            if not present:
                continue

            # Preprocess image
            img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1,3,224,224)

            scene_out, _ = scene_detector(img_tensor)
            inout_scores = scene_out["inout_scores"][0]  # (2,)
            place_scores = scene_out["place_scores"][0]  # (365,)
            attr_scores  = scene_out["attr_scores"][0]   # (102,)

            # Use argmax (hard assignment)
            inout_idx = int(torch.argmax(inout_scores).item())
            place_idx = int(torch.argmax(place_scores).item())
            attr_idx  = int(torch.argmax(attr_scores).item())

            for cls_idx in present:
                obj_inout[cls_idx, inout_idx] += 1.0
                obj_place[cls_idx, place_idx] += 1.0
                obj_attr[cls_idx, attr_idx] += 1.0

    inout_path = os.path.join(save_dir, f"obj_inout_voc{year}.npy")
    place_path = os.path.join(save_dir, f"obj_place_voc{year}.npy")
    attr_path  = os.path.join(save_dir, f"obj_attr_voc{year}.npy")

    np.save(inout_path, obj_inout)
    np.save(place_path, obj_place)
    np.save(attr_path,  obj_attr)

    print(f"Saved Object-IndoorOutdoor {obj_inout.shape} → {inout_path}")
    print(f"Saved Object-Place         {obj_place.shape} → {place_path}")
    print(f"Saved Object-Attribute     {obj_attr.shape}  → {attr_path}")


def load_cooccurrence_matrices(save_dir, year="2007"):
    """Load all 4 co-occurrence matrices from disk."""
    obj_obj   = np.load(os.path.join(save_dir, f"obj_obj_voc{year}.npy"))
    obj_inout = np.load(os.path.join(save_dir, f"obj_inout_voc{year}.npy"))
    obj_place = np.load(os.path.join(save_dir, f"obj_place_voc{year}.npy"))
    obj_attr  = np.load(os.path.join(save_dir, f"obj_attr_voc{year}.npy"))
    return {
        "obj_obj":   torch.from_numpy(obj_obj).float(),
        "obj_inout": torch.from_numpy(obj_inout).float(),
        "obj_place": torch.from_numpy(obj_place).float(),
        "obj_attr":  torch.from_numpy(obj_attr).float(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc_root", default="data/VOCdevkit")
    parser.add_argument("--year", default="2007")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--save_dir", default="data/cooccurrence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--obj_obj_only", action="store_true",
                        help="Only compute object-object matrix (no scene detector needed)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    obj_obj_path = os.path.join(args.save_dir, f"obj_obj_voc{args.year}.npy")
    compute_obj_obj(args.voc_root, args.year, args.split, obj_obj_path)

    if not args.obj_obj_only:
        device = args.device if torch.cuda.is_available() else "cpu"
        compute_obj_scene(args.voc_root, args.year, args.split,
                          args.save_dir, device=device)
