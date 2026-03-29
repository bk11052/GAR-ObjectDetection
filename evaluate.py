"""
GAR Evaluation Script on PASCAL VOC 2007 test set.

Computes per-class AP and mAP using VOC 2007 evaluation protocol.

Usage:
  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
      --checkpoint checkpoints/stage2_best.pth \
      --stage 2
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.voc_dataset import VOCDataset, collate_fn, VOC_CLASSES, get_voc_transforms
from utils.cooccurrence import load_cooccurrence_matrices
from models.gar import GARDetector


def parse_args():
    parser = argparse.ArgumentParser(description="GAR Evaluation")
    parser.add_argument("--config", default="configs/gar_voc.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", type=int, default=2,
                        help="1=baseline Faster R-CNN, 2=full GAR")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def voc_ap(rec, prec):
    """Compute VOC 2007 AP using 11-point interpolation."""
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = prec[rec >= thr]
        if prec_at_rec.size > 0:
            ap += np.max(prec_at_rec) / 11.0
    return ap


def evaluate_detections(all_detections, all_gt, num_classes, iou_thresh=0.5):
    """
    Evaluate detections using VOC 2007 protocol.

    Args:
        all_detections: list of dicts per image, each with 'boxes'(N,4),
                        'scores'(N,), 'labels'(N,) - 1-indexed labels
        all_gt: list of dicts per image, each with 'boxes'(M,4),
                'labels'(M,) - 1-indexed, 'difficult'(M,)

    Returns:
        per_class_ap: dict {class_name: AP}
        mAP: float
    """
    per_class_ap = {}

    for cls_idx in range(1, num_classes + 1):
        cls_name = VOC_CLASSES[cls_idx - 1]

        # Collect all detections for this class across all images
        det_scores = []
        det_boxes = []
        det_img_ids = []

        gt_boxes_per_img = {}     # img_id → (M, 4) GT boxes
        gt_difficult_per_img = {} # img_id → (M,) difficult flags
        gt_detected_per_img = {}  # img_id → (M,) bool, whether GT was matched

        for img_id, (dets, gt) in enumerate(zip(all_detections, all_gt)):
            # GT for this class
            gt_mask = gt["labels"] == cls_idx
            gt_b = gt["boxes"][gt_mask].numpy()  # (M, 4)
            gt_d = gt["difficult"][gt_mask].numpy() if "difficult" in gt else \
                   np.zeros(gt_b.shape[0], dtype=bool)

            gt_boxes_per_img[img_id] = gt_b
            gt_difficult_per_img[img_id] = gt_d.astype(bool)
            gt_detected_per_img[img_id] = np.zeros(gt_b.shape[0], dtype=bool)

            # Detections for this class
            det_mask = dets["labels"] == cls_idx
            if det_mask.sum() > 0:
                d_boxes  = dets["boxes"][det_mask].numpy()
                d_scores = dets["scores"][det_mask].numpy()
                for b, s in zip(d_boxes, d_scores):
                    det_boxes.append(b)
                    det_scores.append(s)
                    det_img_ids.append(img_id)

        if not det_scores:
            per_class_ap[cls_name] = 0.0
            continue

        # Sort by score descending
        sorted_idx = np.argsort(-np.array(det_scores))
        det_scores_s = np.array(det_scores)[sorted_idx]
        det_boxes_s  = np.array(det_boxes)[sorted_idx]
        det_img_ids_s = np.array(det_img_ids)[sorted_idx]

        tp = np.zeros(len(det_scores_s))
        fp = np.zeros(len(det_scores_s))
        num_gt = sum(
            (~gt_difficult_per_img[i]).sum()
            for i in range(len(all_gt))
        )

        for d_idx in range(len(det_scores_s)):
            img_id = det_img_ids_s[d_idx]
            d_box  = det_boxes_s[d_idx]

            gt_b = gt_boxes_per_img[img_id]
            gt_d = gt_difficult_per_img[img_id]
            gt_det = gt_detected_per_img[img_id]

            if gt_b.shape[0] == 0:
                fp[d_idx] = 1
                continue

            # IoU with all GT boxes
            ious = _compute_iou(d_box, gt_b)  # (M,)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_thresh:
                if gt_d[max_iou_idx]:
                    # Difficult GT → ignore
                    pass
                elif not gt_det[max_iou_idx]:
                    tp[d_idx] = 1
                    gt_detected_per_img[img_id][max_iou_idx] = True
                else:
                    fp[d_idx] = 1  # duplicate detection
            else:
                fp[d_idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec  = cum_tp / max(num_gt, 1)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-10)

        ap = voc_ap(rec, prec)
        per_class_ap[cls_name] = ap

    mAP = np.mean(list(per_class_ap.values())) * 100
    return per_class_ap, mAP


def _compute_iou(box, gt_boxes):
    """Compute IoU between one box (4,) and gt_boxes (M, 4)."""
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_gt  = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_box + area_gt - inter
    return inter / np.maximum(union, 1e-10)


@torch.no_grad()
def run_inference(model, data_loader, device, stage, cooc_matrices):
    model.eval()
    all_detections = []
    all_gt = []

    cooc = {k: v.to(device) for k, v in cooc_matrices.items()}

    for images, targets in tqdm(data_loader, desc="Inference"):
        images = [img.to(device) for img in images]

        if stage == 1:
            dets = model(images, cooc_matrices=cooc, stage=1)
        else:
            dets = model(images, cooc_matrices=cooc, stage=2)

        all_detections.extend([{k: v.cpu() for k, v in d.items()} for d in dets])
        all_gt.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    return all_detections, all_gt


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset (test split)
    test_set = VOCDataset(
        root=cfg["dataset"]["root"],
        year="2007",
        split="test",
        transforms=get_voc_transforms(train=False),
    )

    data_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
    )
    print(f"Test set: {len(test_set)} images")

    # Model
    model = GARDetector(
        num_classes=cfg["dataset"]["num_classes"],
        K=cfg["gcr"]["K"],
        pretrained_backbone=False,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Co-occurrence matrices
    cooc_matrices = load_cooccurrence_matrices(
        cfg["cooccurrence"]["save_dir"], year="2007"
    )

    # Inference
    all_detections, all_gt = run_inference(
        model, data_loader, device, args.stage, cooc_matrices
    )

    # Evaluate
    per_class_ap, mAP = evaluate_detections(
        all_detections, all_gt,
        num_classes=cfg["dataset"]["num_classes"],
        iou_thresh=args.iou_thresh,
    )

    # Print results
    print("\n" + "="*60)
    print(f"{'Class':<20} {'AP':>8}")
    print("-"*60)
    for cls_name, ap in per_class_ap.items():
        print(f"{cls_name:<20} {ap*100:>8.2f}%")
    print("="*60)
    print(f"{'mAP':<20} {mAP:>8.2f}%")
    print("="*60)
    print(f"\nTarget (paper): 76.1% mAP on VOC 2007")


if __name__ == "__main__":
    main()
