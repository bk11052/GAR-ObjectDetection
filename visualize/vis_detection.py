"""
Detection result visualization: Baseline (Stage 1) vs GAR (Stage 2) comparison.

Draws bounding boxes with class labels and confidence scores on images.
Side-by-side comparison of Faster R-CNN baseline vs GAR-enhanced results.

Usage:
    # Compare baseline vs GAR on VOC 2007 test images
    python visualize/vis_detection.py \
        --checkpoint_s1 checkpoints/stage1_best.pth \
        --checkpoint_s2 checkpoints/stage2_best.pth \
        --output_dir outputs/ \
        --num_images 10

    # Single checkpoint only
    python visualize/vis_detection.py \
        --checkpoint_s2 checkpoints/stage2_best.pth \
        --output_dir outputs/ \
        --num_images 10
"""

import os
import sys
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.voc_dataset import VOCDataset, VOC_CLASSES, get_voc_transforms
from utils.cooccurrence import load_cooccurrence_matrices
from models.gar import GARDetector


# 20 distinct colors for VOC classes
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


def run_detection(model, image_tensor, cooc_matrices, device, stage):
    """Run model inference on a single image."""
    model.eval()
    with torch.no_grad():
        dets = model(
            [image_tensor.to(device)],
            cooc_matrices={k: v.to(device) for k, v in cooc_matrices.items()},
            stage=stage,
        )
    return dets[0]  # single image


def draw_detections(ax, image, dets, title, score_thresh=0.5):
    """
    Draw bounding boxes on a matplotlib axis.

    Args:
        ax: matplotlib axis
        image: PIL Image
        dets: dict with 'boxes', 'labels', 'scores'
        title: str
        score_thresh: minimum confidence to display
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    if dets["boxes"].numel() == 0:
        ax.text(10, 30, "No detections", color="red", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        return

    boxes = dets["boxes"].cpu().numpy()
    labels = dets["labels"].cpu().numpy()
    scores = dets["scores"].cpu().numpy()

    count = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        count += 1

        cls_idx = int(label) - 1  # 1-indexed → 0-indexed
        if cls_idx < 0 or cls_idx >= len(VOC_CLASSES):
            continue

        cls_name = VOC_CLASSES[cls_idx]
        color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]

        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        text = f"{cls_name} {score:.2f}"
        ax.text(
            x1, y1 - 4,
            text,
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85),
        )

    ax.text(
        10, 25, f"{count} objects detected",
        fontsize=10, color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
    )


def visualize_comparison(model, image, image_id, cooc, device,
                         save_path, score_thresh, stage1_model=None):
    """
    Side-by-side: Stage 1 (baseline) vs Stage 2 (GAR).
    If stage1_model is None, shows only GAR result.
    """
    img_tensor = T.functional.to_tensor(image)

    if stage1_model is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        dets_s1 = run_detection(stage1_model, img_tensor, cooc, device, stage=1)
        draw_detections(ax1, image, dets_s1,
                        f"Baseline (Faster R-CNN) — {image_id}",
                        score_thresh)

        dets_s2 = run_detection(model, img_tensor, cooc, device, stage=2)
        draw_detections(ax2, image, dets_s2,
                        f"GAR (Graph Assisted) — {image_id}",
                        score_thresh)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        dets_s2 = run_detection(model, img_tensor, cooc, device, stage=2)
        draw_detections(ax, image, dets_s2,
                        f"GAR Detection — {image_id}",
                        score_thresh)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_s1", default=None,
                        help="Stage 1 (baseline) checkpoint")
    parser.add_argument("--checkpoint_s2", required=True,
                        help="Stage 2 (GAR) checkpoint")
    parser.add_argument("--voc_root", default="data/VOCdevkit")
    parser.add_argument("--cooc_dir", default="data/cooccurrence")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load GAR model (stage 2)
    model_s2 = GARDetector(num_classes=20, K=3, pretrained_backbone=False).to(device)
    ckpt = torch.load(args.checkpoint_s2, map_location=device)
    model_s2.load_state_dict(ckpt["model_state_dict"])
    model_s2.eval()

    # Load baseline model (stage 1) if provided
    model_s1 = None
    if args.checkpoint_s1:
        model_s1 = GARDetector(num_classes=20, K=3, pretrained_backbone=False).to(device)
        ckpt_s1 = torch.load(args.checkpoint_s1, map_location=device)
        model_s1.load_state_dict(ckpt_s1["model_state_dict"])
        model_s1.eval()

    # Load co-occurrence
    cooc = load_cooccurrence_matrices(args.cooc_dir, year="2007")

    # Dataset (test set)
    dataset = VOCDataset(args.voc_root, year="2007", split="test")

    # Random sample
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_images, len(dataset)))

    for i, idx in enumerate(indices):
        img_id = dataset.get_image_id(idx)
        img_path = os.path.join(dataset.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        save_path = os.path.join(args.output_dir, f"detection_{img_id}.png")
        visualize_comparison(
            model_s2, image, img_id, cooc, device, save_path,
            args.score_thresh, stage1_model=model_s1,
        )

    print(f"\n{len(indices)} detection visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
