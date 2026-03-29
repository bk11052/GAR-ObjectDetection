"""
Co-occurrence matrix visualization.

Generates heatmaps:
  1. Object-Object (20x20)
  2. Object-Indoor/Outdoor (20x2)
  3. Object-Place top-10 (20x10)
  4. Object-Attribute top-10 (20x10)

Usage:
    python visualize/vis_cooccurrence.py \
        --cooc_dir data/cooccurrence/ \
        --output_dir outputs/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.voc_dataset import VOC_CLASSES


def plot_obj_obj(matrix, save_path):
    """20x20 Object-Object co-occurrence heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=VOC_CLASSES,
        yticklabels=VOC_CLASSES,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Object-Object Co-occurrence (VOC 2007 trainval)", fontsize=14)
    ax.set_xlabel("Object Class")
    ax.set_ylabel("Object Class")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_obj_inout(matrix, save_path):
    """20x2 Object-Indoor/Outdoor heatmap."""
    fig, ax = plt.subplots(figsize=(5, 10))
    sns.heatmap(
        matrix,
        xticklabels=["Indoor", "Outdoor"],
        yticklabels=VOC_CLASSES,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Object - Indoor/Outdoor", fontsize=14)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_obj_scene_topk(matrix, labels, title, save_path, k=10):
    """Object vs top-K scene labels heatmap."""
    # Sum across objects to find the most frequent scene labels
    col_sums = matrix.sum(axis=0)
    top_indices = np.argsort(-col_sums)[:k]
    sub_matrix = matrix[:, top_indices]

    if labels is not None:
        col_labels = [labels[i] for i in top_indices]
    else:
        col_labels = [str(i) for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        sub_matrix,
        xticklabels=col_labels,
        yticklabels=VOC_CLASSES,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def load_place_labels():
    """Try to load Places365 category names."""
    # Common location for Places365 categories
    candidates = [
        "data/categories_places365.txt",
        os.path.expanduser("~/.cache/places365/categories_places365.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            labels = []
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        name = parts[0].split("/")[-1]
                        labels.append(name)
            return labels
    # Fallback: use indices
    return [f"place_{i}" for i in range(365)]


def load_attr_labels():
    """Try to load scene attribute names."""
    candidates = [
        "data/scene_attributes.txt",
        os.path.expanduser("~/.cache/places365/scene_attributes.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            labels = []
            with open(path) as f:
                for line in f:
                    labels.append(line.strip())
            return labels
    return [f"attr_{i}" for i in range(102)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooc_dir", default="data/cooccurrence")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--year", default="2007")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    year = args.year

    # Load matrices
    obj_obj = np.load(os.path.join(args.cooc_dir, f"obj_obj_voc{year}.npy"))
    obj_inout = np.load(os.path.join(args.cooc_dir, f"obj_inout_voc{year}.npy"))
    obj_place = np.load(os.path.join(args.cooc_dir, f"obj_place_voc{year}.npy"))
    obj_attr = np.load(os.path.join(args.cooc_dir, f"obj_attr_voc{year}.npy"))

    # 1. Object-Object
    plot_obj_obj(obj_obj, os.path.join(args.output_dir, "cooc_obj_obj.png"))

    # 2. Object-Indoor/Outdoor
    plot_obj_inout(obj_inout, os.path.join(args.output_dir, "cooc_obj_inout.png"))

    # 3. Object-Place top-10
    place_labels = load_place_labels()
    plot_obj_scene_topk(
        obj_place, place_labels,
        "Object - Place Categories (Top 10)",
        os.path.join(args.output_dir, "cooc_obj_place_top10.png"),
        k=10,
    )

    # 4. Object-Attribute top-10
    attr_labels = load_attr_labels()
    plot_obj_scene_topk(
        obj_attr, attr_labels,
        "Object - Scene Attributes (Top 10)",
        os.path.join(args.output_dir, "cooc_obj_attr_top10.png"),
        k=10,
    )

    print(f"\nAll co-occurrence visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
