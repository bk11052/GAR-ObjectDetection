"""
GCR heterogeneous graph visualization.

Given a single image and a trained GAR model, extracts the adjacency matrix
from the GCR module and visualizes the graph structure.

Usage:
    python visualize/vis_graph.py \
        --checkpoint checkpoints/stage2_best.pth \
        --image data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
        --output_dir outputs/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.voc_dataset import VOC_CLASSES
from utils.cooccurrence import load_cooccurrence_matrices
from models.gar import GARDetector
from models.scene_detector import SceneDetector


def extract_graph(model, image_tensor, cooc_matrices, device, K=3):
    """
    Run the model partially to extract the GCR graph components.

    Returns:
        adj_matrix: (N+S, N+S) numpy array
        node_labels: list of str for each node
        node_types: list of 'instance' or 'scene' for each node
        instance_scores: (N, C+1) cursory class scores
    """
    model.eval()
    with torch.no_grad():
        img = image_tensor.to(device)
        image_list = model._to_image_list([img])
        features = model.backbone(image_list.tensors)
        feature_map = {"0": features}

        proposals, _ = model.rpn(image_list, feature_map)
        box_features = model.roi_pool(
            feature_map, proposals, image_list.image_sizes
        )
        instance_features = model.box_head(box_features)
        cursory_scores, _ = model.box_predictor(instance_features)

        # Limit to top-N proposals by objectness (for cleaner graph)
        N = min(proposals[0].shape[0], 20)
        inst_feat = instance_features[:N]
        curs_sc = cursory_scores[:N]

        # Scene detection
        img_224 = T.functional.resize(img, [224, 224])
        img_norm = model.scene_normalize(img_224).unsqueeze(0)
        scene_out, conv5_feat = model.scene_detector(img_norm)

        # Build edges
        obj_obj = cooc_matrices["obj_obj"].to(device)
        obj_obj_norm = F.softmax(obj_obj, dim=1)

        E_ins = model.gcr.build_instance_edges(curs_sc, obj_obj_norm)
        E_scene, _, place_idx, attr_idx = model.gcr.build_scene_edges(
            curs_sc, scene_out, cooc_matrices
        )
        A_tilde = model.gcr.build_adj_matrix(E_ins, E_scene)

    adj = A_tilde.cpu().numpy()

    # Node labels
    fg_scores = curs_sc[:, 1:].cpu()
    cls_indices = torch.argmax(fg_scores, dim=1)
    cls_confs = torch.max(F.softmax(curs_sc, dim=1)[:, 1:], dim=1).values

    node_labels = []
    node_types = []

    for i in range(N):
        cls_name = VOC_CLASSES[cls_indices[i].item()]
        conf = cls_confs[i].item()
        node_labels.append(f"{cls_name}\n{conf:.2f}")
        node_types.append("instance")

    # Scene node labels
    S = E_scene.shape[1]  # 2 + 2K
    inout_names = ["Indoor", "Outdoor"]
    for name in inout_names:
        node_labels.append(name)
        node_types.append("scene")

    place_idx_cpu = place_idx.cpu().numpy()
    for idx in place_idx_cpu:
        node_labels.append(f"Place_{idx}")
        node_types.append("scene")

    attr_idx_cpu = attr_idx.cpu().numpy()
    for idx in attr_idx_cpu:
        node_labels.append(f"Attr_{idx}")
        node_types.append("scene")

    return adj, node_labels, node_types, curs_sc.cpu()


def visualize_graph(adj, node_labels, node_types, image_path, save_path,
                    edge_threshold=0.05):
    """
    Draw the heterogeneous graph with networkx.
    """
    num_nodes = len(node_labels)
    G = nx.DiGraph()

    for i in range(num_nodes):
        G.add_node(i, label=node_labels[i], ntype=node_types[i])

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj[i, j] > edge_threshold:
                G.add_edge(i, j, weight=adj[i, j])

    # Layout
    instance_nodes = [i for i, t in enumerate(node_types) if t == "instance"]
    scene_nodes = [i for i, t in enumerate(node_types) if t == "scene"]

    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: original image
    if os.path.exists(image_path):
        img = Image.open(image_path)
        axes[0].imshow(img)
        axes[0].set_title("Input Image", fontsize=14)
        axes[0].axis("off")
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center")
        axes[0].axis("off")

    ax = axes[1]

    # Draw edges
    edges = G.edges(data=True)
    weights = [d["weight"] * 3 for _, _, d in edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=weights,
        alpha=0.3,
        edge_color="gray",
        arrows=True,
        arrowsize=10,
    )

    # Draw instance nodes (blue)
    nx.draw_networkx_nodes(
        G, pos, nodelist=instance_nodes, ax=ax,
        node_color="#4A90D9",
        node_size=600,
        alpha=0.9,
    )

    # Draw scene nodes (green)
    nx.draw_networkx_nodes(
        G, pos, nodelist=scene_nodes, ax=ax,
        node_color="#50C878",
        node_size=800,
        node_shape="s",
        alpha=0.9,
    )

    # Labels
    labels = {i: node_labels[i] for i in range(num_nodes)}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7,
        font_weight="bold",
    )

    # Legend
    inst_patch = mpatches.Patch(color="#4A90D9", label="Instance Node (object)")
    scene_patch = mpatches.Patch(color="#50C878", label="Scene Node")
    ax.legend(handles=[inst_patch, scene_patch], loc="upper left", fontsize=10)

    ax.set_title("GCR Heterogeneous Graph", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True,
                        help="Path to a single VOC image")
    parser.add_argument("--cooc_dir", default="data/cooccurrence")
    parser.add_argument("--config", default="configs/gar_voc.yaml")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--edge_threshold", type=float, default=0.05)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = GARDetector(num_classes=20, K=3, pretrained_backbone=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load co-occurrence
    cooc = load_cooccurrence_matrices(args.cooc_dir, year="2007")
    cooc = {k: v.to(device) for k, v in cooc.items()}

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_tensor = T.functional.to_tensor(img)

    # Extract graph
    adj, labels, types, _ = extract_graph(model, img_tensor, cooc, device)

    # Visualize
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(args.output_dir, f"graph_{img_name}.png")
    visualize_graph(adj, labels, types, args.image, save_path,
                    edge_threshold=args.edge_threshold)


if __name__ == "__main__":
    main()
