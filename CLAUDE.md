# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of **GAR: Graph Assisted Reasoning for Object Detection** (WACV 2020).
GAR adds a lightweight GCN module on top of Faster R-CNN (VGG-16) to model object-object and object-scene co-occurrence relationships.

Target: **mAP 76.1%** on PASCAL VOC 2007 (baseline Faster R-CNN: 73.2%).

## Environment

- **Server**: NVIDIA RTX 5000 Ada (32GB), CUDA 12.2, GPU 0 only
- **Execution**: Docker container via `docker compose` (shared server, must use Docker)
- **Dev flow**: local edit → git push → server pull → docker run

## Docker Commands (server)

```bash
# Build
docker compose build

# Interactive shell (all training/eval runs here)
docker compose run --rm gar bash

# Or run a single command
docker compose run --rm gar python train.py --stage 1
```

## Workflow (inside Docker container)

### 1. Compute co-occurrence matrices (once)
```bash
python utils/cooccurrence.py --voc_root data/VOCdevkit --year 2007 --split trainval --save_dir data/cooccurrence/
```

### 2. Stage 1: Faster R-CNN only (4 epochs)
```bash
python train.py --stage 1
```

### 3. Stage 2: full GAR with GCR (6 epochs)
```bash
python train.py --stage 2 --resume checkpoints/stage1_best.pth
```

### 4. Evaluate
```bash
python evaluate.py --checkpoint checkpoints/stage2_best.pth --stage 2
```

## Architecture

```
Image → VGG-16 backbone → feature map (512, H/32, W/32)
         ├─ RPN → N proposals → ROI Pooling → VGG16BoxHead → instance_features (N, 4096)
         │    └─ GARBoxPredictor → cursory_scores (N, 21) + bbox_deltas (N, 84)
         │
         └─ Scene Detector (frozen, Places365) → scene_out (469-dim) + conv5_feat
              └─ SceneNodeEmbedding → scene_nodes (S, 4096)    S = 2+2K
                   └─ GCRModule:
                        build_instance_edges → E_ins (N, N)    [Algorithm 1]
                        build_scene_edges    → E_scene (N, S)  [Algorithm 2]
                        build_adj_matrix     → Ã (N+S, N+S)
                        gcn_forward          → graph_scores
                        fuse_scores          → Z = wb·Yb + wg·Yg
```

### Key Files
| File | Role |
|---|---|
| `models/gar.py` | `GARDetector` — full model with forward/loss/post-process |
| `models/gcr_module.py` | `GCRModule` — Algorithm 1 & 2 + 2-layer GCN + score fusion |
| `models/scene_detector.py` | `SceneDetector` (Places365 VGG-16) + `SceneNodeEmbedding` |
| `utils/cooccurrence.py` | Offline co-occurrence matrix computation |
| `utils/voc_dataset.py` | VOC parser; labels are **1-indexed** (0 = background) |
| `configs/gar_voc.yaml` | All hyperparameters |

### Two-Stage Training
- **Stage 1**: freezes `scene_detector`, `gcr`, `scene_node_embed`; trains backbone+RPN+box head. lr=5e-4.
- **Stage 2**: freezes `backbone` and `scene_detector`; trains GCR+box_predictor+scene_node_embed. lr=5e-5.

### Custom transforms
`utils/voc_dataset.py` defines `Compose`, `ToTensor`, `RandomHorizontalFlip` that accept `(image, target)` pairs (not standard torchvision Compose which only takes image).

### Co-occurrence Matrices
Raw counts stored as `.npy`. Row-wise softmax normalization happens at runtime inside `GCRModule`.

## Visualization

```bash
# Co-occurrence heatmaps (공출현 행렬 계산 후 실행 가능)
python visualize/vis_cooccurrence.py

# GCR graph structure (Stage 2 checkpoint 필요)
python visualize/vis_graph.py --checkpoint checkpoints/stage2_best.pth --image <image_path>

# Detection comparison: baseline vs GAR (양쪽 checkpoint 필요)
python visualize/vis_detection.py --checkpoint_s1 checkpoints/stage1_best.pth --checkpoint_s2 checkpoints/stage2_best.pth
```

All outputs go to `outputs/` (gitignored). Uses `matplotlib` Agg backend (no display needed).

### Key Files
| File | Output |
|---|---|
| `visualize/vis_cooccurrence.py` | 4 heatmap PNGs (obj-obj, obj-inout, obj-place, obj-attr) |
| `visualize/vis_graph.py` | Image + graph side-by-side (instance=blue circle, scene=green square) |
| `visualize/vis_detection.py` | Baseline vs GAR bbox comparison per image |

## Known Limitations
1. `SceneDetector` fc_inout and fc_attr heads are randomly initialized (Places365 only covers 365-class place head).
2. `_compute_losses` uses simplified proposal-GT matching (not torchvision's full `roi_heads` matcher).
