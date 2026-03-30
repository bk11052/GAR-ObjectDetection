"""
GAR Training Script - Two-stage training on PASCAL VOC 2007.

Usage:
  # Compute co-occurrence matrices first (once):
  CUDA_VISIBLE_DEVICES=0 python utils/cooccurrence.py \
      --voc_root data/VOCdevkit --year 2007 --split trainval \
      --save_dir data/cooccurrence/

  # Stage 1: Train Faster R-CNN backbone (4 epochs)
  CUDA_VISIBLE_DEVICES=0 python train.py --stage 1

  # Stage 2: Joint training with GCR (6 epochs)
  CUDA_VISIBLE_DEVICES=0 python train.py --stage 2 \
      --resume checkpoints/stage1_best.pth
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.voc_dataset import VOCDataset, collate_fn, get_voc_transforms
from utils.cooccurrence import load_cooccurrence_matrices
from models.gar import GARDetector


def parse_args():
    parser = argparse.ArgumentParser(description="GAR Training")
    parser.add_argument("--config", default="configs/gar_voc.yaml")
    parser.add_argument("--stage", type=int, default=1,
                        help="1=backbone only, 2=full GAR with GCR")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_optimizer(model, cfg, stage):
    params = []
    if stage == 1:
        # Stage 1: train only Faster R-CNN components (backbone + RPN + box head)
        for name, param in model.named_parameters():
            if "scene_detector" in name or "gcr" in name or "scene_node_embed" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                params.append(param)
    else:
        # Stage 2: train only GCR + box_predictor + scene_node_embed
        # Freeze backbone, RPN, box_head, and scene_detector
        trainable = {"gcr.", "box_predictor.", "scene_node_embed."}
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in trainable):
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False

    lr = cfg["training"][f"stage{stage}_lr"]
    optimizer = optim.SGD(
        params,
        lr=lr,
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    return optimizer


def train_one_epoch(model, optimizer, data_loader, device, stage,
                    cooc_matrices, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Move co-occurrence to device once
    cooc = {k: v.to(device) for k, v in cooc_matrices.items()}

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Stage {stage}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            losses = model(images, targets=targets,
                           cooc_matrices=cooc, stage=stage)
        except Exception as e:
            print(f"\n[WARNING] Skipping batch due to error: {e}")
            continue

        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, stage, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved → {path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_set = VOCDataset(
        root=cfg["dataset"]["root"],
        year="2007",
        split="trainval",
        transforms=get_voc_transforms(train=True),
    )
    data_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Dataset: {len(train_set)} training images")

    # Model
    model = GARDetector(
        num_classes=cfg["dataset"]["num_classes"],
        K=cfg["gcr"]["K"],
        pretrained_backbone=cfg["model"]["pretrained"],
    ).to(device)

    # Load co-occurrence matrices (must be pre-computed)
    cooc_dir = cfg["cooccurrence"]["save_dir"]
    if not os.path.exists(os.path.join(cooc_dir, f"obj_obj_voc2007.npy")):
        print("[ERROR] Co-occurrence matrices not found!")
        print("Run first: CUDA_VISIBLE_DEVICES=0 python utils/cooccurrence.py")
        sys.exit(1)
    cooc_matrices = load_cooccurrence_matrices(cooc_dir, year="2007")
    print("Co-occurrence matrices loaded.")

    # Optimizer
    optimizer = get_optimizer(model, cfg, args.stage)

    # Resume
    start_epoch = 1
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt["model_state_dict"]
        # Remove keys with shape mismatch (e.g. gcr.fc_out changed dims)
        model_state = model.state_dict()
        for k in list(state.keys()):
            if k in model_state and state[k].shape != model_state[k].shape:
                print(f"  Skipping {k}: checkpoint {state[k].shape} vs model {model_state[k].shape}")
                del state[k]
        model.load_state_dict(state, strict=False)
        if ckpt["stage"] == args.stage:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1

    # Training loop
    num_epochs = cfg["training"][f"stage{args.stage}_epochs"]
    print(f"\n=== Stage {args.stage} Training: {num_epochs} epochs ===")

    # LR decay: halve LR at 75% of training
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(num_epochs * 0.75)],
        gamma=0.1,
    )

    ckpt_dir = cfg["training"]["checkpoint_dir"]
    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = train_one_epoch(
            model, optimizer, data_loader, device,
            args.stage, cooc_matrices, epoch
        )
        scheduler.step()

        print(f"Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(ckpt_dir, f"stage{args.stage}_epoch{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, args.stage, ckpt_path)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, f"stage{args.stage}_best.pth")
            save_checkpoint(model, optimizer, epoch, args.stage, best_path)

    print(f"\nStage {args.stage} training complete. Best loss: {best_loss:.4f}")
    if args.stage == 1:
        print("\nNext step: run Stage 2 training with:")
        print(f"  CUDA_VISIBLE_DEVICES={args.gpu} python train.py --stage 2 "
              f"--resume {os.path.join(ckpt_dir, 'stage1_best.pth')}")


if __name__ == "__main__":
    main()
