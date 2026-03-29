"""
GAR: Graph Assisted Reasoning for Object Detection
Full model integrating:
  1. VGG-16 backbone → shared convolutional features
  2. RPN → region proposals
  3. cRCN (Box Head + Box Predictor) → cursory scores + bbox regression
  4. Scene Detector (Places365) → scene labels + conv5 features
  5. Scene Node Embedding → S scene node vectors
  6. GCR module (2-layer GCN) → cogitative scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.detection.rpn import (
    AnchorGenerator, RPNHead, RegionProposalNetwork
)
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection._utils import BoxCoder
from torchvision.ops import MultiScaleRoIAlign, box_iou, nms
import torchvision.transforms as T

from .scene_detector import SceneDetector, SceneNodeEmbedding
from .gcr_module import GCRModule


class VGG16BoxHead(nn.Module):
    """Two FC layers: ROI-pooled features (512*7*7) → 4096-dim."""
    def __init__(self, in_channels=512, representation_size=4096):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.out_channels = representation_size

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class GARBoxPredictor(nn.Module):
    """Classification + bbox regression heads."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x)


class GARDetector(nn.Module):
    """
    Full GAR detector with VGG-16 backbone.

    Stage 1: Train Faster R-CNN only (backbone + RPN + cRCN).
    Stage 2: Add GCR module on top of cursory scores.

    Args:
        num_classes: number of foreground classes (20 for VOC)
        K: top-K place/attribute scene nodes (default 3)
    """

    def __init__(self, num_classes=20, K=3, pretrained_backbone=True,
                 scene_weights_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))

        # ---- VGG-16 backbone ----
        vgg = vgg16(weights="IMAGENET1K_V1" if pretrained_backbone else None)
        self.backbone = vgg.features  # conv1_1 ~ conv5_3, stride=32, 512ch

        # ---- RPN ----
        # Single feature map → all anchor sizes in one tuple
        self.anchor_gen = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        rpn_head = RPNHead(
            in_channels=512,
            num_anchors=self.anchor_gen.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_gen,
            head=rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": 2000, "testing": 1000},
            post_nms_top_n={"training": 2000, "testing": 300},
            nms_thresh=0.7,
        )

        # ---- ROI Pooling ----
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2,
        )

        # ---- Box Head (cRCN) ----
        self.box_head = VGG16BoxHead(in_channels=512, representation_size=4096)
        self.box_predictor = GARBoxPredictor(
            in_channels=4096, num_classes=num_classes + 1,
        )

        # ---- Scene Detector (frozen) ----
        self.scene_detector = SceneDetector(
            pretrained_weights_path=scene_weights_path, freeze=True,
        )

        # ---- Scene Node Embedding ----
        self.scene_node_embed = SceneNodeEmbedding(
            in_channels=512, feature_dim=4096, num_scene_labels=469,
        )

        # ---- GCR Module ----
        self.gcr = GCRModule(
            num_classes=num_classes, feature_dim=4096,
            gcn_hidden=4096, gcn_out=512, K=K,
        )

        # Scene image preprocessing
        self.scene_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )

    # ------------------------------------------------------------------
    # Image batching
    # ------------------------------------------------------------------
    def _to_image_list(self, images):
        """Pad variable-size images into a batch tensor → ImageList."""
        image_sizes = [img.shape[-2:] for img in images]
        max_h = max(s[0] for s in image_sizes)
        max_w = max(s[1] for s in image_sizes)
        batch = torch.zeros(len(images), 3, max_h, max_w,
                            device=images[0].device)
        for i, img in enumerate(images):
            batch[i, :, :img.shape[-2], :img.shape[-1]] = img
        return ImageList(batch, image_sizes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images, targets=None, cooc_matrices=None, stage=1):
        """
        Args:
            images:  list of (3, H, W) float tensors
            targets: list of target dicts (training only)
            cooc_matrices: dict of co-occurrence tensors
            stage: 1 = Faster R-CNN only, 2 = full GAR with GCR

        Returns:
            Training (targets is not None): dict of loss tensors
            Inference (targets is None): list of detection dicts
        """
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        is_training = targets is not None

        # ---- Backbone ----
        image_list = self._to_image_list(images)
        features = self.backbone(image_list.tensors)  # (B, 512, H/32, W/32)
        feature_map = {"0": features}

        # ---- RPN ----
        if is_training:
            proposals, rpn_losses = self.rpn(image_list, feature_map, targets)
        else:
            proposals, _ = self.rpn(image_list, feature_map)
            rpn_losses = {}

        # ---- ROI Pooling → Box Head → cRCN ----
        box_features = self.roi_pool(
            feature_map, proposals, image_list.image_sizes
        )
        instance_features = self.box_head(box_features)   # (total_N, 4096)
        cursory_scores, bbox_deltas = self.box_predictor(instance_features)

        # ---- Select scores to use ----
        if stage == 2 and cooc_matrices is not None:
            final_scores = self._run_gcr(
                images, instance_features, cursory_scores,
                proposals, cooc_matrices, device
            )
        else:
            final_scores = cursory_scores

        # ---- Training: compute losses / Inference: post-process ----
        if is_training:
            return self._compute_losses(
                final_scores, bbox_deltas, proposals,
                image_list.image_sizes, targets, rpn_losses
            )
        else:
            return self._post_process(
                final_scores, bbox_deltas, proposals,
                image_list.image_sizes
            )

    # ------------------------------------------------------------------
    # GCR reasoning (Stage 2)
    # ------------------------------------------------------------------
    def _run_gcr(self, images, instance_features, cursory_scores,
                 proposals, cooc_matrices, device):
        """Run GCR per image, return enhanced cogitative scores."""
        proposal_counts = [len(p) for p in proposals]
        split_features = torch.split(instance_features, proposal_counts)
        split_cursory = torch.split(cursory_scores, proposal_counts)

        cogitative_list = []
        for img_idx, (inst_feat, curs_sc) in enumerate(
            zip(split_features, split_cursory)
        ):
            img = images[img_idx]
            img_224 = T.functional.resize(img, [224, 224])
            img_norm = self.scene_normalize(img_224).unsqueeze(0)

            with torch.no_grad():
                scene_out, conv5_feat = self.scene_detector(img_norm)

            K = self.K
            _, place_idx = torch.topk(scene_out["place_scores"][0], K)
            _, attr_idx = torch.topk(scene_out["attr_scores"][0], K)

            inout_idx = torch.tensor([0, 1], device=device)
            place_idx_global = place_idx + 2
            attr_idx_global = attr_idx + 2 + 365
            selected_idx = torch.cat([inout_idx, place_idx_global,
                                      attr_idx_global])

            scene_nodes = self.scene_node_embed(
                conv5_feat, selected_idx.unsqueeze(0)
            ).squeeze(0)  # (S, 4096)

            cog_sc = self.gcr(
                inst_feat, curs_sc, scene_out, scene_nodes, cooc_matrices,
            )
            cogitative_list.append(cog_sc)

        return torch.cat(cogitative_list, dim=0)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def _compute_losses(self, cls_logits, bbox_deltas, proposals,
                        image_sizes, targets, rpn_losses):
        """Classification + regression loss with proper target encoding."""
        # Assign proposals to GT
        all_labels = []
        all_regression_targets = []
        all_proposals = []

        for props, tgt in zip(proposals, targets):
            gt_boxes = tgt["boxes"].to(props.device)
            gt_labels = tgt["labels"].to(props.device)

            if gt_boxes.numel() == 0:
                labels = torch.zeros(props.shape[0], dtype=torch.int64,
                                     device=props.device)
                reg_targets = torch.zeros_like(props)
            else:
                ious = box_iou(props, gt_boxes)
                max_iou, matched_idxs = ious.max(dim=1)

                labels = gt_labels[matched_idxs].clone()
                labels[max_iou < 0.5] = 0
                labels[(max_iou >= 0.4) & (max_iou < 0.5)] = -1  # ignore

                matched_gt_boxes = gt_boxes[matched_idxs]
                reg_targets = self.box_coder.encode(
                    [matched_gt_boxes], [props]
                )[0]

            all_labels.append(labels)
            all_regression_targets.append(reg_targets)
            all_proposals.append(props)

        labels = torch.cat(all_labels, dim=0)
        regression_targets = torch.cat(all_regression_targets, dim=0)

        # Subsample
        pos_inds, neg_inds = self._subsample(labels)
        sampled = torch.cat([pos_inds, neg_inds], dim=0)

        # Classification loss (all sampled)
        cls_loss = F.cross_entropy(cls_logits[sampled], labels[sampled])

        # Regression loss (positive samples only)
        if pos_inds.numel() > 0:
            pos_labels = labels[pos_inds]
            # Per-class regression: select deltas for the matched class
            pos_deltas = bbox_deltas[pos_inds]
            # Reshape to (N_pos, num_classes+1, 4)
            pos_deltas = pos_deltas.view(pos_inds.numel(), -1, 4)
            # Select the class-specific deltas
            pos_deltas = pos_deltas[
                torch.arange(pos_inds.numel(), device=pos_deltas.device),
                pos_labels
            ]  # (N_pos, 4)
            pos_reg_targets = regression_targets[pos_inds]
            reg_loss = F.smooth_l1_loss(pos_deltas, pos_reg_targets,
                                        beta=1.0, reduction="sum")
            reg_loss = reg_loss / max(sampled.numel(), 1)
        else:
            reg_loss = torch.tensor(0.0, device=cls_logits.device)

        losses = {
            "loss_classifier": cls_loss,
            "loss_box_reg": reg_loss,
        }
        losses.update(rpn_losses)
        return losses

    def _subsample(self, labels, batch_size=512, pos_fraction=0.25):
        pos = (labels > 0).nonzero(as_tuple=True)[0]
        neg = (labels == 0).nonzero(as_tuple=True)[0]

        num_pos = min(pos.numel(), int(batch_size * pos_fraction))
        num_neg = min(neg.numel(), batch_size - num_pos)

        perm_pos = torch.randperm(pos.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(neg.numel(), device=labels.device)[:num_neg]

        return pos[perm_pos], neg[perm_neg]

    # ------------------------------------------------------------------
    # Post-processing (inference)
    # ------------------------------------------------------------------
    def _post_process(self, cls_logits, bbox_deltas, proposals, image_sizes):
        """Decode predictions → NMS → final detections."""
        scores = F.softmax(cls_logits, dim=-1)

        detections = []
        start = 0
        for img_idx, (props, img_size) in enumerate(zip(proposals, image_sizes)):
            N = props.shape[0]
            sc = scores[start:start + N]
            bd = bbox_deltas[start:start + N]
            start += N

            # Decode: bbox_deltas (N, (C+1)*4) + proposals (N, 4)
            bd_reshaped = bd.view(N, -1, 4)   # (N, C+1, 4)
            props_exp = props.unsqueeze(1).expand_as(bd_reshaped)  # (N, C+1, 4)
            boxes = self.box_coder.decode(
                bd_reshaped.reshape(-1, 4),
                props_exp.reshape(-1, 4),
            ).view(N, -1, 4)  # (N, C+1, 4)

            # Clip to image
            boxes[:, :, 0].clamp_(min=0)
            boxes[:, :, 1].clamp_(min=0)
            boxes[:, :, 2].clamp_(max=img_size[1])
            boxes[:, :, 3].clamp_(max=img_size[0])

            result_boxes = []
            result_labels = []
            result_scores = []
            for cls_idx in range(1, self.num_classes + 1):
                cls_scores = sc[:, cls_idx]
                cls_boxes = boxes[:, cls_idx, :]

                keep = cls_scores > 0.05
                cls_scores = cls_scores[keep]
                cls_boxes = cls_boxes[keep]

                if cls_scores.numel() == 0:
                    continue

                keep_nms = nms(cls_boxes, cls_scores, iou_threshold=0.5)
                result_boxes.append(cls_boxes[keep_nms])
                result_scores.append(cls_scores[keep_nms])
                result_labels.append(torch.full(
                    (keep_nms.numel(),), cls_idx,
                    dtype=torch.int64, device=cls_boxes.device
                ))

            if result_boxes:
                detections.append({
                    "boxes": torch.cat(result_boxes),
                    "scores": torch.cat(result_scores),
                    "labels": torch.cat(result_labels),
                })
            else:
                detections.append({
                    "boxes": torch.zeros((0, 4), device=props.device),
                    "scores": torch.zeros((0,), device=props.device),
                    "labels": torch.zeros((0,), dtype=torch.int64,
                                          device=props.device),
                })

        return detections
