"""
Scene Detector for GAR.

Uses VGG-16 pretrained on Places365 to extract scene features:
  - indoor/outdoor label (2-dim)
  - place categories (365-dim)
  - scene attributes (102-dim)
  total: 469-dim

The scene detector is frozen during GAR training (used as feature extractor).
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models


# Places365 VGG-16 pretrained weight download URL
PLACES365_VGG16_URL = (
    "http://places2.csail.mit.edu/models_places365/vgg16_places365.pt"
)

# Scene attribute labels (102 classes) - from Places365
# Reference: https://github.com/CSAILVision/places365
SCENE_ATTRIBUTE_FILE = "data/scene_attributes.txt"

# Indoor/Outdoor label index from Places365 (2 classes)
# Index 0: indoor, Index 1: outdoor
# (derived from Places365 io_places365.txt)


class SceneDetector(nn.Module):
    """
    Scene Detector based on VGG-16 pretrained on Places365.

    Outputs:
        - inout_scores: (B, 2)   - indoor/outdoor logits
        - place_scores: (B, 365) - place category logits
        - attr_scores:  (B, 102) - scene attribute logits
        - scene_features: (B, 469) - concatenated scene prediction scores

    The conv5_3 feature map is also exposed for scene node embedding.
    """

    def __init__(self, pretrained_weights_path=None, freeze=True):
        super().__init__()

        # VGG-16 backbone (conv layers only, no original classifier)
        vgg = models.vgg16(weights=None)
        self.features = vgg.features          # conv1_1 ~ conv5_3: output 512×14×14 for 224 input
        self.avgpool = vgg.avgpool            # AdaptiveAvgPool2d(7,7)

        # Shared FC layers (equivalent to VGG-16 classifier layers 0~3)
        self.shared_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        # Three output heads
        self.fc_inout = nn.Linear(4096, 2)      # indoor / outdoor
        self.fc_place = nn.Linear(4096, 365)    # place categories
        self.fc_attr  = nn.Linear(4096, 102)    # scene attributes

        # Load pretrained Places365 weights if provided
        if pretrained_weights_path is not None:
            self._load_places365_weights(pretrained_weights_path)
        else:
            self._try_load_places365_auto()

        if freeze:
            self._freeze_backbone()

    def _load_places365_weights(self, path):
        """Load VGG-16 Places365 checkpoint."""
        print(f"[SceneDetector] Loading Places365 weights from {path}")
        checkpoint = torch.load(path, map_location="cpu")

        # Places365 official checkpoint stores weights under 'state_dict'
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Map Places365 keys → our model keys
        new_state = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if k.startswith("features."):
                new_state[k] = v
            elif k == "classifier.0.weight":
                new_state["shared_fc.0.weight"] = v
            elif k == "classifier.0.bias":
                new_state["shared_fc.0.bias"] = v
            elif k == "classifier.3.weight":
                new_state["shared_fc.3.weight"] = v
            elif k == "classifier.3.bias":
                new_state["shared_fc.3.bias"] = v
            elif k == "classifier.6.weight":
                # Places365 original has 365-class output → fc_place
                new_state["fc_place.weight"] = v
            elif k == "classifier.6.bias":
                new_state["fc_place.bias"] = v

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        print(f"[SceneDetector] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def _try_load_places365_auto(self):
        """Auto-download Places365 VGG-16 weights."""
        cache_path = os.path.expanduser("~/.cache/places365/vgg16_places365.pt")
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"[SceneDetector] Downloading Places365 weights → {cache_path}")
            try:
                import urllib.request
                urllib.request.urlretrieve(PLACES365_VGG16_URL, cache_path)
                print("[SceneDetector] Download complete.")
            except Exception as e:
                print(f"[SceneDetector] WARNING: Could not download weights: {e}")
                print("[SceneDetector] Continuing with random-initialized fc_inout and fc_attr heads.")
                return
        self._load_places365_weights(cache_path)

    def _freeze_backbone(self):
        """Freeze all parameters (scene detector is not trained during GAR training)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_conv5_feature(self, x):
        """Return conv5_3 feature map: (B, 512, H, W)."""
        return self.features(x)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - input images, normalized for VGG

        Returns:
            dict with keys: inout_scores, place_scores, attr_scores, scene_features
            conv5_feat: (B, 512, H//32, W//32) - raw conv5_3 features
        """
        conv5_feat = self.features(x)          # (B, 512, H/32, W/32)
        pooled = self.avgpool(conv5_feat)       # (B, 512, 7, 7)
        flat = pooled.view(pooled.size(0), -1)  # (B, 25088)
        shared = self.shared_fc(flat)           # (B, 4096)

        inout_scores = self.fc_inout(shared)    # (B, 2)
        place_scores  = self.fc_place(shared)   # (B, 365)
        attr_scores   = self.fc_attr(shared)    # (B, 102)

        # Concatenate all scene scores: (B, 469)
        scene_features = torch.cat([inout_scores, place_scores, attr_scores], dim=1)

        return {
            "inout_scores":   inout_scores,
            "place_scores":   place_scores,
            "attr_scores":    attr_scores,
            "scene_features": scene_features,
        }, conv5_feat


class SceneNodeEmbedding(nn.Module):
    """
    Projects the conv5_3 feature map into scene node vectors.

    The scene detector outputs 469 scene label probabilities.
    For each selected scene node (S = 2 + 2K nodes), we embed the
    corresponding label into a feature vector of dimension F (same as
    instance nodes).

    Implementation:
      - conv5_3 feature → Global Average Pool → FC → 4096-dim embedding
      - This embedding is then used as S scene node vectors in GCR
    """

    def __init__(self, in_channels=512, feature_dim=4096, num_scene_labels=469):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embed_fc = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.ReLU(inplace=True),
        )
        # Per-label embedding: each of 469 scene labels gets a feature_dim vector
        self.label_embedding = nn.Embedding(num_scene_labels, feature_dim)

    def forward(self, conv5_feat, selected_indices):
        """
        Args:
            conv5_feat: (B, 512, H, W) - conv5_3 feature map
            selected_indices: (B, S) - indices of selected scene nodes

        Returns:
            scene_nodes: (B, S, feature_dim)
        """
        # Global context from conv5 feature
        gap_feat = self.gap(conv5_feat).view(conv5_feat.size(0), -1)  # (B, 512)
        context = self.embed_fc(gap_feat)  # (B, 4096)

        # Label embeddings for selected scene nodes
        # selected_indices: (B, S)
        label_embs = self.label_embedding(selected_indices)  # (B, S, 4096)

        # Modulate label embeddings with global context
        # context: (B, 4096) → (B, 1, 4096)
        scene_nodes = label_embs + context.unsqueeze(1)  # (B, S, 4096)
        return scene_nodes
