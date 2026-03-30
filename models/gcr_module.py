"""
GCR (Graph Convolutional Reasoning) module for GAR.

Implements:
  1. Instance edge construction (Algorithm 1 from paper)
  2. Scene node selection and edge construction (Algorithm 2)
  3. Two-layer GCN on the heterogeneous graph
  4. Score fusion: Z = softmax(wb)/(softmax(wb)+softmax(wg)) * Yb
                       + softmax(wg)/(softmax(wb)+softmax(wg)) * Yg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCRModule(nn.Module):
    """
    Graph Convolutional Reasoning module.

    Args:
        num_classes: O (number of object classes, background excluded)
        feature_dim: F = 4096 (ROI feature / scene node dim)
        gcn_hidden: GCN first layer output dim (default 4096)
        gcn_out: GCN second layer output dim (default 512)
        K: top-K place/attribute scene nodes (default 3)
    """

    def __init__(self, num_classes, feature_dim=4096, gcn_hidden=4096,
                 gcn_out=512, K=3):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.K = K
        self.S = 2 + 2 * K  # total scene nodes per image

        # Two-layer GCN weights
        # W0: feature_dim → gcn_hidden
        self.W0 = nn.Linear(feature_dim, gcn_hidden, bias=False)
        # W1: gcn_hidden → gcn_out
        self.W1 = nn.Linear(gcn_hidden, gcn_out, bias=False)
        # Final projection: gcn_out → num_classes + 1 (including background)
        self.fc_out = nn.Linear(gcn_out, num_classes + 1)

        # Learnable fusion weights (scalar parameters)
        self.wb = nn.Parameter(torch.tensor(0.0))  # cursory score weight
        self.wg = nn.Parameter(torch.tensor(0.0))  # graph score weight

    def build_instance_edges(self, cursory_scores, obj_obj_matrix):
        """
        Algorithm 1: Construct instance-instance relation edges.

        Args:
            cursory_scores: (N, O+1) - cursory class scores from cRCN
                            (includes background at index 0)
            obj_obj_matrix: (O, O) - normalized object-object co-occurrence

        Returns:
            E_ins: (N, N) - normalized instance relation edge matrix
        """
        N = cursory_scores.size(0)
        device = cursory_scores.device

        # Get cursory class label for each instance (exclude background)
        # cursory_scores[:, 1:] are O foreground class scores
        scores_fg = cursory_scores[:, 1:]  # (N, O)
        cls_labels = torch.argmax(scores_fg, dim=1)  # (N,) each in [0, O-1]

        # Build N x N edge matrix from obj_obj_matrix lookup
        # E_ins[i,j] = obj_obj_matrix[cls_i, cls_j]
        E_ins = obj_obj_matrix[cls_labels][:, cls_labels]  # (N, N)

        # Add learnable self-loop: diagonal set to 1.0
        E_ins = E_ins + torch.eye(N, device=device)

        # Row-wise softmax normalization
        E_ins = F.softmax(E_ins, dim=1)  # (N, N)
        return E_ins

    def build_scene_edges(self, cursory_scores, scene_out, cooc_matrices):
        """
        Algorithm 2: Select scene nodes and build instance-scene edges.

        Args:
            cursory_scores: (N, O+1) - cursory class scores
            scene_out: dict with 'inout_scores'(B,2), 'place_scores'(B,365),
                       'attr_scores'(B,102) - scene detector outputs for this image
            cooc_matrices: dict of {obj_inout, obj_place, obj_attr} tensors

        Returns:
            E_scene: (N, S) - normalized instance-scene edge matrix
            selected_inout_idx: (2,) - always [0, 1] (indoor, outdoor)
            selected_place_idx: (K,) - top-K place category indices
            selected_attr_idx:  (K,) - top-K scene attribute indices
        """
        N = cursory_scores.size(0)
        device = cursory_scores.device

        scores_fg = cursory_scores[:, 1:]     # (N, O)
        cls_labels = torch.argmax(scores_fg, dim=1)  # (N,)

        # --- Select scene nodes ---
        # Always include both indoor & outdoor nodes
        inout_idx = torch.tensor([0, 1], device=device)  # (2,)

        # Top-K place categories by score
        place_scores = scene_out["place_scores"][0]  # (365,)
        _, place_idx = torch.topk(place_scores, self.K)  # (K,)

        # Top-K scene attributes by score
        attr_scores = scene_out["attr_scores"][0]  # (102,)
        _, attr_idx = torch.topk(attr_scores, self.K)  # (K,)

        # --- Build instance-scene edges ---
        obj_inout = cooc_matrices["obj_inout"].to(device)  # (O, 2)
        obj_place = cooc_matrices["obj_place"].to(device)  # (O, 365)
        obj_attr  = cooc_matrices["obj_attr"].to(device)   # (O, 102)

        # Indoor/Outdoor: (N, 2)
        # Softmax the full obj_inout row for each class, then select [in, out]
        obj_inout_norm = F.softmax(obj_inout, dim=1)  # (O, 2)
        E_inout = obj_inout_norm[cls_labels]  # (N, 2)

        # Place: (N, K)
        obj_place_norm = F.softmax(obj_place, dim=1)  # (O, 365)
        E_place = obj_place_norm[cls_labels][:, place_idx]  # (N, K)
        E_place = F.softmax(E_place, dim=1)

        # Attribute: (N, K)
        obj_attr_norm = F.softmax(obj_attr, dim=1)  # (O, 102)
        E_attr = obj_attr_norm[cls_labels][:, attr_idx]  # (N, K)
        E_attr = F.softmax(E_attr, dim=1)

        # Concatenate: (N, S=2+K+K)
        E_scene = torch.cat([E_inout, E_place, E_attr], dim=1)  # (N, S)

        return E_scene, inout_idx, place_idx, attr_idx

    def build_adj_matrix(self, E_ins, E_scene):
        """
        Build normalized adjacency matrix Ã for the full heterogeneous graph.

        Nodes: [N instance nodes | S scene nodes] = N+S total
        Ã is (N+S) x (N+S):
          - Top-left (N x N): instance-instance edges
          - Top-right (N x S): instance-scene edges
          - Bottom-left (S x N): scene-instance edges (transpose)
          - Bottom-right (S x S): identity (scene self-loops only)

        Args:
            E_ins:   (N, N) - instance-instance edges (already softmax-normalized)
            E_scene: (N, S) - instance-scene edges (already softmax-normalized)

        Returns:
            A_tilde: (N+S, N+S) - normalized adjacency matrix
        """
        N = E_ins.size(0)
        S = E_scene.size(1)
        device = E_ins.device

        A_tilde = torch.zeros(N + S, N + S, device=device)
        A_tilde[:N, :N] = E_ins
        A_tilde[:N, N:] = E_scene
        A_tilde[N:, :N] = E_scene.t()
        A_tilde[N:, N:] = torch.eye(S, device=device)  # scene self-loops

        return A_tilde

    def gcn_forward(self, X, A_tilde):
        """
        Two-layer GCN: Yg = Ã · ReLU(Ã · X · W0) · W1

        Args:
            X:       (N+S, F) - node feature matrix
            A_tilde: (N+S, N+S) - normalized adjacency

        Returns:
            out: (N+S, num_classes)
        """
        # Layer 1: Ã · X · W0 → ReLU
        h = A_tilde @ self.W0(X)     # (N+S, gcn_hidden)
        h = F.relu(h)
        # Layer 2: Ã · h · W1
        h = A_tilde @ self.W1(h)     # (N+S, gcn_out)
        out = self.fc_out(h)         # (N+S, num_classes)
        return out

    def fuse_scores(self, cursory_scores, graph_scores):
        """
        Equation (6): Z = softmax(wb)/(wb+wg) * Yb + softmax(wg)/(wb+wg) * Yg

        Args:
            cursory_scores: (N, O+1) - from cRCN (includes background)
            graph_scores:   (N, O)   - from GCN (instance nodes only)

        Returns:
            cogitative_scores: (N, O+1)
        """
        # Learnable weight normalization
        w = F.softmax(torch.stack([self.wb, self.wg]), dim=0)  # (2,)
        wb, wg = w[0], w[1]

        # Fuse all classes including background
        fused = wb * cursory_scores + wg * graph_scores

        return fused

    def forward(self, instance_features, cursory_scores, scene_out,
                scene_node_features, cooc_matrices):
        """
        Full GCR forward pass.

        Args:
            instance_features: (N, F) - ROI-pooled + FC features
            cursory_scores:    (N, O+1) - cursory classification scores
            scene_out:         dict from SceneDetector
            scene_node_features: (S, F) - scene node embeddings
            cooc_matrices:     dict of {obj_obj, obj_inout, obj_place, obj_attr}

        Returns:
            cogitative_scores: (N, O+1) - final detection scores
        """
        # --- Build edges ---
        obj_obj = cooc_matrices["obj_obj"].to(instance_features.device)
        # Normalize obj_obj row-wise before edge construction
        obj_obj_norm = F.softmax(obj_obj, dim=1)

        E_ins = self.build_instance_edges(cursory_scores, obj_obj_norm)
        E_scene, _, _, _ = self.build_scene_edges(
            cursory_scores, scene_out, cooc_matrices
        )

        A_tilde = self.build_adj_matrix(E_ins, E_scene)  # (N+S, N+S)

        # --- Build node feature matrix ---
        # X: [instance_features | scene_node_features] = (N+S, F)
        X = torch.cat([instance_features, scene_node_features], dim=0)  # (N+S, F)

        # --- GCN ---
        graph_out = self.gcn_forward(X, A_tilde)  # (N+S, num_classes)

        # Take only instance node outputs
        graph_scores = graph_out[:instance_features.size(0)]  # (N, num_classes)

        # --- Fuse with cursory scores ---
        cogitative_scores = self.fuse_scores(cursory_scores, graph_scores)

        return cogitative_scores
