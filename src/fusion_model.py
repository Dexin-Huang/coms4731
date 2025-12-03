"""
Multi-Modal Fusion Model for Basketball Free Throw Prediction.

Combines three modalities:
1. Video Swin Transformer - Appearance and global motion
2. ST-GCN - Skeleton-based body mechanics
3. Ball Trajectory - Physics of the shot

This multi-modal approach aims to beat single-modality SOTA by
leveraging complementary information from each stream.

Architecture:
- Each modality encoder extracts features independently
- Cross-attention allows modalities to attend to each other
- Fusion layer combines all features
- Classification head predicts make/miss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal
import numpy as np


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.

    Allows one modality to attend to another, learning
    which parts of one modality are relevant given the other.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Project query and key to same dimension
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.output_proj = nn.Linear(hidden_dim, query_dim)
        self.layer_norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor of shape (N, query_dim)
            key_value: Key/value tensor of shape (N, key_dim)

        Returns:
            Attended features of shape (N, query_dim)
        """
        # Add sequence dimension for attention
        query = query.unsqueeze(1)  # (N, 1, query_dim)
        key_value = key_value.unsqueeze(1)  # (N, 1, key_dim)

        # Project
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)

        # Attention
        attended, _ = self.attention(q, k, v)

        # Project back and residual connection
        output = self.output_proj(attended.squeeze(1))
        output = self.layer_norm(output + query.squeeze(1))

        return output


class ModalityEncoder(nn.Module):
    """
    Generic modality encoder with projection head.

    Wraps any feature extractor and projects to common dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and project to common space."""
        features = self.backbone(x)
        return self.projection(features)


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining multiple modalities.

    Learns to weight each modality based on content.
    """

    def __init__(self, input_dims: list, output_dim: int = 256):
        super().__init__()

        self.num_modalities = len(input_dims)
        total_dim = sum(input_dims)

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(total_dim, self.num_modalities * 64),
            nn.ReLU(),
            nn.Linear(self.num_modalities * 64, self.num_modalities),
            nn.Softmax(dim=-1)
        )

        # Projection for each modality
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of tensors, each (N, dim_i)

        Returns:
            Fused features of shape (N, output_dim)
        """
        # Concatenate for gating
        concat = torch.cat(features, dim=-1)

        # Compute gates
        gates = self.gate(concat)  # (N, num_modalities)

        # Project each modality
        projected = [proj(f) for proj, f in zip(self.projections, features)]

        # Weighted sum
        stacked = torch.stack(projected, dim=1)  # (N, num_modalities, output_dim)
        gates = gates.unsqueeze(-1)  # (N, num_modalities, 1)

        fused = (stacked * gates).sum(dim=1)  # (N, output_dim)

        return fused


class MultiModalFreeThrowClassifier(nn.Module):
    """
    Multi-Modal Fusion Model for Free Throw Prediction.

    Combines:
    - Video features (from Video Swin Transformer)
    - Skeleton features (from ST-GCN)
    - Trajectory features (from ball tracker)

    With cross-attention fusion.
    """

    def __init__(
        self,
        video_dim: int = 768,      # Video Swin-T output
        skeleton_dim: int = 256,    # ST-GCN output
        trajectory_dim: int = 128,  # Trajectory encoder output
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention

        # Project each modality to fusion dimension
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

        self.skeleton_proj = nn.Sequential(
            nn.Linear(skeleton_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

        self.trajectory_proj = nn.Sequential(
            nn.Linear(trajectory_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

        # Cross-attention modules
        if use_cross_attention:
            # Video attends to skeleton
            self.video_skeleton_attn = CrossModalAttention(
                fusion_dim, fusion_dim, fusion_dim
            )
            # Video attends to trajectory
            self.video_trajectory_attn = CrossModalAttention(
                fusion_dim, fusion_dim, fusion_dim
            )
            # Skeleton attends to trajectory
            self.skeleton_trajectory_attn = CrossModalAttention(
                fusion_dim, fusion_dim, fusion_dim
            )

        # Gated fusion
        self.fusion = GatedFusion(
            [fusion_dim, fusion_dim, fusion_dim],
            output_dim=fusion_dim
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, num_classes)
        )

    def forward(
        self,
        video_features: torch.Tensor,
        skeleton_features: torch.Tensor,
        trajectory_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_features: (N, video_dim)
            skeleton_features: (N, skeleton_dim)
            trajectory_features: (N, trajectory_dim)

        Returns:
            Logits of shape (N, num_classes)
        """
        # Project to common dimension
        v = self.video_proj(video_features)
        s = self.skeleton_proj(skeleton_features)
        t = self.trajectory_proj(trajectory_features)

        # Cross-modal attention
        if self.use_cross_attention:
            v = v + self.video_skeleton_attn(v, s) + self.video_trajectory_attn(v, t)
            s = s + self.skeleton_trajectory_attn(s, t)

        # Gated fusion
        fused = self.fusion([v, s, t])

        # Classify
        logits = self.classifier(fused)

        return logits

    def get_attention_weights(
        self,
        video_features: torch.Tensor,
        skeleton_features: torch.Tensor,
        trajectory_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get attention weights for visualization."""
        v = self.video_proj(video_features)
        s = self.skeleton_proj(skeleton_features)
        t = self.trajectory_proj(trajectory_features)

        concat = torch.cat([v, s, t], dim=-1)

        # This returns the gate weights
        gates = self.fusion.gate(concat)

        return {
            'video_weight': gates[:, 0],
            'skeleton_weight': gates[:, 1],
            'trajectory_weight': gates[:, 2]
        }


class EndToEndMultiModalModel(nn.Module):
    """
    End-to-End Multi-Modal Model with all encoders.

    Combines:
    - Video Swin Transformer for video encoding
    - ST-GCN for skeleton encoding
    - Trajectory encoder for ball trajectory

    Can be trained end-to-end or with frozen encoders.
    """

    def __init__(
        self,
        # Video encoder config
        video_model_size: Literal['tiny', 'small', 'base'] = 'tiny',
        video_pretrained: bool = True,
        # Skeleton encoder config
        num_joints: int = 17,
        skeleton_in_channels: int = 3,
        # Trajectory encoder config
        trajectory_input_dim: int = 20,
        # Fusion config
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        freeze_video: bool = False,
        freeze_skeleton: bool = False
    ):
        super().__init__()

        # Import models
        from .model import create_video_swin
        from .stgcn import LightweightSTGCN
        from .ball_tracker import TrajectoryEncoder

        # Video encoder
        self.video_encoder = create_video_swin(
            num_classes=1000,  # Keep original head temporarily
            pretrained=video_pretrained,
            model_size=video_model_size
        )
        # Remove classification head to get features
        video_dim = self.video_encoder.head[1].in_features
        self.video_encoder.head = nn.Identity()

        if freeze_video:
            for param in self.video_encoder.parameters():
                param.requires_grad = False

        # Skeleton encoder
        self.skeleton_encoder = LightweightSTGCN(
            num_classes=num_classes,  # Will extract features
            in_channels=skeleton_in_channels,
            num_joints=num_joints
        )
        # Get feature dim before classifier
        skeleton_dim = self.skeleton_encoder.fc.in_features
        self.skeleton_encoder.fc = nn.Identity()

        if freeze_skeleton:
            for param in self.skeleton_encoder.parameters():
                param.requires_grad = False

        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            input_dim=trajectory_input_dim,
            output_dim=128
        )
        trajectory_dim = 128

        # Fusion model
        self.fusion = MultiModalFreeThrowClassifier(
            video_dim=video_dim,
            skeleton_dim=skeleton_dim,
            trajectory_dim=trajectory_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        video: torch.Tensor,           # (N, C, T, H, W)
        skeleton: torch.Tensor,         # (N, C, T, V)
        trajectory_features: torch.Tensor  # (N, traj_dim)
    ) -> torch.Tensor:
        """
        Forward pass through all encoders and fusion.

        Returns:
            Logits of shape (N, num_classes)
        """
        # Encode each modality
        video_feat = self.video_encoder(video)
        skeleton_feat = self.skeleton_encoder.extract_features(skeleton)
        traj_feat = self.trajectory_encoder(trajectory_features)

        # Fuse and classify
        logits = self.fusion(video_feat, skeleton_feat, traj_feat)

        return logits

    def freeze_encoders(self):
        """Freeze all encoders, only train fusion."""
        for param in self.video_encoder.parameters():
            param.requires_grad = False
        for param in self.skeleton_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class TwoStreamFusion(nn.Module):
    """
    Simplified two-stream fusion (Video + Skeleton only).

    For when trajectory data is not available.
    """

    def __init__(
        self,
        video_dim: int = 768,
        skeleton_dim: int = 256,
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

        self.skeleton_proj = nn.Sequential(
            nn.Linear(skeleton_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim)
        )

        # Simple concatenation + MLP fusion
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(
        self,
        video_features: torch.Tensor,
        skeleton_features: torch.Tensor
    ) -> torch.Tensor:
        v = self.video_proj(video_features)
        s = self.skeleton_proj(skeleton_features)

        concat = torch.cat([v, s], dim=-1)
        return self.fusion(concat)


class LateFusion(nn.Module):
    """
    Late fusion baseline - average predictions from each modality.
    """

    def __init__(
        self,
        video_classifier: nn.Module,
        skeleton_classifier: nn.Module,
        trajectory_classifier: Optional[nn.Module] = None,
        weights: Tuple[float, ...] = (0.4, 0.4, 0.2)
    ):
        super().__init__()

        self.video_classifier = video_classifier
        self.skeleton_classifier = skeleton_classifier
        self.trajectory_classifier = trajectory_classifier
        self.weights = weights

    def forward(
        self,
        video: torch.Tensor,
        skeleton: torch.Tensor,
        trajectory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns averaged predictions.
        """
        video_logits = self.video_classifier(video)
        skeleton_logits = self.skeleton_classifier(skeleton)

        if trajectory is not None and self.trajectory_classifier is not None:
            traj_logits = self.trajectory_classifier(trajectory)
            return (
                self.weights[0] * video_logits +
                self.weights[1] * skeleton_logits +
                self.weights[2] * traj_logits
            )
        else:
            w_v = self.weights[0] / (self.weights[0] + self.weights[1])
            w_s = self.weights[1] / (self.weights[0] + self.weights[1])
            return w_v * video_logits + w_s * skeleton_logits


def get_fusion_model_info(model: nn.Module) -> dict:
    """Get model parameter information."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total,
        'trainable_params': trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6
    }


if __name__ == '__main__':
    print("Testing Multi-Modal Fusion Model...")

    # Test fusion model (without actual encoders)
    fusion = MultiModalFreeThrowClassifier(
        video_dim=768,
        skeleton_dim=128,
        trajectory_dim=128,
        fusion_dim=256,
        num_classes=2
    )

    info = get_fusion_model_info(fusion)
    print(f"\nFusion model parameters: {info['total_millions']:.2f}M")

    # Test forward pass
    video_feat = torch.randn(4, 768)
    skeleton_feat = torch.randn(4, 128)
    traj_feat = torch.randn(4, 128)

    logits = fusion(video_feat, skeleton_feat, traj_feat)
    print(f"Input shapes: video={video_feat.shape}, skeleton={skeleton_feat.shape}, trajectory={traj_feat.shape}")
    print(f"Output shape: {logits.shape}")

    # Get attention weights
    weights = fusion.get_attention_weights(video_feat, skeleton_feat, traj_feat)
    print(f"\nGate weights (sample):")
    print(f"  Video: {weights['video_weight'][0]:.3f}")
    print(f"  Skeleton: {weights['skeleton_weight'][0]:.3f}")
    print(f"  Trajectory: {weights['trajectory_weight'][0]:.3f}")

    # Test two-stream fusion
    two_stream = TwoStreamFusion(video_dim=768, skeleton_dim=128)
    two_stream_out = two_stream(video_feat, skeleton_feat)
    print(f"\nTwo-stream output: {two_stream_out.shape}")

    print("\nAll tests passed!")
