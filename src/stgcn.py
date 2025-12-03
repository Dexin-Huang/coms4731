"""
Spatio-Temporal Graph Convolutional Network (ST-GCN) for Skeleton-Based Action Recognition.

This implements ST-GCN for basketball free throw prediction using pose skeleton sequences.
Based on: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"
(Yan et al., AAAI 2018)

The skeleton is represented as a graph where:
- Nodes = body joints
- Edges = bones (natural body connections)
- Temporal edges connect same joints across frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# Skeleton graph definition for MediaPipe (33 landmarks) - simplified to key joints
MEDIAPIPE_SKELETON = {
    'num_joints': 17,  # Using 17 key joints
    'joint_names': [
        'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle', 'left_ear', 'right_ear',
        'left_eye', 'right_eye'
    ],
    # MediaPipe indices for these joints
    'mediapipe_indices': [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 7, 8, 2, 5],
    # Edge connections (bone structure)
    'edges': [
        (0, 15), (0, 16),  # nose to eyes
        (15, 13), (16, 14),  # eyes to ears
        (0, 1), (0, 2),  # nose to shoulders
        (1, 2),  # shoulder to shoulder
        (1, 3), (3, 5),  # left arm
        (2, 4), (4, 6),  # right arm
        (1, 7), (2, 8),  # shoulders to hips
        (7, 8),  # hip to hip
        (7, 9), (9, 11),  # left leg
        (8, 10), (10, 12),  # right leg
    ],
    'center': 0  # Center joint (nose)
}

# SAM3D skeleton (22 joints)
SAM3D_SKELETON = {
    'num_joints': 22,
    'edges': [
        (0, 1), (0, 2), (0, 3),  # pelvis connections
        (1, 4), (4, 7), (7, 10),  # left leg
        (2, 5), (5, 8), (8, 11),  # right leg
        (3, 6), (6, 9), (9, 12),  # spine
        (12, 13), (12, 14), (12, 15),  # neck/head
        (13, 16), (16, 18), (18, 20),  # left arm
        (14, 17), (17, 19), (19, 21),  # right arm
    ],
    'center': 0  # pelvis
}


def build_adjacency_matrix(num_joints: int, edges: list, self_loops: bool = True) -> np.ndarray:
    """
    Build adjacency matrix for skeleton graph.

    Args:
        num_joints: Number of joints in skeleton
        edges: List of (i, j) tuples defining bone connections
        self_loops: Whether to add self-connections

    Returns:
        Adjacency matrix of shape (num_joints, num_joints)
    """
    A = np.zeros((num_joints, num_joints))

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Undirected graph

    if self_loops:
        A += np.eye(num_joints)

    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: D^(-1/2) A D^(-1/2)
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.

    Applies graph convolution on skeleton joints at each time step.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        adaptive: bool = True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Register adjacency matrix as buffer (not a parameter)
        self.register_buffer('A', A)

        # Learnable transformation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Adaptive adjacency (learnable graph structure)
        if adaptive:
            self.adaptive_A = nn.Parameter(torch.zeros_like(A))
        else:
            self.adaptive_A = None

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, T, V)
               N = batch size, C = channels, T = frames, V = joints

        Returns:
            Output tensor of shape (N, C', T, V)
        """
        N, C, T, V = x.shape

        # Get effective adjacency matrix
        A = self.A
        if self.adaptive_A is not None:
            A = A + self.adaptive_A

        # Apply graph convolution
        # Reshape for matrix multiplication: (N, C, T, V) -> (N*C*T, V)
        x_reshape = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)

        # Graph convolution: X' = A * X * W
        # (N*T, V, V) @ (N*T, V, C) -> (N*T, V, C)
        x_graph = torch.matmul(A.unsqueeze(0), x_reshape)

        # Reshape back: (N*T, V, C) -> (N, C, T, V)
        x_graph = x_graph.view(N, T, V, C).permute(0, 3, 1, 2).contiguous()

        # Channel transformation
        x_out = self.conv(x_graph)
        x_out = self.bn(x_out)

        return x_out


class TemporalConv(nn.Module):
    """
    Temporal Convolution layer.

    Captures temporal dynamics across frames.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1
    ):
        super().__init__()

        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, T, V)

        Returns:
            Output tensor of shape (N, C', T', V)
        """
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolution Block.

    Combines spatial graph convolution and temporal convolution
    with residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True
    ):
        super().__init__()

        self.spatial_conv = SpatialGraphConv(in_channels, out_channels, A, adaptive)
        self.temporal_conv = TemporalConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (N, C, T, V)

        Returns:
            Output tensor of shape (N, C', T', V)
        """
        res = self.residual(x)
        x = self.spatial_conv(x)
        x = self.relu(x)
        x = self.temporal_conv(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for skeleton-based action recognition.

    Architecture:
    - Input: Skeleton sequences (N, C, T, V)
    - Multiple ST-GCN blocks
    - Global average pooling
    - Classification head
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,  # (x, y, z) or (x, y, confidence)
        num_joints: int = 17,
        edges: list = None,
        hidden_channels: list = [64, 64, 64, 128, 128, 128, 256, 256, 256],
        dropout: float = 0.5,
        adaptive: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_joints = num_joints

        # Build adjacency matrix
        if edges is None:
            edges = MEDIAPIPE_SKELETON['edges']

        A = build_adjacency_matrix(num_joints, edges)
        A = normalize_adjacency(A)
        A = torch.tensor(A, dtype=torch.float32)

        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN layers
        self.layers = nn.ModuleList()

        channels = [in_channels] + hidden_channels
        for i in range(len(hidden_channels)):
            stride = 2 if i in [3, 6] else 1  # Downsample at layers 4 and 7
            self.layers.append(
                STGCNBlock(
                    channels[i],
                    channels[i + 1],
                    A,
                    stride=stride,
                    adaptive=adaptive
                )
            )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input skeleton sequence of shape (N, C, T, V)
               N = batch, C = channels (xyz), T = frames, V = joints

        Returns:
            Logits of shape (N, num_classes)
        """
        N, C, T, V = x.shape

        # Data normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # ST-GCN blocks
        for layer in self.layers:
            x = layer(x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(N, -1)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.

        Args:
            x: Input skeleton sequence

        Returns:
            Feature tensor of shape (N, hidden_channels[-1])
        """
        N, C, T, V = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(N, -1)

        return x


class LightweightSTGCN(nn.Module):
    """
    Lightweight ST-GCN for faster inference and training.

    Fewer layers and channels for smaller datasets.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        num_joints: int = 17,
        edges: list = None,
        hidden_channels: list = [32, 64, 128],
        dropout: float = 0.5
    ):
        super().__init__()

        if edges is None:
            edges = MEDIAPIPE_SKELETON['edges']

        A = build_adjacency_matrix(num_joints, edges)
        A = normalize_adjacency(A)
        A = torch.tensor(A, dtype=torch.float32)

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.layers = nn.ModuleList()
        channels = [in_channels] + hidden_channels

        for i in range(len(hidden_channels)):
            stride = 2 if i == len(hidden_channels) - 1 else 1
            self.layers.append(
                STGCNBlock(channels[i], channels[i + 1], A, stride=stride)
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(N, -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(N, -1)

        return x


def get_stgcn_model_info(model: nn.Module) -> dict:
    """Get model parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_millions': total_params / 1e6
    }


if __name__ == '__main__':
    print("Testing ST-GCN implementation...")

    # Test full ST-GCN
    model = STGCN(num_classes=2, in_channels=3, num_joints=17)
    info = get_stgcn_model_info(model)
    print(f"\nST-GCN Model:")
    print(f"  Parameters: {info['total_params_millions']:.2f}M")

    # Test input: (batch, channels, frames, joints)
    dummy_input = torch.randn(4, 3, 16, 17)
    print(f"  Input shape: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")

    features = model.extract_features(dummy_input)
    print(f"  Feature shape: {features.shape}")

    # Test lightweight version
    model_light = LightweightSTGCN(num_classes=2)
    info_light = get_stgcn_model_info(model_light)
    print(f"\nLightweight ST-GCN:")
    print(f"  Parameters: {info_light['total_params_millions']:.2f}M")

    output_light = model_light(dummy_input)
    print(f"  Output shape: {output_light.shape}")

    print("\nAll tests passed!")
