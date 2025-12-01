"""
Video Classification Models for Basketball Free Throw Prediction.

Provides Video Swin Transformer and alternative architectures with transfer learning.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


def create_video_swin(
    num_classes: int = 2,
    pretrained: bool = True,
    model_size: Literal['tiny', 'small', 'base'] = 'tiny',
    dropout: float = 0.5,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Create Video Swin Transformer model for binary classification.

    Args:
        num_classes: Number of output classes (2 for make/miss)
        pretrained: Whether to use pretrained weights from Kinetics-400
        model_size: Model size ('tiny', 'small', 'base')
        dropout: Dropout rate before classification head
        freeze_backbone: Whether to freeze backbone weights

    Returns:
        Video Swin Transformer model
    """
    import torchvision.models.video as video_models
    from torchvision.models.video import (
        Swin3D_T_Weights,
        Swin3D_S_Weights,
        Swin3D_B_Weights
    )

    # Select model and weights based on size
    if model_size == 'tiny':
        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        model = video_models.swin3d_t(weights=weights)
    elif model_size == 'small':
        weights = Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
        model = video_models.swin3d_s(weights=weights)
    elif model_size == 'base':
        weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1 if pretrained else None
        model = video_models.swin3d_b(weights=weights)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )

    # Ensure head is trainable even if backbone is frozen
    for param in model.head.parameters():
        param.requires_grad = True

    return model


def create_r3d(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Create R3D-18 model as a simpler baseline.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        freeze_backbone: Whether to freeze backbone

    Returns:
        R3D-18 model
    """
    import torchvision.models.video as video_models
    from torchvision.models.video import R3D_18_Weights

    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model = video_models.r3d_18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


class FreeThrowClassifier(nn.Module):
    """
    Wrapper class for video classification models with additional utilities.

    Provides methods for:
    - Two-stage fine-tuning (freeze/unfreeze backbone)
    - Feature extraction
    - Prediction with probabilities
    """

    def __init__(
        self,
        model_type: Literal['swin_tiny', 'swin_small', 'swin_base', 'r3d'] = 'swin_tiny',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()

        self.model_type = model_type
        self.num_classes = num_classes

        if model_type.startswith('swin'):
            size = model_type.split('_')[1]
            self.model = create_video_swin(
                num_classes=num_classes,
                pretrained=pretrained,
                model_size=size,
                dropout=dropout
            )
        elif model_type == 'r3d':
            self.model = create_r3d(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.

        Args:
            x: Input tensor

        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.

        Args:
            x: Input tensor

        Returns:
            Probabilities of shape (B, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def freeze_backbone(self):
        """Freeze backbone weights, keep head trainable."""
        if self.model_type.startswith('swin'):
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        elif self.model_type == 'r3d':
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all weights."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-4):
        """
        Get parameter groups with discriminative learning rates.

        Args:
            backbone_lr: Learning rate for backbone
            head_lr: Learning rate for classification head

        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = []
        head_params = []

        head_name = 'head' if self.model_type.startswith('swin') else 'fc'

        for name, param in self.model.named_parameters():
            if head_name in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ]


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information including parameter counts.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6
    }


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")

    # Test Video Swin Transformer
    model = FreeThrowClassifier(model_type='swin_tiny', pretrained=True)
    print(f"\nVideo Swin-T model created")
    info = get_model_info(model)
    print(f"  Total params: {info['total_params_millions']:.2f}M")
    print(f"  Trainable params: {info['trainable_params_millions']:.2f}M")

    # Test input
    dummy_input = torch.randn(2, 3, 16, 224, 224)  # (B, C, T, H, W)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test predictions
    probs = model.predict_proba(dummy_input)
    print(f"Probabilities: {probs}")

    # Test freeze/unfreeze
    print("\nTesting freeze/unfreeze...")
    model.freeze_backbone()
    info = get_model_info(model)
    print(f"  After freeze - Trainable: {info['trainable_params_millions']:.2f}M")

    model.unfreeze_backbone()
    info = get_model_info(model)
    print(f"  After unfreeze - Trainable: {info['trainable_params_millions']:.2f}M")

    print("\nAll tests passed!")
