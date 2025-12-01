"""
Training script for Basketball Free Throw Prediction.

Uses PyTorch Lightning for training with two-stage fine-tuning:
1. Stage 1: Train classification head only (frozen backbone)
2. Stage 2: Fine-tune all layers with discriminative learning rates

Supports:
- Class weights for imbalanced data
- Focal Loss for hard example mining
- Label smoothing
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from typing import Optional, Dict, Any
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from dataset import get_dataloaders, prepare_data_splits
from model import FreeThrowClassifier, get_model_info


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Args:
        alpha: Weighting factor for each class. Can be a scalar (for binary) or tensor
        gamma: Focusing parameter. Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)

        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            # Create smoothed one-hot targets
            with torch.no_grad():
                targets_one_hot = torch.zeros_like(inputs)
                targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                                  self.label_smoothing / num_classes
        else:
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Compute cross entropy
        ce = -targets_one_hot * torch.log(p + 1e-8)

        # Compute focal weight: (1 - p_t)^gamma
        p_t = (targets_one_hot * p).sum(dim=1, keepdim=True)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets].unsqueeze(1)
            focal_weight = focal_weight * alpha_t

        # Compute focal loss
        focal_loss = focal_weight * ce
        focal_loss = focal_loss.sum(dim=1)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FreeThrowLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for free throw prediction.

    Handles training, validation, and testing with metrics logging.
    """

    def __init__(
        self,
        model_type: str = 'swin_tiny',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        backbone_lr: Optional[float] = None,
        weight_decay: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        freeze_backbone: bool = False,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 2,
        total_epochs: int = 30,
        loss_type: str = 'cross_entropy',
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = FreeThrowClassifier(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )

        # Freeze backbone if requested (Stage 1)
        if freeze_backbone:
            self.model.freeze_backbone()

        # Loss function selection
        if loss_type == 'focal':
            print(f"Using Focal Loss (gamma={focal_gamma})")
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            # Default: Cross Entropy Loss with optional class weights and label smoothing
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.backbone_lr = backbone_lr or learning_rate / 10
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        self.training_step_outputs.append({'loss': loss, 'acc': acc})

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'preds': preds,
            'labels': y,
            'probs': probs[:, 1]  # Probability of class 1 (make)
        })

        return loss

    def on_validation_epoch_end(self):
        # Aggregate predictions
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()

        # Calculate metrics
        accuracy = (all_preds == all_labels).float().mean()

        # F1 Score (binary)
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)

        # Clear outputs
        self.validation_step_outputs.clear()

    def on_training_epoch_end(self):
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        # Use discriminative learning rates if backbone is not frozen
        if self.hparams.freeze_backbone:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            param_groups = self.model.get_param_groups(
                backbone_lr=self.backbone_lr,
                head_lr=self.learning_rate
            )
            # Add weight decay to param groups
            for pg in param_groups:
                pg['weight_decay'] = self.weight_decay

            optimizer = torch.optim.AdamW(param_groups)

        # Learning rate scheduler
        if self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.total_epochs - self.warmup_epochs,
                eta_min=1e-7
            )
        elif self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return optimizer

        # Warmup scheduler
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.warmup_epochs]
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config_path: str = 'configs/config.yaml'):
    """
    Main training function with two-stage fine-tuning.

    Stage 1: Train classification head only (frozen backbone)
    Stage 2: Fine-tune all layers with discriminative learning rates
    """
    # Load config
    config = load_config(config_path)

    # Set seed
    pl.seed_everything(config.get('seed', 42))

    # Prepare data splits if they don't exist
    splits_dir = Path(config['data']['splits_dir'])
    if not (splits_dir / 'train.csv').exists():
        print("Preparing data splits...")
        prepare_data_splits(
            data_dir=config['data']['data_dir'],
            output_dir=str(splits_dir),
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            random_state=config.get('seed', 42)
        )

    # Create dataloaders with optional balancing
    print("Creating dataloaders...")
    balance_strategy = config['data'].get('balance_strategy', 'none')
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config['data']['data_dir'],
        splits_dir=str(splits_dir),
        batch_size=config['training']['batch_size'],
        n_frames=config['data']['n_frames'],
        frame_size=tuple(config['data']['frame_size']),
        num_workers=config['training']['num_workers'],
        balance_strategy=balance_strategy
    )

    # Compute class weights if requested
    class_weights = None
    if config['training'].get('use_class_weights', False):
        import pandas as pd
        train_df = pd.read_csv(splits_dir / 'train.csv')
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=train_df['label'].values
        )
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Class weights: {class_weights}")

    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name='free_throw_prediction'
    )

    # ========================
    # STAGE 1: Train head only
    # ========================
    print("\n" + "="*50)
    print("STAGE 1: Training classification head (frozen backbone)")
    print("="*50)

    stage1_config = config['training']['stage1']
    loss_type = config['training'].get('loss_type', 'cross_entropy')
    focal_gamma = config['training'].get('focal_gamma', 2.0)

    model_stage1 = FreeThrowLightningModule(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        learning_rate=stage1_config['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        class_weights=class_weights,
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        freeze_backbone=True,
        scheduler_type='none',
        total_epochs=stage1_config['epochs'],
        loss_type=loss_type,
        focal_gamma=focal_gamma
    )

    # Print model info
    info = get_model_info(model_stage1.model)
    print(f"Model: {config['model']['type']}")
    print(f"Total parameters: {info['total_params_millions']:.2f}M")
    print(f"Trainable parameters: {info['trainable_params_millions']:.2f}M")

    # Stage 1 callbacks
    checkpoint_callback_s1 = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='stage1-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    trainer_stage1 = pl.Trainer(
        max_epochs=stage1_config['epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        logger=logger,
        callbacks=[checkpoint_callback_s1, LearningRateMonitor()],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        enable_progress_bar=True
    )

    trainer_stage1.fit(model_stage1, train_loader, val_loader)

    # ========================
    # STAGE 2: Fine-tune all
    # ========================
    print("\n" + "="*50)
    print("STAGE 2: Fine-tuning all layers")
    print("="*50)

    stage2_config = config['training']['stage2']

    # Create new model and load Stage 1 weights
    model_stage2 = FreeThrowLightningModule(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        pretrained=False,  # Will load from checkpoint
        dropout=config['model']['dropout'],
        learning_rate=stage2_config['head_lr'],
        backbone_lr=stage2_config['backbone_lr'],
        weight_decay=config['training']['weight_decay'],
        class_weights=class_weights,
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        freeze_backbone=False,
        scheduler_type=config['training']['scheduler'],
        warmup_epochs=config['training']['warmup_epochs'],
        total_epochs=stage2_config['epochs'],
        loss_type=loss_type,
        focal_gamma=focal_gamma
    )

    # Load Stage 1 weights
    checkpoint = torch.load(checkpoint_callback_s1.best_model_path)
    model_stage2.load_state_dict(checkpoint['state_dict'])
    model_stage2.model.unfreeze_backbone()

    info = get_model_info(model_stage2.model)
    print(f"Trainable parameters: {info['trainable_params_millions']:.2f}M")

    # Stage 2 callbacks
    checkpoint_callback_s2 = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='stage2-{epoch:02d}-{val_acc:.3f}',
        monitor='val_acc',
        mode='max',
        save_top_k=config['logging']['save_top_k']
    )

    early_stopping = EarlyStopping(
        monitor='val_acc',
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        mode='max'
    )

    trainer_stage2 = pl.Trainer(
        max_epochs=stage2_config['epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        logger=logger,
        callbacks=[checkpoint_callback_s2, early_stopping, LearningRateMonitor()],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        enable_progress_bar=True
    )

    trainer_stage2.fit(model_stage2, train_loader, val_loader)

    # ========================
    # Final Evaluation
    # ========================
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)

    # Load best checkpoint
    best_model = FreeThrowLightningModule.load_from_checkpoint(
        checkpoint_callback_s2.best_model_path
    )

    # Test
    trainer_stage2.test(best_model, test_loader)

    print(f"\nBest model saved to: {checkpoint_callback_s2.best_model_path}")
    print("Training complete!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Free Throw Prediction Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    train(args.config)
