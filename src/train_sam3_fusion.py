"""
Train SAM3 Trajectory + Video Swin Fusion Model.
Uses precise ball trajectory from SAM3 segmentation combined with Video Swin features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json


class SAM3FusionDataset(Dataset):
    """Dataset with SAM3 trajectory features and Video Swin features."""

    def __init__(
        self,
        video_features_path: str,
        sam3_trajectory_path: str,
        labels_path: str,
        traj_mean: np.ndarray = None,
        traj_std: np.ndarray = None
    ):
        self.video_features = np.load(video_features_path).astype(np.float32)
        self.trajectories = np.load(sam3_trajectory_path).astype(np.float32)
        self.labels = np.load(labels_path)

        # Normalize trajectories
        if traj_mean is None:
            self.traj_mean = self.trajectories.mean(axis=0, keepdims=True)
            self.traj_std = self.trajectories.std(axis=0, keepdims=True)
            self.traj_std[self.traj_std < 1e-6] = 1.0
        else:
            self.traj_mean = traj_mean
            self.traj_std = traj_std

        self.trajectories = (self.trajectories - self.traj_mean) / self.traj_std

        # Handle NaN/Inf
        self.trajectories = np.nan_to_num(self.trajectories, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.video_features[idx]),
            torch.from_numpy(self.trajectories[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class SAM3FusionClassifier(nn.Module):
    """Fusion model combining Video Swin features with SAM3 trajectory features."""

    def __init__(
        self,
        video_dim: int = 768,
        trajectory_dim: int = 20,
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        # Video encoder
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Trajectory encoder with physics-aware design
        self.traj_encoder = nn.Sequential(
            nn.Linear(trajectory_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, fusion_dim)
        )

        # Cross-attention for modality interaction
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout * 0.5,
            batch_first=True
        )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, video_feat, traj_feat):
        # Encode both modalities
        video_enc = self.video_encoder(video_feat)  # (B, fusion_dim)
        traj_enc = self.traj_encoder(traj_feat)     # (B, fusion_dim)

        # Cross-attention: trajectory attends to video
        # Add sequence dimension for attention
        video_seq = video_enc.unsqueeze(1)  # (B, 1, fusion_dim)
        traj_seq = traj_enc.unsqueeze(1)    # (B, 1, fusion_dim)

        attn_out, _ = self.cross_attn(traj_seq, video_seq, video_seq)
        attn_out = attn_out.squeeze(1)  # (B, fusion_dim)

        # Concatenate and fuse
        fused = torch.cat([video_enc, attn_out], dim=-1)  # (B, fusion_dim * 2)
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)
        return logits


def main():
    parser = argparse.ArgumentParser(description='Train SAM3 Fusion Model')
    parser.add_argument('--video_features_dir', type=str, default='data/stgcn_features')
    parser.add_argument('--sam3_features_dir', type=str, default='data/sam3_features')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/sam3_fusion')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 60)
    print("SAM3 TRAJECTORY + VIDEO SWIN FUSION TRAINING")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    video_dir = Path(args.video_features_dir)
    sam3_dir = Path(args.sam3_features_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SAM3FusionDataset(
        video_dir / 'train_video_features.npy',
        sam3_dir / 'train_sam3_trajectories.npy',
        sam3_dir / 'train_labels.npy'
    )

    val_dataset = SAM3FusionDataset(
        video_dir / 'val_video_features.npy',
        sam3_dir / 'val_sam3_trajectories.npy',
        sam3_dir / 'val_labels.npy',
        traj_mean=train_dataset.traj_mean,
        traj_std=train_dataset.traj_std
    )

    video_feat_dim = train_dataset.video_features.shape[1]
    trajectory_dim = train_dataset.trajectories.shape[1]

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Video feature dim: {video_feat_dim}")
    print(f"  SAM3 trajectory dim: {trajectory_dim}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    print("\nCreating SAM3 Fusion model...")
    model = SAM3FusionClassifier(
        video_dim=video_feat_dim,
        trajectory_dim=trajectory_dim,
        fusion_dim=256,
        num_classes=2,
        dropout=0.5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params / 1e6:.2f}M")

    # Loss with class weights
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    print(f"  Class distribution: {class_counts}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    print("\nTraining...")
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for video_feat, trajectories, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            video_feat = video_feat.to(device)
            trajectories = trajectories.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(video_feat, trajectories)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for video_feat, trajectories, labels in val_loader:
                video_feat = video_feat.to(device)
                trajectories = trajectories.to(device)
                labels = labels.to(device)

                outputs = model(video_feat, trajectories)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # F1 score
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        val_f1 = 2 * precision * recall / (precision + recall + 1e-8)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | "
              f"LR: {lr:.2e}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'traj_mean': train_dataset.traj_mean,
                'traj_std': train_dataset.traj_std,
            }, checkpoint_dir / 'best_sam3_fusion_model.pt')
            print(f"  -> Saved best model (val_acc: {val_acc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save history
    with open(checkpoint_dir / 'sam3_fusion_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("SAM3 FUSION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Best Val F1: {best_val_f1:.3f}")
    print(f"\nComparison:")
    print(f"  Majority Baseline:      75.3%")
    print(f"  Video Swin alone:       74.6%")
    print(f"  ST-GCN alone:           69.4%")
    print(f"  Full Fusion (V+S+T):    64.4%")
    print(f"  SAM3 FUSION (V+SAM3):   {best_val_acc*100:.1f}%")

    if best_val_acc > 0.753:
        print(f"\n  BEATS MAJORITY BASELINE by: +{(best_val_acc - 0.753)*100:.1f}%")
    elif best_val_acc > 0.746:
        print(f"\n  IMPROVEMENT over Video Swin: +{(best_val_acc - 0.746)*100:.1f}%")
    else:
        print(f"\n  Underperforms Video Swin by: {(0.746 - best_val_acc)*100:.1f}%")


if __name__ == '__main__':
    main()
