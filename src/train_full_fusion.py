"""
Train Full Multi-Modal Fusion Model (Video + Skeleton + Trajectory).
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

from stgcn import LightweightSTGCN
from fusion_model import MultiModalFreeThrowClassifier


class FullFusionDataset(Dataset):
    """Dataset with pre-extracted video, skeleton, and trajectory features."""

    def __init__(
        self,
        video_features_path: str,
        skeleton_path: str,
        trajectory_path: str,
        labels_path: str,
        traj_mean: np.ndarray = None,
        traj_std: np.ndarray = None
    ):
        self.video_features = np.load(video_features_path).astype(np.float32)
        self.skeletons = np.load(skeleton_path).astype(np.float32)
        self.trajectories = np.load(trajectory_path).astype(np.float32)
        self.labels = np.load(labels_path)

        # Normalize skeletons
        self.skeletons = self._normalize_skeletons(self.skeletons)

        # Normalize trajectories
        if traj_mean is None:
            self.traj_mean = self.trajectories.mean(axis=0, keepdims=True)
            self.traj_std = self.trajectories.std(axis=0, keepdims=True)
            self.traj_std[self.traj_std < 1e-6] = 1.0
        else:
            self.traj_mean = traj_mean
            self.traj_std = traj_std

        self.trajectories = (self.trajectories - self.traj_mean) / self.traj_std

    def _normalize_skeletons(self, skeletons):
        """Center and scale skeletons."""
        center = (skeletons[:, :, :, 7:8] + skeletons[:, :, :, 8:9]) / 2
        skeletons = skeletons - center
        std = np.std(skeletons, axis=(1, 2, 3), keepdims=True)
        std[std < 1e-6] = 1.0
        return skeletons / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.video_features[idx]),
            torch.from_numpy(self.skeletons[idx]),
            torch.from_numpy(self.trajectories[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def main():
    parser = argparse.ArgumentParser(description='Train Full Fusion Model')
    parser.add_argument('--features_dir', type=str, default='data/stgcn_features')
    parser.add_argument('--skeleton_ckpt', type=str, default='checkpoints/stgcn/best_model.pt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/full_fusion')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 60)
    print("FULL MULTI-MODAL FUSION TRAINING")
    print("(Video + Skeleton + Trajectory)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features_dir = Path(args.features_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load ST-GCN model for skeleton feature extraction
    print("\nLoading ST-GCN model...")
    skeleton_model = LightweightSTGCN(
        num_classes=2,
        in_channels=3,
        num_joints=17,
        hidden_channels=[64, 128, 256],
        dropout=0.5
    )
    skeleton_ckpt = torch.load(args.skeleton_ckpt, map_location=device, weights_only=False)
    skeleton_model.load_state_dict(skeleton_ckpt['model_state_dict'])
    skeleton_model = skeleton_model.to(device)
    skeleton_model.eval()
    print(f"  Loaded from {args.skeleton_ckpt}")

    # Get skeleton feature dimension
    dummy_skeleton = torch.randn(1, 3, 16, 17).to(device)
    with torch.no_grad():
        skeleton_feat_dim = skeleton_model.extract_features(dummy_skeleton).shape[1]
    print(f"  Skeleton feature dim: {skeleton_feat_dim}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = FullFusionDataset(
        features_dir / 'train_video_features.npy',
        features_dir / 'train_skeletons.npy',
        features_dir / 'train_trajectories.npy',
        features_dir / 'train_labels.npy'
    )

    val_dataset = FullFusionDataset(
        features_dir / 'val_video_features.npy',
        features_dir / 'val_skeletons.npy',
        features_dir / 'val_trajectories.npy',
        features_dir / 'val_labels.npy',
        traj_mean=train_dataset.traj_mean,
        traj_std=train_dataset.traj_std
    )

    video_feat_dim = train_dataset.video_features.shape[1]
    trajectory_dim = train_dataset.trajectories.shape[1]

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Video feature dim: {video_feat_dim}")
    print(f"  Skeleton feature dim: {skeleton_feat_dim}")
    print(f"  Trajectory dim: {trajectory_dim}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create fusion model
    print("\nCreating fusion model...")
    fusion_model = MultiModalFreeThrowClassifier(
        video_dim=video_feat_dim,
        skeleton_dim=skeleton_feat_dim,
        trajectory_dim=trajectory_dim,
        fusion_dim=256,
        num_classes=2,
        dropout=0.5,
        use_cross_attention=True
    ).to(device)

    total_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"  Fusion model parameters: {total_params / 1e6:.2f}M")

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
        fusion_model.parameters(),
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
        fusion_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for video_feat, skeletons, trajectories, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            video_feat = video_feat.to(device)
            skeletons = skeletons.to(device)
            trajectories = trajectories.to(device)
            labels = labels.to(device)

            # Extract skeleton features
            with torch.no_grad():
                skeleton_feat = skeleton_model.extract_features(skeletons)

            optimizer.zero_grad()
            outputs = fusion_model(video_feat, skeleton_feat, trajectories)
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
        fusion_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for video_feat, skeletons, trajectories, labels in val_loader:
                video_feat = video_feat.to(device)
                skeletons = skeletons.to(device)
                trajectories = trajectories.to(device)
                labels = labels.to(device)

                skeleton_feat = skeleton_model.extract_features(skeletons)
                outputs = fusion_model(video_feat, skeleton_feat, trajectories)
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
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, checkpoint_dir / 'best_full_fusion_model.pt')
            print(f"  -> Saved best model (val_acc: {val_acc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save history
    with open(checkpoint_dir / 'full_fusion_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("FULL FUSION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Best Val F1: {best_val_f1:.3f}")
    print(f"\nComparison:")
    print(f"  Video Swin alone:       74.6%")
    print(f"  ST-GCN alone:           69.4%")
    print(f"  Skel+Traj fusion:       59.5%")
    print(f"  FULL FUSION (V+S+T):    {best_val_acc*100:.1f}%")

    if best_val_acc > 0.746:
        print(f"\n  IMPROVEMENT over Video Swin: +{(best_val_acc - 0.746)*100:.1f}%")
    else:
        print(f"\n  Underperforms Video Swin by: {(0.746 - best_val_acc)*100:.1f}%")


if __name__ == '__main__':
    main()
