"""
Train Multi-Modal Fusion Model.

Combines:
- Video Swin Transformer features
- ST-GCN skeleton features
- Ball trajectory features
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
import cv2

from stgcn import LightweightSTGCN
from fusion_model import MultiModalFreeThrowClassifier, TwoStreamFusion


class FusionDataset(Dataset):
    """Dataset that provides video, skeleton, and trajectory features."""

    def __init__(
        self,
        video_dir: str,
        skeleton_path: str,
        trajectory_path: str,
        labels_path: str,
        video_ids_path: str,
        n_frames: int = 16,
        img_size: int = 224
    ):
        self.video_dir = Path(video_dir)
        self.n_frames = n_frames
        self.img_size = img_size

        # Load pre-extracted features
        self.skeletons = np.load(skeleton_path).astype(np.float32)
        self.trajectories = np.load(trajectory_path).astype(np.float32)
        self.labels = np.load(labels_path)

        # Load video paths
        with open(video_ids_path, 'r') as f:
            self.video_paths = [line.strip() for line in f.readlines()]

        # Normalize skeletons
        self.skeletons = self._normalize_skeletons(self.skeletons)

        # Normalize trajectories
        self.trajectories = self._normalize_trajectories(self.trajectories)

    def _normalize_skeletons(self, skeletons):
        """Center and scale skeletons."""
        center = (skeletons[:, :, :, 7:8] + skeletons[:, :, :, 8:9]) / 2
        skeletons = skeletons - center
        std = np.std(skeletons, axis=(1, 2, 3), keepdims=True)
        std[std < 1e-6] = 1.0
        return skeletons / std

    def _normalize_trajectories(self, trajectories):
        """Normalize trajectory features."""
        mean = trajectories.mean(axis=0, keepdims=True)
        std = trajectories.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (trajectories - mean) / std

    def _load_video_frames(self, video_path: str):
        """Load and preprocess video frames."""
        full_path = self.video_dir / video_path

        cap = cv2.VideoCapture(str(full_path))
        if not cap.isOpened():
            # Return zeros if video can't be opened
            return torch.zeros(3, self.n_frames, self.img_size, self.img_size)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return torch.zeros(3, self.n_frames, self.img_size, self.img_size)

        indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame.astype(np.float32) / 255.0
                # Normalize with ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame = (frame - mean) / std
                frames.append(frame)
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3)))

        cap.release()

        # Stack and transpose: (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames, axis=0)
        video = video.transpose(3, 0, 1, 2)

        return torch.from_numpy(video.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load video frames
        video = self._load_video_frames(self.video_paths[idx])

        # Get pre-extracted features
        skeleton = torch.from_numpy(self.skeletons[idx])
        trajectory = torch.from_numpy(self.trajectories[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return video, skeleton, trajectory, label


class PreextractedFusionDataset(Dataset):
    """Dataset using pre-extracted features (faster training)."""

    def __init__(
        self,
        video_features_path: str,
        skeleton_path: str,
        trajectory_path: str,
        labels_path: str
    ):
        self.video_features = np.load(video_features_path).astype(np.float32)
        self.skeletons = np.load(skeleton_path).astype(np.float32)
        self.trajectories = np.load(trajectory_path).astype(np.float32)
        self.labels = np.load(labels_path)

        # Normalize
        self.skeletons = self._normalize_skeletons(self.skeletons)
        self.trajectories = self._normalize_trajectories(self.trajectories)

    def _normalize_skeletons(self, skeletons):
        center = (skeletons[:, :, :, 7:8] + skeletons[:, :, :, 8:9]) / 2
        skeletons = skeletons - center
        std = np.std(skeletons, axis=(1, 2, 3), keepdims=True)
        std[std < 1e-6] = 1.0
        return skeletons / std

    def _normalize_trajectories(self, trajectories):
        mean = trajectories.mean(axis=0, keepdims=True)
        std = trajectories.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (trajectories - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.video_features[idx]),
            torch.from_numpy(self.skeletons[idx]),
            torch.from_numpy(self.trajectories[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def extract_features(
    video_model,
    skeleton_model,
    dataset,
    device,
    batch_size=8
):
    """Extract features from pretrained models."""
    video_model.eval()
    skeleton_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    video_features = []
    skeleton_features = []
    trajectory_features = []
    labels = []

    with torch.no_grad():
        for video, skeleton, trajectory, label in tqdm(loader, desc="Extracting features"):
            video = video.to(device)
            skeleton = skeleton.to(device)

            # Extract video features
            v_feat = video_model(video)
            video_features.append(v_feat.cpu())

            # Extract skeleton features
            s_feat = skeleton_model.extract_features(skeleton)
            skeleton_features.append(s_feat.cpu())

            trajectory_features.append(trajectory)
            labels.append(label)

    return (
        torch.cat(video_features, dim=0),
        torch.cat(skeleton_features, dim=0),
        torch.cat(trajectory_features, dim=0),
        torch.cat(labels, dim=0)
    )


def train_epoch(model, loader, criterion, optimizer, device, skeleton_model=None):
    model.train()
    if skeleton_model:
        skeleton_model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        if len(batch) == 4:
            video_feat, skeleton, trajectory, labels = batch
            video_feat = video_feat.to(device)
            skeleton = skeleton.to(device)
            trajectory = trajectory.to(device)
            labels = labels.to(device)

            # Extract skeleton features if needed
            if skeleton_model:
                with torch.no_grad():
                    skeleton_feat = skeleton_model.extract_features(skeleton)
            else:
                skeleton_feat = skeleton
        else:
            raise ValueError("Unexpected batch format")

        optimizer.zero_grad()
        outputs = model(video_feat, skeleton_feat, trajectory)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def validate(model, loader, criterion, device, skeleton_model=None):
    model.eval()
    if skeleton_model:
        skeleton_model.eval()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if len(batch) == 4:
                video_feat, skeleton, trajectory, labels = batch
                video_feat = video_feat.to(device)
                skeleton = skeleton.to(device)
                trajectory = trajectory.to(device)
                labels = labels.to(device)

                if skeleton_model:
                    skeleton_feat = skeleton_model.extract_features(skeleton)
                else:
                    skeleton_feat = skeleton

            outputs = model(video_feat, skeleton_feat, trajectory)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return total_loss / total, correct / total, f1


def main():
    parser = argparse.ArgumentParser(description='Train Fusion Model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base data directory')
    parser.add_argument('--features_dir', type=str, default='data/stgcn_features',
                        help='Directory with extracted features')
    parser.add_argument('--video_ckpt', type=str,
                        default='checkpoints/stage2-epoch=20-val_acc=0.746.ckpt',
                        help='Video Swin checkpoint')
    parser.add_argument('--skeleton_ckpt', type=str,
                        default='checkpoints/stgcn/best_model.pt',
                        help='ST-GCN checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/fusion',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--extract_video_features', action='store_true',
                        help='Extract video features (slow, requires video files)')

    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-MODAL FUSION TRAINING")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features_dir = Path(args.features_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load ST-GCN model
    print("\nLoading ST-GCN model...")
    skeleton_model = LightweightSTGCN(
        num_classes=2,
        in_channels=3,
        num_joints=17,
        hidden_channels=[64, 128, 256],
        dropout=0.5
    )

    skeleton_ckpt = torch.load(args.skeleton_ckpt, map_location=device)
    skeleton_model.load_state_dict(skeleton_ckpt['model_state_dict'])
    skeleton_model = skeleton_model.to(device)
    skeleton_model.eval()
    print(f"  Loaded from {args.skeleton_ckpt}")
    print(f"  Val accuracy: {skeleton_ckpt.get('val_acc', 'N/A')}")

    # Get skeleton feature dimension
    dummy_skeleton = torch.randn(1, 3, 16, 17).to(device)
    with torch.no_grad():
        skeleton_feat_dim = skeleton_model.extract_features(dummy_skeleton).shape[1]
    print(f"  Skeleton feature dim: {skeleton_feat_dim}")

    # For now, use pre-extracted skeleton features + trajectory
    # Video features will be simulated or extracted separately
    print("\nLoading datasets...")

    # Check if video features are pre-extracted
    train_video_feat_path = features_dir / 'train_video_features.npy'

    if train_video_feat_path.exists():
        print("  Using pre-extracted video features")
        # Use pre-extracted features
        train_video_features = np.load(train_video_feat_path)
        val_video_features = np.load(features_dir / 'val_video_features.npy')
        video_feat_dim = train_video_features.shape[1]
    else:
        print("  Video features not found - using skeleton features as proxy")
        print("  (For full pipeline, run video feature extraction first)")
        # Use skeleton features as a placeholder for video features
        # In production, you'd extract actual video features
        video_feat_dim = skeleton_feat_dim

    # Load skeleton and trajectory features
    train_skeletons = np.load(features_dir / 'train_skeletons.npy').astype(np.float32)
    train_trajectories = np.load(features_dir / 'train_trajectories.npy').astype(np.float32)
    train_labels = np.load(features_dir / 'train_labels.npy')

    val_skeletons = np.load(features_dir / 'val_skeletons.npy').astype(np.float32)
    val_trajectories = np.load(features_dir / 'val_trajectories.npy').astype(np.float32)
    val_labels = np.load(features_dir / 'val_labels.npy')

    print(f"  Train samples: {len(train_labels)}")
    print(f"  Val samples: {len(val_labels)}")

    # Normalize
    def normalize_skeletons(skeletons):
        center = (skeletons[:, :, :, 7:8] + skeletons[:, :, :, 8:9]) / 2
        skeletons = skeletons - center
        std = np.std(skeletons, axis=(1, 2, 3), keepdims=True)
        std[std < 1e-6] = 1.0
        return skeletons / std

    def normalize_trajectories(trajectories, mean=None, std=None):
        if mean is None:
            mean = trajectories.mean(axis=0, keepdims=True)
        if std is None:
            std = trajectories.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (trajectories - mean) / std, mean, std

    train_skeletons = normalize_skeletons(train_skeletons)
    val_skeletons = normalize_skeletons(val_skeletons)
    train_trajectories, traj_mean, traj_std = normalize_trajectories(train_trajectories)
    val_trajectories, _, _ = normalize_trajectories(val_trajectories, traj_mean, traj_std)

    trajectory_dim = train_trajectories.shape[1]
    print(f"  Trajectory dim: {trajectory_dim}")

    # Create fusion model
    print("\nCreating fusion model...")

    # Use TwoStreamFusion (skeleton + trajectory) if no video features
    if not train_video_feat_path.exists():
        print("  Using Two-Stream Fusion (Skeleton + Trajectory)")

        # Simple fusion model for skeleton + trajectory
        class SkeletonTrajectoryFusion(nn.Module):
            def __init__(self, skeleton_dim, trajectory_dim, hidden_dim=256, num_classes=2):
                super().__init__()
                self.skeleton_proj = nn.Sequential(
                    nn.Linear(skeleton_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                self.trajectory_proj = nn.Sequential(
                    nn.Linear(trajectory_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dim, num_classes)
                )

            def forward(self, skeleton_feat, trajectory):
                s = self.skeleton_proj(skeleton_feat)
                t = self.trajectory_proj(trajectory)
                combined = torch.cat([s, t], dim=-1)
                return self.classifier(combined)

        fusion_model = SkeletonTrajectoryFusion(
            skeleton_dim=skeleton_feat_dim,
            trajectory_dim=trajectory_dim,
            hidden_dim=256,
            num_classes=2
        ).to(device)

        # Create simple dataset
        class SimpleFusionDataset(Dataset):
            def __init__(self, skeletons, trajectories, labels):
                self.skeletons = torch.from_numpy(skeletons)
                self.trajectories = torch.from_numpy(trajectories)
                self.labels = torch.from_numpy(labels).long()

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.skeletons[idx], self.trajectories[idx], self.labels[idx]

        train_dataset = SimpleFusionDataset(train_skeletons, train_trajectories, train_labels)
        val_dataset = SimpleFusionDataset(val_skeletons, val_trajectories, val_labels)

    else:
        fusion_model = MultiModalFreeThrowClassifier(
            video_dim=video_feat_dim,
            skeleton_dim=skeleton_feat_dim,
            trajectory_dim=trajectory_dim,
            fusion_dim=256,
            num_classes=2,
            dropout=0.5
        ).to(device)

    total_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"  Fusion model parameters: {total_params / 1e6:.2f}M")

    # Training setup
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        fusion_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Training loop
    print("\nTraining fusion model...")
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        fusion_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            if len(batch) == 3:
                skeletons, trajectories, labels = batch
                skeletons = skeletons.to(device)
                trajectories = trajectories.to(device)
                labels = labels.to(device)

                # Extract skeleton features
                with torch.no_grad():
                    skeleton_feat = skeleton_model.extract_features(skeletons)

                optimizer.zero_grad()
                outputs = fusion_model(skeleton_feat, trajectories)
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
            for batch in val_loader:
                if len(batch) == 3:
                    skeletons, trajectories, labels = batch
                    skeletons = skeletons.to(device)
                    trajectories = trajectories.to(device)
                    labels = labels.to(device)

                    skeleton_feat = skeleton_model.extract_features(skeletons)
                    outputs = fusion_model(skeleton_feat, trajectories)
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
            }, checkpoint_dir / 'best_fusion_model.pt')
            print(f"  -> Saved best model (val_acc: {val_acc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save history
    with open(checkpoint_dir / 'fusion_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("FUSION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Best Val F1: {best_val_f1:.3f}")
    print(f"\nComparison:")
    print(f"  Video Swin alone:     74.6%")
    print(f"  ST-GCN alone:         69.4%")
    print(f"  Fusion (Skel+Traj):   {best_val_acc*100:.1f}%")


if __name__ == '__main__':
    main()
