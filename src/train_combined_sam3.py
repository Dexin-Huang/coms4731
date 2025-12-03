"""
Train combined model: Video Swin + SAM3 Trajectory + SAM3 Pose features.
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

class CombinedDataset(Dataset):
    def __init__(self, video_path, traj_path, pose_path, labels_path,
                 traj_mean=None, traj_std=None, pose_mean=None, pose_std=None):
        self.video_features = np.load(video_path).astype(np.float32)
        self.trajectories = np.load(traj_path).astype(np.float32)
        self.pose_features = np.load(pose_path).astype(np.float32)
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
        self.trajectories = np.nan_to_num(self.trajectories, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize pose features
        if pose_mean is None:
            self.pose_mean = self.pose_features.mean(axis=0, keepdims=True)
            self.pose_std = self.pose_features.std(axis=0, keepdims=True)
            self.pose_std[self.pose_std < 1e-6] = 1.0
        else:
            self.pose_mean = pose_mean
            self.pose_std = pose_std
        self.pose_features = (self.pose_features - self.pose_mean) / self.pose_std
        self.pose_features = np.nan_to_num(self.pose_features, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.video_features[idx]),
            torch.from_numpy(self.trajectories[idx]),
            torch.from_numpy(self.pose_features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class CombinedClassifier(nn.Module):
    def __init__(self, video_dim=768, traj_dim=20, pose_dim=100,
                 fusion_dim=256, num_classes=2, dropout=0.5):
        super().__init__()

        # Video encoder
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Trajectory encoder
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, fusion_dim // 2),
            nn.ReLU()
        )

        # Pose encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, fusion_dim // 2),
            nn.ReLU()
        )

        # Fusion layers
        total_dim = fusion_dim + fusion_dim // 2 + fusion_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, video_feat, traj_feat, pose_feat):
        video_enc = self.video_encoder(video_feat)
        traj_enc = self.traj_encoder(traj_feat)
        pose_enc = self.pose_encoder(pose_feat)

        combined = torch.cat([video_enc, traj_enc, pose_enc], dim=-1)
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='data/stgcn_features')
    parser.add_argument('--sam3_dir', type=str, default='data/sam3_features')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/combined_sam3')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    print('=' * 60)
    print('COMBINED SAM3 MODEL TRAINING')
    print('Video + Trajectory + Pose Features')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    video_dir = Path(args.video_dir)
    sam3_dir = Path(args.sam3_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print('\nLoading datasets...')
    train_dataset = CombinedDataset(
        video_dir / 'train_video_features.npy',
        sam3_dir / 'train_sam3_trajectories.npy',
        sam3_dir / 'train_pose_features.npy',
        sam3_dir / 'train_labels.npy'
    )

    val_dataset = CombinedDataset(
        video_dir / 'val_video_features.npy',
        sam3_dir / 'val_sam3_trajectories.npy',
        sam3_dir / 'val_pose_features.npy',
        sam3_dir / 'val_labels.npy',
        traj_mean=train_dataset.traj_mean,
        traj_std=train_dataset.traj_std,
        pose_mean=train_dataset.pose_mean,
        pose_std=train_dataset.pose_std
    )

    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Val: {len(val_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = CombinedClassifier(
        video_dim=train_dataset.video_features.shape[1],
        traj_dim=train_dataset.trajectories.shape[1],
        pose_dim=train_dataset.pose_features.shape[1],
        fusion_dim=256,
        num_classes=2,
        dropout=0.5
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Model params: {total_params / 1e6:.2f}M')

    # Loss with class weights
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    print(f'  Class distribution: {class_counts}')
    print(f'  Class weights: {class_weights.cpu().numpy()}')

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print('\nTraining...')
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for video, traj, pose, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            video, traj, pose, labels = video.to(device), traj.to(device), pose.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(video, traj, pose)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for video, traj, pose, labels in val_loader:
                video, traj, pose, labels = video.to(device), traj.to(device), pose.to(device), labels.to(device)

                outputs = model(video, traj, pose)
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
        tn = ((all_preds == 0) & (all_labels == 0)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        val_f1 = 2 * precision * recall / (precision + recall + 1e-8)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # Print prediction distribution
        pred_dist = np.bincount(all_preds, minlength=2)

        print(f'Epoch {epoch+1:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | '
              f'Preds: {pred_dist} | LR: {lr:.2e}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, checkpoint_dir / 'best_combined_sam3.pt')
            print(f'  -> Saved best model (acc: {val_acc:.3f}, f1: {val_f1:.3f})')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print('\n' + '=' * 60)
    print('COMBINED SAM3 TRAINING COMPLETE')
    print('=' * 60)
    print(f'Best Val Accuracy: {best_val_acc:.3f}')
    print(f'Best Val F1: {best_val_f1:.3f}')
    print(f'\nComparison:')
    print(f'  Majority Baseline:      75.3%')
    print(f'  Video Swin alone:       74.6%')
    print(f'  SAM3 Traj only:         75.2%')
    print(f'  COMBINED SAM3:          {best_val_acc*100:.1f}%')

if __name__ == '__main__':
    main()
