"""
Train classifier using hoop-relative features.
Research suggests hoop-relative features have 3-4x larger effect sizes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (75% makes)."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce)
        # alpha weighting for class imbalance
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


class HoopFeatureDataset(Dataset):
    """Dataset for hoop-relative features."""
    def __init__(self, features_path, labels_path, video_features_path=None,
                 mean=None, std=None, video_mean=None, video_std=None):
        self.features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path)

        # Normalize hoop features
        if mean is None:
            self.mean = self.features.mean(axis=0, keepdims=True)
            self.std = self.features.std(axis=0, keepdims=True)
            self.std[self.std < 1e-6] = 1.0
        else:
            self.mean = mean
            self.std = std
        self.features = (self.features - self.mean) / self.std
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional video features
        self.video_features = None
        if video_features_path and Path(video_features_path).exists():
            self.video_features = np.load(video_features_path).astype(np.float32)
            if video_mean is None:
                self.video_mean = self.video_features.mean(axis=0, keepdims=True)
                self.video_std = self.video_features.std(axis=0, keepdims=True)
                self.video_std[self.video_std < 1e-6] = 1.0
            else:
                self.video_mean = video_mean
                self.video_std = video_std
            self.video_features = (self.video_features - self.video_mean) / self.video_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.video_features is not None:
            return (
                torch.from_numpy(self.features[idx]),
                torch.from_numpy(self.video_features[idx]),
                torch.tensor(self.labels[idx], dtype=torch.long)
            )
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class HoopClassifier(nn.Module):
    """Simple MLP classifier for hoop features only."""
    def __init__(self, input_dim=18, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.model(x)


class FusionClassifier(nn.Module):
    """Fusion of hoop features + video features."""
    def __init__(self, hoop_dim=18, video_dim=768, fusion_dim=128, dropout=0.4):
        super().__init__()

        # Hoop feature encoder
        self.hoop_encoder = nn.Sequential(
            nn.Linear(hoop_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Video feature encoder
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, hoop_feat, video_feat):
        hoop_enc = self.hoop_encoder(hoop_feat)
        video_enc = self.video_encoder(video_feat)
        combined = torch.cat([hoop_enc, video_enc], dim=-1)
        fused = self.fusion(combined)
        return self.classifier(fused)


def train_hoop_only(args):
    """Train with hoop features only."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    hoop_dir = Path(args.hoop_dir)

    # Load datasets
    print('\nLoading datasets...')
    train_dataset = HoopFeatureDataset(
        hoop_dir / 'train_hoop_features.npy',
        hoop_dir / 'train_labels.npy'
    )

    val_dataset = HoopFeatureDataset(
        hoop_dir / 'val_hoop_features.npy',
        hoop_dir / 'val_labels.npy',
        mean=train_dataset.mean,
        std=train_dataset.std
    )

    print(f'  Train: {len(train_dataset)} samples, {train_dataset.features.shape[1]} features')
    print(f'  Val: {len(val_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = HoopClassifier(
        input_dim=train_dataset.features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    print(f'  Model params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K')

    # Class distribution
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    print(f'  Class distribution: {class_counts}')

    # Loss - focal loss for class imbalance
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print('  Using Focal Loss (alpha=0.25, gamma=2.0)')
    else:
        class_weights = torch.tensor(
            [len(train_labels) / (2 * c) for c in class_counts],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f'  Using weighted CrossEntropy: {class_weights.cpu().numpy()}')

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

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            features, labels = batch[0].to(device), batch[-1].to(device)

            optimizer.zero_grad()
            outputs = model(features)
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
            for batch in val_loader:
                features, labels = batch[0].to(device), batch[-1].to(device)
                outputs = model(features)
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

        # Prediction distribution
        pred_dist = np.bincount(all_preds, minlength=2)

        print(f'Epoch {epoch+1:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | '
              f'Preds: {pred_dist} | LR: {lr:.2e}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0

            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, checkpoint_dir / 'best_hoop_classifier.pt')
            print(f'  -> Saved best model (acc: {val_acc:.3f}, f1: {val_f1:.3f})')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print('\n' + '=' * 60)
    print('HOOP FEATURES TRAINING COMPLETE')
    print('=' * 60)
    print(f'Best Val Accuracy: {best_val_acc:.3f}')
    print(f'Best Val F1: {best_val_f1:.3f}')

    return best_val_acc, best_val_f1


def train_fusion(args):
    """Train with hoop + video features."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    hoop_dir = Path(args.hoop_dir)
    video_dir = Path(args.video_dir)

    # Load datasets
    print('\nLoading datasets...')
    train_dataset = HoopFeatureDataset(
        hoop_dir / 'train_hoop_features.npy',
        hoop_dir / 'train_labels.npy',
        video_dir / 'train_video_features.npy'
    )

    val_dataset = HoopFeatureDataset(
        hoop_dir / 'val_hoop_features.npy',
        hoop_dir / 'val_labels.npy',
        video_dir / 'val_video_features.npy',
        mean=train_dataset.mean,
        std=train_dataset.std,
        video_mean=train_dataset.video_mean,
        video_std=train_dataset.video_std
    )

    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Hoop features: {train_dataset.features.shape[1]}')
    print(f'  Video features: {train_dataset.video_features.shape[1]}')
    print(f'  Val: {len(val_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = FusionClassifier(
        hoop_dim=train_dataset.features.shape[1],
        video_dim=train_dataset.video_features.shape[1],
        fusion_dim=128,
        dropout=args.dropout
    ).to(device)

    print(f'  Model params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K')

    # Class distribution
    train_labels = train_dataset.labels
    class_counts = np.bincount(train_labels)
    print(f'  Class distribution: {class_counts}')

    # Loss - focal loss
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print('  Using Focal Loss')
    else:
        class_weights = torch.tensor(
            [len(train_labels) / (2 * c) for c in class_counts],
            dtype=torch.float32
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f'  Using weighted CrossEntropy: {class_weights.cpu().numpy()}')

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

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
            hoop_feat, video_feat, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(hoop_feat, video_feat)
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
            for batch in val_loader:
                hoop_feat, video_feat, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = model(hoop_feat, video_feat)
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

        pred_dist = np.bincount(all_preds, minlength=2)

        print(f'Epoch {epoch+1:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | '
              f'Preds: {pred_dist} | LR: {lr:.2e}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0

            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, checkpoint_dir / 'best_hoop_fusion.pt')
            print(f'  -> Saved best model (acc: {val_acc:.3f}, f1: {val_f1:.3f})')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print('\n' + '=' * 60)
    print('HOOP+VIDEO FUSION TRAINING COMPLETE')
    print('=' * 60)
    print(f'Best Val Accuracy: {best_val_acc:.3f}')
    print(f'Best Val F1: {best_val_f1:.3f}')

    return best_val_acc, best_val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hoop_dir', type=str, default='data/hoop_features')
    parser.add_argument('--video_dir', type=str, default='data/stgcn_features')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/hoop')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'weighted_ce'])
    parser.add_argument('--mode', type=str, default='hoop_only', choices=['hoop_only', 'fusion'])
    args = parser.parse_args()

    print('=' * 60)
    print('HOOP-RELATIVE FEATURES TRAINING')
    print('Research shows 3-4x larger effect sizes vs raw trajectory')
    print('=' * 60)

    if args.mode == 'hoop_only':
        train_hoop_only(args)
    else:
        train_fusion(args)

    print('\nComparison:')
    print('  Majority Baseline:      75.3%')
    print('  Video Swin alone:       74.6%')
    print('  Combined (V+T+P):       66.5%')


if __name__ == '__main__':
    main()
