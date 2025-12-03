"""
Train ST-GCN on extracted skeleton sequences.
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

from stgcn import LightweightSTGCN, get_stgcn_model_info


class SkeletonDataset(Dataset):
    """Dataset for skeleton sequences."""

    def __init__(self, skeletons_path: str, labels_path: str):
        self.skeletons = np.load(skeletons_path)  # (N, C, T, V)
        self.labels = np.load(labels_path)

        # Normalize skeletons
        self.skeletons = self._normalize(self.skeletons)

    def _normalize(self, skeletons):
        """Normalize skeleton coordinates."""
        # Center around hip (joint 7 or 8)
        # skeletons shape: (N, C, T, V) where C=3 (x,y,z)

        # Use mean of left and right hip as center
        center = (skeletons[:, :, :, 7:8] + skeletons[:, :, :, 8:9]) / 2
        skeletons = skeletons - center

        # Scale to unit variance
        std = np.std(skeletons, axis=(1, 2, 3), keepdims=True)
        std[std < 1e-6] = 1.0
        skeletons = skeletons / std

        return skeletons.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        skeleton = torch.from_numpy(self.skeletons[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return skeleton, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for skeletons, labels in tqdm(loader, desc="Training", leave=False):
        skeletons = skeletons.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(skeletons)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * skeletons.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for skeletons, labels in tqdm(loader, desc="Validating", leave=False):
            skeletons = skeletons.to(device)
            labels = labels.to(device)

            outputs = model(skeletons)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * skeletons.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate F1 score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return total_loss / total, correct / total, f1


def main():
    parser = argparse.ArgumentParser(description='Train ST-GCN')
    parser.add_argument('--data_dir', type=str, default='data/stgcn_features',
                        help='Directory with skeleton features')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/stgcn',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 60)
    print("ST-GCN TRAINING")
    print("=" * 60)

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading data...")
    train_dataset = SkeletonDataset(
        data_dir / 'train_skeletons.npy',
        data_dir / 'train_labels.npy'
    )
    val_dataset = SkeletonDataset(
        data_dir / 'val_skeletons.npy',
        data_dir / 'val_labels.npy'
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Skeleton shape: {train_dataset.skeletons.shape}")

    # Check class distribution
    train_labels = train_dataset.labels
    print(f"  Train class distribution: {np.bincount(train_labels)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    print("\nCreating model...")
    model = LightweightSTGCN(
        num_classes=2,
        in_channels=3,
        num_joints=17,
        hidden_channels=[64, 128, 256],
        dropout=0.5
    )
    model = model.to(device)

    info = get_stgcn_model_info(model)
    print(f"  Parameters: {info['total_params_millions']:.2f}M")

    # Loss with class weights for imbalanced data
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
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
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
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
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_acc: {val_acc:.3f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save final model and history
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_f1': val_f1,
    }, checkpoint_dir / 'final_model.pt')

    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.3f}")
    print(f"Best Val F1: {best_val_f1:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
