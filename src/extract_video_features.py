"""
Extract Video Swin Transformer features for fusion training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import create_video_swin


class VideoDataset(Dataset):
    """Simple video dataset for feature extraction."""

    def __init__(self, video_ids_path: str, data_dir: str, n_frames: int = 16, img_size: int = 224):
        self.data_dir = Path(data_dir)
        self.n_frames = n_frames
        self.img_size = img_size

        with open(video_ids_path, 'r') as f:
            self.video_paths = [line.strip() for line in f.readlines()]

    def _load_video(self, video_path: str):
        """Load and preprocess video frames."""
        full_path = self.data_dir / video_path

        cap = cv2.VideoCapture(str(full_path))
        if not cap.isOpened():
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
                # ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                frame = (frame - mean) / std
                frames.append(frame)
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3)))

        cap.release()

        # (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames, axis=0)
        video = video.transpose(3, 0, 1, 2)

        return torch.from_numpy(video.astype(np.float32))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = self._load_video(self.video_paths[idx])
        return video, idx


def extract_features(model, loader, device):
    """Extract features from all videos."""
    model.eval()
    all_features = []

    with torch.no_grad():
        for videos, _ in tqdm(loader, desc="Extracting features"):
            videos = videos.to(device)
            features = model(videos)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def main():
    parser = argparse.ArgumentParser(description='Extract Video Swin features')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base data directory')
    parser.add_argument('--features_dir', type=str, default='data/stgcn_features',
                        help='Directory with extracted features')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/stage2-epoch=20-val_acc=0.746.ckpt',
                        help='Video Swin checkpoint')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    print("=" * 60)
    print("VIDEO SWIN FEATURE EXTRACTION")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    features_dir = Path(args.features_dir)

    # Load Video Swin model
    print("\nLoading Video Swin model...")
    model = create_video_swin(
        num_classes=2,
        pretrained=False,
        model_size='tiny'
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle Lightning checkpoint format
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            # Remove 'model.' prefix if present
            if k.startswith('model.'):
                state_dict[k[6:]] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Remove classification head to get features
    feature_dim = model.head[1].in_features
    model.head = nn.Identity()
    model = model.to(device)
    print(f"  Feature dimension: {feature_dim}")

    # Process each split
    for split in ['train', 'val', 'test']:
        video_ids_path = features_dir / f'{split}_video_ids.txt'

        if not video_ids_path.exists():
            print(f"\nSkipping {split} - no video IDs file")
            continue

        print(f"\nProcessing {split} split...")

        dataset = VideoDataset(
            str(video_ids_path),
            args.data_dir,
            n_frames=16,
            img_size=224
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        features = extract_features(model, loader, device)
        print(f"  Extracted features shape: {features.shape}")

        # Save features
        output_path = features_dir / f'{split}_video_features.npy'
        np.save(output_path, features.numpy())
        print(f"  Saved to {output_path}")

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
