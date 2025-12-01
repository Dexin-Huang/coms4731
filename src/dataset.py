"""
Video Dataset for Basketball Free Throw Prediction.

Loads videos from Basketball-51 dataset and samples frames uniformly.
Includes support for handling class imbalance via:
- Oversampling minority class
- Weighted random sampling
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, Callable, List, Tuple
from sklearn.model_selection import train_test_split


class FreeThrowDataset(Dataset):
    """
    Dataset for loading basketball free throw videos.

    Args:
        video_paths: List of paths to video files
        labels: List of labels (0=miss, 1=make)
        n_frames: Number of frames to sample uniformly from each video
        transform: Optional transform to apply to frames
        frame_size: Tuple of (height, width) to resize frames
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        n_frames: int = 16,
        transform: Optional[Callable] = None,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.n_frames = n_frames
        self.transform = transform
        self.frame_size = frame_size

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Load and sample frames
        frames = self._load_video(video_path)

        # Apply transforms if provided
        if self.transform:
            frames = self.transform(frames)
        else:
            # Default: normalize to [0, 1] and convert to tensor
            frames = torch.from_numpy(frames).float() / 255.0
            # Rearrange from (T, H, W, C) to (C, T, H, W) for 3D convolutions
            frames = frames.permute(3, 0, 1, 2)

        return frames, label

    def _load_video(self, video_path: str) -> np.ndarray:
        """
        Load video and sample n_frames uniformly.

        Returns:
            numpy array of shape (n_frames, H, W, C)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to sample uniformly
        if total_frames <= self.n_frames:
            indices = list(range(total_frames))
            # Pad with last frame if video is too short
            while len(indices) < self.n_frames:
                indices.append(total_frames - 1)
        else:
            indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                frames.append(frame)
            else:
                # If frame read fails, use last successful frame or zeros
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8))

        cap.release()

        return np.stack(frames, axis=0)


class VideoTransform:
    """
    Transform pipeline for video data.

    Applies spatial and temporal augmentations.
    """

    def __init__(
        self,
        mode: str = 'train',
        frame_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, ...] = (0.45, 0.45, 0.45),
        std: Tuple[float, ...] = (0.225, 0.225, 0.225)
    ):
        self.mode = mode
        self.frame_size = frame_size
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Apply transforms to video frames.

        Args:
            frames: numpy array of shape (T, H, W, C)

        Returns:
            torch.Tensor of shape (C, T, H, W)
        """
        # Convert to tensor and normalize to [0, 1]
        frames = torch.from_numpy(frames).float() / 255.0

        # Rearrange to (C, T, H, W)
        frames = frames.permute(3, 0, 1, 2)

        if self.mode == 'train':
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                frames = torch.flip(frames, dims=[3])  # Flip width dimension

            # Random crop (if frame is larger than target)
            if frames.shape[2] > self.frame_size[0] or frames.shape[3] > self.frame_size[1]:
                frames = self._random_crop(frames)

            # Color jitter (simplified)
            frames = self._color_jitter(frames)

        # Normalize with ImageNet stats
        frames = (frames - self.mean) / self.std

        return frames

    def _random_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """Random spatial crop."""
        _, _, h, w = frames.shape
        new_h, new_w = self.frame_size

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        return frames[:, :, top:top+new_h, left:left+new_w]

    def _color_jitter(self, frames: torch.Tensor, brightness: float = 0.2, contrast: float = 0.2) -> torch.Tensor:
        """Simple color jitter augmentation."""
        # Brightness
        brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * brightness
        frames = frames * brightness_factor

        # Contrast
        contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * contrast
        mean = frames.mean()
        frames = (frames - mean) * contrast_factor + mean

        # Clamp to valid range
        frames = torch.clamp(frames, 0, 1)

        return frames


def prepare_data_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train/val/test splits from Basketball-51 free throw data.

    Args:
        data_dir: Directory containing ft0/ (miss) and ft1/ (make) folders
        output_dir: Directory to save split CSV files
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    labels = []

    # Load miss videos (ft0)
    miss_dir = data_dir / 'ft0'
    if miss_dir.exists():
        for video_file in miss_dir.glob('*.avi'):
            video_paths.append(str(video_file))
            labels.append(0)
        for video_file in miss_dir.glob('*.mp4'):
            video_paths.append(str(video_file))
            labels.append(0)

    # Load make videos (ft1)
    make_dir = data_dir / 'ft1'
    if make_dir.exists():
        for video_file in make_dir.glob('*.avi'):
            video_paths.append(str(video_file))
            labels.append(1)
        for video_file in make_dir.glob('*.mp4'):
            video_paths.append(str(video_file))
            labels.append(1)

    print(f"Found {len(video_paths)} videos total")
    print(f"  Miss (ft0): {labels.count(0)}")
    print(f"  Make (ft1): {labels.count(1)}")

    if len(video_paths) == 0:
        raise ValueError(f"No videos found in {data_dir}. Make sure ft0/ and ft1/ folders exist.")

    # Create DataFrame
    df = pd.DataFrame({
        'video_path': video_paths,
        'label': labels
    })

    # Stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df['label'],
        random_state=random_state
    )

    # Save splits
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def oversample_minority_class(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Oversample the minority class to balance the dataset.

    Args:
        df: DataFrame with 'video_path' and 'label' columns
        random_state: Random seed for reproducibility

    Returns:
        Balanced DataFrame with oversampled minority class
    """
    # Count samples per class
    class_counts = df['label'].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    n_majority = class_counts[majority_class]
    n_minority = class_counts[minority_class]

    print(f"Original class distribution: {dict(class_counts)}")

    # Separate classes
    df_majority = df[df['label'] == majority_class]
    df_minority = df[df['label'] == minority_class]

    # Oversample minority class
    df_minority_oversampled = df_minority.sample(
        n=n_majority,
        replace=True,
        random_state=random_state
    )

    # Combine
    df_balanced = pd.concat([df_majority, df_minority_oversampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Balanced class distribution: {dict(df_balanced['label'].value_counts())}")

    return df_balanced


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for handling class imbalance.

    This samples with replacement, giving higher probability to minority class samples.

    Args:
        labels: List of class labels

    Returns:
        WeightedRandomSampler instance
    """
    labels = np.array(labels)

    # Calculate class weights (inverse frequency)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[labels]

    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler


def get_dataloaders(
    data_dir: str,
    splits_dir: str,
    batch_size: int = 8,
    n_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    balance_strategy: str = 'none'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.

    Args:
        data_dir: Directory containing the videos (not used if CSVs have full paths)
        splits_dir: Directory containing train.csv, val.csv, test.csv
        batch_size: Batch size for training
        n_frames: Number of frames to sample per video
        frame_size: Size to resize frames to
        num_workers: Number of worker processes for data loading
        balance_strategy: Strategy for handling class imbalance:
            - 'none': No balancing (use shuffle)
            - 'oversample': Oversample minority class in the dataset
            - 'weighted': Use WeightedRandomSampler

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    splits_dir = Path(splits_dir)

    # Load splits
    train_df = pd.read_csv(splits_dir / 'train.csv')
    val_df = pd.read_csv(splits_dir / 'val.csv')
    test_df = pd.read_csv(splits_dir / 'test.csv')

    # Apply oversampling if requested
    if balance_strategy == 'oversample':
        print("\nApplying oversampling to training set...")
        train_df = oversample_minority_class(train_df)

    # Create transforms
    train_transform = VideoTransform(mode='train', frame_size=frame_size)
    val_transform = VideoTransform(mode='val', frame_size=frame_size)

    # Create datasets
    train_dataset = FreeThrowDataset(
        video_paths=train_df['video_path'].tolist(),
        labels=train_df['label'].tolist(),
        n_frames=n_frames,
        transform=train_transform,
        frame_size=frame_size
    )

    val_dataset = FreeThrowDataset(
        video_paths=val_df['video_path'].tolist(),
        labels=val_df['label'].tolist(),
        n_frames=n_frames,
        transform=val_transform,
        frame_size=frame_size
    )

    test_dataset = FreeThrowDataset(
        video_paths=test_df['video_path'].tolist(),
        labels=test_df['label'].tolist(),
        n_frames=n_frames,
        transform=val_transform,
        frame_size=frame_size
    )

    # Create train loader with appropriate sampler
    if balance_strategy == 'weighted':
        print("\nUsing weighted random sampling for training...")
        sampler = create_weighted_sampler(train_df['label'].tolist())
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Cannot use shuffle with sampler
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Prepare data splits')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with ft0/ and ft1/ folders')
    parser.add_argument('--output_dir', type=str, default='data/splits', help='Output directory for split CSVs')
    args = parser.parse_args()

    prepare_data_splits(args.data_dir, args.output_dir)
