"""
Data Preparation Script for Basketball Free Throw Prediction.

This script:
1. Extracts the Basketball-51 dataset (if needed)
2. Filters free throw videos only (ft0 and ft1)
3. Creates train/val/test splits

Usage:
    python scripts/prepare_data.py --data_dir path/to/basketball51 --output_dir data/raw
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset import prepare_data_splits


def extract_free_throws(source_dir: str, output_dir: str) -> int:
    """
    Extract free throw videos from Basketball-51 dataset.

    The Basketball-51 dataset has the following structure:
    - ft0/ : Free throw miss
    - ft1/ : Free throw make

    Args:
        source_dir: Path to extracted Basketball-51 directory
        output_dir: Path to output directory

    Returns:
        Number of videos copied
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    total_copied = 0

    # Process ft0 (miss) and ft1 (make)
    for folder in ['ft0', 'ft1']:
        src_folder = source_dir / folder
        dst_folder = output_dir / folder

        if not src_folder.exists():
            print(f"Warning: {src_folder} does not exist, skipping...")
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)

        # Copy video files
        for video_file in src_folder.glob('*'):
            if video_file.suffix.lower() in ['.avi', '.mp4', '.mov', '.mkv']:
                dst_path = dst_folder / video_file.name
                if not dst_path.exists():
                    shutil.copy2(video_file, dst_path)
                    total_copied += 1

        print(f"Processed {folder}/: {len(list(dst_folder.glob('*')))} videos")

    return total_copied


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Basketball-51 Free Throw Dataset'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Path to extracted Basketball-51 dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Output directory for free throw videos'
    )
    parser.add_argument(
        '--splits_dir',
        type=str,
        default='data/splits',
        help='Directory for train/val/test split CSVs'
    )
    parser.add_argument(
        '--skip_copy',
        action='store_true',
        help='Skip copying videos (use if already copied)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Basketball Free Throw Dataset Preparation")
    print("="*60)

    # Step 1: Extract free throws
    if not args.skip_copy:
        print("\nStep 1: Extracting free throw videos...")
        n_copied = extract_free_throws(args.source_dir, args.output_dir)
        print(f"Copied {n_copied} new videos to {args.output_dir}")
    else:
        print("\nStep 1: Skipping copy (--skip_copy flag set)")

    # Step 2: Create splits
    print("\nStep 2: Creating train/val/test splits...")
    try:
        train_df, val_df, test_df = prepare_data_splits(
            data_dir=args.output_dir,
            output_dir=args.splits_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        print("\nData preparation complete!")
        print(f"  Videos directory: {args.output_dir}")
        print(f"  Splits directory: {args.splits_dir}")
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. Basketball-51 dataset is downloaded from Kaggle")
        print("  2. Dataset is extracted")
        print("  3. --source_dir points to the extracted folder containing ft0/ and ft1/")
        return 1

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Start training:")
    print("   cd src && python train.py --config ../configs/config.yaml")
    print("\n3. Evaluate model:")
    print("   cd src && python evaluate.py --checkpoint ../checkpoints/best.ckpt")

    return 0


if __name__ == '__main__':
    sys.exit(main())
