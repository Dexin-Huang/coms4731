"""
Extract skeleton sequences for ST-GCN training.

Converts pose data into the format expected by ST-GCN:
- Input: Video files
- Output: Skeleton sequences of shape (C, T, V) where
  - C = 3 (x, y, z coordinates)
  - T = number of frames (e.g., 16)
  - V = number of joints (17 for simplified skeleton)

Also extracts ball trajectory features for the fusion model.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

# MediaPipe indices for 17-joint skeleton (matching ST-GCN)
SKELETON_INDICES = {
    'nose': 0,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_eye': 2,
    'right_eye': 5,
    'left_ear': 7,
    'right_ear': 8,
}

# Mapping from MediaPipe 33 joints to our 17 joints
MP_TO_17 = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 7, 8, 2, 5]


class SkeletonExtractor:
    """Extract skeleton sequences from videos using MediaPipe."""

    def __init__(self, n_frames: int = 16, n_joints: int = 17):
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.pose = None
        self._initialized = False

    def _lazy_init(self):
        if self._initialized:
            return

        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._initialized = True

    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract skeleton sequence from video.

        Args:
            video_path: Path to video file

        Returns:
            Skeleton array of shape (3, T, V) or None if extraction fails
        """
        self._lazy_init()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return None

        # Sample frame indices uniformly
        indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)

        skeleton_sequence = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                # Use zeros for missing frames
                skeleton_sequence.append(np.zeros((self.n_joints, 3)))
                continue

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract 17 joints from 33 MediaPipe landmarks
                joints = []
                for mp_idx in MP_TO_17:
                    lm = results.pose_landmarks.landmark[mp_idx]
                    joints.append([lm.x, lm.y, lm.z])
                skeleton_sequence.append(np.array(joints))
            else:
                # Use zeros for frames without detection
                skeleton_sequence.append(np.zeros((self.n_joints, 3)))

        cap.release()

        # Stack and transpose: (T, V, C) -> (C, T, V)
        skeleton = np.stack(skeleton_sequence, axis=0)  # (T, V, C)
        skeleton = skeleton.transpose(2, 0, 1)  # (C, T, V)

        return skeleton.astype(np.float32)

    def close(self):
        if self.pose:
            self.pose.close()


class BallTrajectoryExtractor:
    """Extract ball trajectory features from videos."""

    def __init__(self, n_frames: int = 16):
        self.n_frames = n_frames

        # Orange color range for basketball (HSV)
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([25, 255, 255])

    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect basketball using color segmentation."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find most circular contour
        best_circle = None
        best_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Min area threshold
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity > best_circularity and circularity > 0.5:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if 10 <= radius <= 100:
                    best_circle = (int(x), int(y), int(radius))
                    best_circularity = circularity

        return best_circle

    def extract_features(self, video_path: str) -> Dict[str, float]:
        """
        Extract trajectory features from video.

        Returns dictionary of 20 trajectory features.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._empty_features()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

        if total_frames < 1:
            cap.release()
            return self._empty_features()

        indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
        positions = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            detection = self.detect_ball(frame)
            if detection:
                x, y, r = detection
                positions.append([x / frame_width, y / frame_height])

        cap.release()

        if len(positions) < 5:
            return self._empty_features()

        trajectory = np.array(positions)
        return self._compute_features(trajectory, fps)

    def _compute_features(self, trajectory: np.ndarray, fps: float) -> Dict[str, float]:
        """Compute physics-based trajectory features."""
        features = {}

        features['traj_length'] = len(trajectory)
        features['traj_x_range'] = np.ptp(trajectory[:, 0])
        features['traj_y_range'] = np.ptp(trajectory[:, 1])

        # Release point
        features['release_x'] = trajectory[0, 0]
        features['release_y'] = trajectory[0, 1]

        # Arc apex (highest point - min y since y increases downward)
        min_y_idx = np.argmin(trajectory[:, 1])
        features['apex_x'] = trajectory[min_y_idx, 0]
        features['apex_y'] = trajectory[min_y_idx, 1]
        features['arc_height'] = features['release_y'] - features['apex_y']

        # Velocities
        dt = 1.0 / fps
        velocities = np.diff(trajectory, axis=0) / dt

        if len(velocities) > 0:
            features['release_vx'] = velocities[0, 0]
            features['release_vy'] = velocities[0, 1]
            features['release_speed'] = np.linalg.norm(velocities[0])

            if velocities[0, 0] != 0:
                features['release_angle'] = np.degrees(
                    np.arctan2(-velocities[0, 1], velocities[0, 0])
                )
            else:
                features['release_angle'] = 90.0

            speeds = np.linalg.norm(velocities, axis=1)
            features['avg_speed'] = np.mean(speeds)
            features['max_speed'] = np.max(speeds)
            features['speed_std'] = np.std(speeds)
        else:
            features.update({
                'release_vx': 0, 'release_vy': 0, 'release_speed': 0,
                'release_angle': 0, 'avg_speed': 0, 'max_speed': 0, 'speed_std': 0
            })

        # Parabola fit
        if len(trajectory) >= 3:
            try:
                coeffs = np.polyfit(trajectory[:, 0], trajectory[:, 1], 2)
                features['parabola_a'] = coeffs[0]
                features['parabola_b'] = coeffs[1]
                features['parabola_c'] = coeffs[2]
                y_pred = np.polyval(coeffs, trajectory[:, 0])
                features['parabola_residual'] = np.mean((trajectory[:, 1] - y_pred) ** 2)
            except:
                features.update({'parabola_a': 0, 'parabola_b': 0, 'parabola_c': 0, 'parabola_residual': 1.0})
        else:
            features.update({'parabola_a': 0, 'parabola_b': 0, 'parabola_c': 0, 'parabola_residual': 1.0})

        # Entry angle
        if len(velocities) >= 3:
            final_v = np.mean(velocities[-3:], axis=0)
            if final_v[0] != 0:
                features['entry_angle'] = np.degrees(np.arctan2(-final_v[1], final_v[0]))
            else:
                features['entry_angle'] = -90.0
        else:
            features['entry_angle'] = 0

        return features

    def _empty_features(self) -> Dict[str, float]:
        return {
            'traj_length': 0, 'traj_x_range': 0, 'traj_y_range': 0,
            'release_x': 0, 'release_y': 0, 'apex_x': 0, 'apex_y': 0, 'arc_height': 0,
            'release_vx': 0, 'release_vy': 0, 'release_speed': 0, 'release_angle': 0,
            'avg_speed': 0, 'max_speed': 0, 'speed_std': 0,
            'parabola_a': 0, 'parabola_b': 0, 'parabola_c': 0, 'parabola_residual': 1.0,
            'entry_angle': 0
        }


def process_split(
    split_csv: str,
    data_dir: str,
    output_dir: str,
    n_frames: int = 16,
    extract_trajectory: bool = True
):
    """
    Process all videos in a split and save skeleton sequences.

    Args:
        split_csv: Path to split CSV file
        data_dir: Base directory containing video folders
        output_dir: Directory to save extracted features
        n_frames: Number of frames to sample
        extract_trajectory: Whether to also extract ball trajectory
    """
    df = pd.read_csv(split_csv)
    split_name = Path(split_csv).stem

    skeleton_extractor = SkeletonExtractor(n_frames=n_frames)
    trajectory_extractor = BallTrajectoryExtractor(n_frames=n_frames) if extract_trajectory else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    skeletons = []
    trajectories = []
    labels = []
    video_ids = []

    print(f"\nProcessing {split_name} split ({len(df)} videos)...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = Path(data_dir) / row['video_path']

        if not video_path.exists():
            print(f"  Warning: {video_path} not found")
            continue

        # Extract skeleton
        skeleton = skeleton_extractor.extract_from_video(str(video_path))

        if skeleton is None:
            print(f"  Warning: Failed to extract skeleton from {video_path}")
            continue

        skeletons.append(skeleton)
        labels.append(row['label'])
        video_ids.append(row['video_path'])

        # Extract trajectory features
        if trajectory_extractor:
            traj_features = trajectory_extractor.extract_features(str(video_path))
            trajectories.append(list(traj_features.values()))

    skeleton_extractor.close()

    # Save as numpy arrays
    skeletons = np.stack(skeletons, axis=0)  # (N, C, T, V)
    labels = np.array(labels)

    np.save(output_path / f'{split_name}_skeletons.npy', skeletons)
    np.save(output_path / f'{split_name}_labels.npy', labels)

    print(f"  Saved skeletons: {skeletons.shape}")

    if trajectories:
        trajectories = np.array(trajectories, dtype=np.float32)
        np.save(output_path / f'{split_name}_trajectories.npy', trajectories)
        print(f"  Saved trajectories: {trajectories.shape}")

    # Save video IDs for reference
    with open(output_path / f'{split_name}_video_ids.txt', 'w') as f:
        for vid in video_ids:
            f.write(f"{vid}\n")

    print(f"  Processed {len(skeletons)} videos successfully")

    return skeletons, labels


def main():
    parser = argparse.ArgumentParser(description='Extract skeleton sequences for ST-GCN')
    parser.add_argument('--splits_dir', type=str, default='data/splits',
                        help='Directory containing split CSV files')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base directory containing video data')
    parser.add_argument('--output_dir', type=str, default='data/stgcn_features',
                        help='Output directory for extracted features')
    parser.add_argument('--n_frames', type=int, default=16,
                        help='Number of frames to sample per video')
    parser.add_argument('--no_trajectory', action='store_true',
                        help='Skip trajectory extraction')

    args = parser.parse_args()

    print("=" * 60)
    print("SKELETON & TRAJECTORY EXTRACTION FOR ST-GCN")
    print("=" * 60)

    splits_dir = Path(args.splits_dir)

    for split in ['train', 'val', 'test']:
        split_csv = splits_dir / f'{split}.csv'
        if split_csv.exists():
            process_split(
                str(split_csv),
                args.data_dir,
                args.output_dir,
                n_frames=args.n_frames,
                extract_trajectory=not args.no_trajectory
            )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
