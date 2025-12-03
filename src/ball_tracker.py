"""
Basketball Trajectory Tracking and Feature Extraction.

Tracks the basketball through video frames and extracts trajectory-based features
that are predictive of free throw success:
- Release point and angle
- Arc height
- Ball velocity
- Entry angle to basket

This is a key differentiator from existing approaches that only use pose.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class BallDetector:
    """
    Basketball detection using color-based detection and optional YOLO.

    For basketball, orange color detection is surprisingly effective
    and much faster than deep learning approaches.
    """

    def __init__(
        self,
        use_yolo: bool = False,
        yolo_model: str = 'yolov8n.pt',
        min_radius: int = 10,
        max_radius: int = 100
    ):
        self.use_yolo = use_yolo
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Orange color range for basketball (HSV)
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([25, 255, 255])

        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_model)
                # Basketball is class 32 in COCO (sports ball)
                self.ball_class = 32
            except ImportError:
                print("YOLO not available, falling back to color detection")
                self.use_yolo = False

    def detect_color(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect basketball using orange color segmentation.

        Args:
            frame: BGR image

        Returns:
            (x, y, radius) of detected ball or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for orange color
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the most circular contour (basketball is round)
        best_circle = None
        best_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < np.pi * self.min_radius**2:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            if circularity > best_circularity and circularity > 0.5:
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)

                if self.min_radius <= radius <= self.max_radius:
                    best_circle = (int(x), int(y), int(radius))
                    best_circularity = circularity

        return best_circle

    def detect_yolo(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect basketball using YOLO.

        Args:
            frame: BGR image

        Returns:
            (x, y, radius) of detected ball or None
        """
        if not self.use_yolo:
            return None

        results = self.yolo(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == self.ball_class:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)
                    radius = int(max(x2 - x1, y2 - y1) / 2)
                    return (x, y, radius)

        return None

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect basketball in frame.

        Args:
            frame: BGR image

        Returns:
            (x, y, radius) or None
        """
        if self.use_yolo:
            result = self.detect_yolo(frame)
            if result:
                return result

        return self.detect_color(frame)


class TrajectoryTracker:
    """
    Track basketball trajectory across frames with smoothing and interpolation.
    """

    def __init__(self, max_gap: int = 5, smooth_window: int = 3):
        self.max_gap = max_gap
        self.smooth_window = smooth_window
        self.positions = []
        self.frames_since_detection = 0

    def update(self, detection: Optional[Tuple[int, int, int]], frame_idx: int):
        """
        Update tracker with new detection.

        Args:
            detection: (x, y, radius) or None
            frame_idx: Current frame index
        """
        if detection:
            self.positions.append({
                'frame': frame_idx,
                'x': detection[0],
                'y': detection[1],
                'radius': detection[2],
                'interpolated': False
            })
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1

            # Interpolate if gap is small
            if len(self.positions) >= 2 and self.frames_since_detection <= self.max_gap:
                last = self.positions[-1]
                prev = self.positions[-2]

                # Linear interpolation
                dx = last['x'] - prev['x']
                dy = last['y'] - prev['y']

                self.positions.append({
                    'frame': frame_idx,
                    'x': last['x'] + dx,
                    'y': last['y'] + dy,
                    'radius': last['radius'],
                    'interpolated': True
                })

    def get_trajectory(self) -> np.ndarray:
        """
        Get smoothed trajectory.

        Returns:
            Array of shape (N, 2) with (x, y) positions
        """
        if not self.positions:
            return np.array([])

        coords = np.array([[p['x'], p['y']] for p in self.positions])

        # Smooth trajectory
        if len(coords) >= self.smooth_window:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            coords[:, 0] = np.convolve(coords[:, 0], kernel, mode='same')
            coords[:, 1] = np.convolve(coords[:, 1], kernel, mode='same')

        return coords

    def reset(self):
        """Reset tracker for new video."""
        self.positions = []
        self.frames_since_detection = 0


class TrajectoryFeatureExtractor:
    """
    Extract trajectory-based features predictive of free throw success.

    Based on physics of basketball shooting:
    - Optimal release angle: ~52 degrees
    - Higher arc = larger margin for error
    - Consistent release point = better accuracy
    """

    def __init__(self, frame_height: int = 480, frame_width: int = 640):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def extract_features(
        self,
        trajectory: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, float]:
        """
        Extract features from ball trajectory.

        Args:
            trajectory: Array of (x, y) positions
            fps: Video frame rate

        Returns:
            Dictionary of trajectory features
        """
        features = {}

        if len(trajectory) < 5:
            # Not enough data
            return self._empty_features()

        # Normalize coordinates
        trajectory_norm = trajectory.copy()
        trajectory_norm[:, 0] /= self.frame_width
        trajectory_norm[:, 1] /= self.frame_height

        # Basic trajectory statistics
        features['traj_length'] = len(trajectory)
        features['traj_x_range'] = np.ptp(trajectory_norm[:, 0])
        features['traj_y_range'] = np.ptp(trajectory_norm[:, 1])

        # Release point (first detection)
        features['release_x'] = trajectory_norm[0, 0]
        features['release_y'] = trajectory_norm[0, 1]

        # Highest point (arc apex)
        min_y_idx = np.argmin(trajectory[:, 1])  # Y increases downward
        features['apex_x'] = trajectory_norm[min_y_idx, 0]
        features['apex_y'] = trajectory_norm[min_y_idx, 1]
        features['arc_height'] = features['release_y'] - features['apex_y']

        # Calculate velocities
        dt = 1.0 / fps
        velocities = np.diff(trajectory_norm, axis=0) / dt

        if len(velocities) > 0:
            features['release_vx'] = velocities[0, 0]
            features['release_vy'] = velocities[0, 1]
            features['release_speed'] = np.linalg.norm(velocities[0])

            # Release angle (from horizontal)
            if velocities[0, 0] != 0:
                # Note: y is inverted (0 at top), so negate for angle calculation
                features['release_angle'] = np.degrees(
                    np.arctan2(-velocities[0, 1], velocities[0, 0])
                )
            else:
                features['release_angle'] = 90.0

            # Velocity statistics
            speeds = np.linalg.norm(velocities, axis=1)
            features['avg_speed'] = np.mean(speeds)
            features['max_speed'] = np.max(speeds)
            features['speed_std'] = np.std(speeds)

        else:
            features.update({
                'release_vx': 0, 'release_vy': 0, 'release_speed': 0,
                'release_angle': 0, 'avg_speed': 0, 'max_speed': 0, 'speed_std': 0
            })

        # Trajectory curvature (how parabolic is the path)
        if len(trajectory_norm) >= 3:
            # Fit parabola to trajectory
            try:
                coeffs = np.polyfit(trajectory_norm[:, 0], trajectory_norm[:, 1], 2)
                features['parabola_a'] = coeffs[0]  # Curvature coefficient
                features['parabola_b'] = coeffs[1]
                features['parabola_c'] = coeffs[2]

                # Residual from parabolic fit (how well does parabola fit)
                y_pred = np.polyval(coeffs, trajectory_norm[:, 0])
                features['parabola_residual'] = np.mean((trajectory_norm[:, 1] - y_pred) ** 2)
            except:
                features['parabola_a'] = 0
                features['parabola_b'] = 0
                features['parabola_c'] = 0
                features['parabola_residual'] = 1.0
        else:
            features['parabola_a'] = 0
            features['parabola_b'] = 0
            features['parabola_c'] = 0
            features['parabola_residual'] = 1.0

        # Entry angle (last part of trajectory)
        if len(velocities) >= 3:
            # Average of last few velocity vectors
            final_v = np.mean(velocities[-3:], axis=0)
            if final_v[0] != 0:
                features['entry_angle'] = np.degrees(np.arctan2(-final_v[1], final_v[0]))
            else:
                features['entry_angle'] = -90.0  # Straight down
        else:
            features['entry_angle'] = 0

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features when trajectory is too short."""
        return {
            'traj_length': 0,
            'traj_x_range': 0, 'traj_y_range': 0,
            'release_x': 0, 'release_y': 0,
            'apex_x': 0, 'apex_y': 0, 'arc_height': 0,
            'release_vx': 0, 'release_vy': 0, 'release_speed': 0,
            'release_angle': 0, 'avg_speed': 0, 'max_speed': 0, 'speed_std': 0,
            'parabola_a': 0, 'parabola_b': 0, 'parabola_c': 0, 'parabola_residual': 1.0,
            'entry_angle': 0
        }


class TrajectoryEncoder(nn.Module):
    """
    Neural network encoder for trajectory features.

    Transforms raw trajectory into learned representation
    that can be fused with other modalities.
    """

    def __init__(
        self,
        input_dim: int = 20,  # Number of trajectory features
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trajectory features of shape (N, input_dim)

        Returns:
            Encoded features of shape (N, output_dim)
        """
        return self.encoder(x)


class TrajectoryLSTM(nn.Module):
    """
    LSTM-based trajectory encoder for sequence modeling.

    Takes raw trajectory positions as input and learns
    temporal patterns.
    """

    def __init__(
        self,
        input_dim: int = 2,  # (x, y) positions
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),  # *2 for bidirectional
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Trajectory positions of shape (N, T, 2)
            lengths: Actual sequence lengths for packing

        Returns:
            Encoded features of shape (N, output_dim)
        """
        if lengths is not None:
            # Pack padded sequences
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        _, (h_n, _) = self.lstm(x)

        # Concatenate forward and backward hidden states
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return self.fc(h_combined)


def process_video_trajectory(
    video_path: str,
    detector: BallDetector,
    tracker: TrajectoryTracker,
    feature_extractor: TrajectoryFeatureExtractor,
    sample_frames: int = 16
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process video to extract ball trajectory and features.

    Args:
        video_path: Path to video file
        detector: Ball detector instance
        tracker: Trajectory tracker instance
        feature_extractor: Feature extractor instance
        sample_frames: Number of frames to sample

    Returns:
        Tuple of (trajectory array, feature dict)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Update feature extractor dimensions
    feature_extractor.frame_height = frame_height
    feature_extractor.frame_width = frame_width

    tracker.reset()

    # Sample frame indices
    indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        detection = detector.detect(frame)
        tracker.update(detection, idx)

    cap.release()

    trajectory = tracker.get_trajectory()
    features = feature_extractor.extract_features(trajectory, fps)

    return trajectory, features


if __name__ == '__main__':
    print("Testing Ball Tracker implementation...")

    # Test detector
    detector = BallDetector(use_yolo=False)
    print("Ball detector created (color-based)")

    # Test tracker
    tracker = TrajectoryTracker()
    tracker.update((100, 200, 15), 0)
    tracker.update((110, 180, 15), 1)
    tracker.update((120, 165, 15), 2)
    tracker.update(None, 3)  # Missing detection
    tracker.update((140, 150, 15), 4)
    trajectory = tracker.get_trajectory()
    print(f"Trajectory shape: {trajectory.shape}")

    # Test feature extractor
    extractor = TrajectoryFeatureExtractor()
    features = extractor.extract_features(trajectory)
    print(f"Extracted {len(features)} features:")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v:.3f}")

    # Test neural encoders
    traj_encoder = TrajectoryEncoder(input_dim=len(features))
    feat_tensor = torch.tensor([list(features.values())], dtype=torch.float32)
    encoded = traj_encoder(feat_tensor)
    print(f"Encoded feature shape: {encoded.shape}")

    lstm_encoder = TrajectoryLSTM()
    traj_tensor = torch.randn(4, 16, 2)  # (batch, time, xy)
    lstm_out = lstm_encoder(traj_tensor)
    print(f"LSTM output shape: {lstm_out.shape}")

    print("\nAll tests passed!")
