"""
Pose-Based Baseline for Basketball Free Throw Prediction.

Uses SAM 3D Body (Meta's 3D human mesh recovery) for pose estimation and extracts
biomechanical features to predict free throw outcomes. This serves as a traditional
CV baseline to compare against the end-to-end deep learning approach.

Supports:
- SAM 3D Body (default): State-of-the-art 3D human mesh recovery
- MediaPipe (fallback): Lightweight 2D/3D pose estimation

Pipeline:
1. Extract pose landmarks from video frames using SAM 3D Body
2. Compute shooting-relevant features (angles, alignment, motion)
3. Train a classifier (XGBoost) on these features
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import warnings
import torch

warnings.filterwarnings('ignore')


# SAM 3D Body joint indices (based on MHR - Momentum Human Rig)
# These map to the standard body skeleton joints
SAM3D_JOINTS = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
}

# MediaPipe pose landmark indices (fallback)
MEDIAPIPE_LANDMARKS = {
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
}


class SAM3DBodyPoseExtractor:
    """
    SAM 3D Body pose extractor for basketball shooting analysis.

    Uses Meta's SAM 3D Body model for accurate 3D human mesh recovery,
    providing superior pose estimation especially for sports movements.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-3d-body-dinov3",
        device: str = None
    ):
        """
        Initialize SAM 3D Body model.

        Args:
            model_name: HuggingFace model ID or local checkpoint path
            device: Device to run on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize the model on first use."""
        if self._initialized:
            return

        try:
            # Try to import SAM 3D Body
            from sam_3d_body.notebook.utils import setup_sam_3d_body

            print(f"Loading SAM 3D Body model: {self.model_name}")
            self.estimator = setup_sam_3d_body(hf_repo_id=self.model_name)
            self._initialized = True
            print("SAM 3D Body initialized successfully")

        except ImportError:
            print("SAM 3D Body not installed. Please install with:")
            print("  pip install sam-3d-body")
            print("  # Or clone from: https://github.com/facebookresearch/sam-3d-body")
            raise

    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 3D pose from a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Joint positions array (N_joints, 3) or None if detection fails
        """
        self._lazy_init()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Run SAM 3D Body inference
            outputs = self.estimator.process_one_image(rgb_frame)

            if outputs is None or 'joints' not in outputs:
                return None

            # Extract joint positions (3D coordinates)
            joints = outputs['joints']  # Shape: (N_joints, 3)

            # Add visibility score (all visible for SAM3D)
            visibility = np.ones((joints.shape[0], 1))
            joints_with_vis = np.concatenate([joints, visibility], axis=1)

            return joints_with_vis

        except Exception as e:
            print(f"SAM 3D Body inference error: {e}")
            return None

    def get_joint_indices(self) -> Dict[str, int]:
        """Get joint name to index mapping."""
        return SAM3D_JOINTS


class MediaPipePoseExtractor:
    """
    MediaPipe pose extractor (fallback option).

    Lightweight alternative to SAM 3D Body for faster processing
    or when SAM 3D Body is not available.
    """

    def __init__(self, model_complexity: int = 1):
        """
        Initialize MediaPipe pose model.

        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
        """
        self.model_complexity = model_complexity
        self.pose = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize MediaPipe on first use."""
        if self._initialized:
            return

        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            min_detection_confidence=0.5
        )
        self._initialized = True

    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from a single frame.

        Args:
            frame: BGR image

        Returns:
            Landmarks array (33, 4) [x, y, z, visibility] or None
        """
        self._lazy_init()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(landmarks)

        return None

    def get_joint_indices(self) -> Dict[str, int]:
        """Get joint name to index mapping."""
        return MEDIAPIPE_LANDMARKS

    def close(self):
        """Release resources."""
        if self.pose:
            self.pose.close()


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b given three points a, b, c.

    Args:
        a, b, c: Points as numpy arrays (x, y, z)

    Returns:
        Angle in degrees
    """
    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))

    return np.degrees(angle)


def calculate_3d_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(a - b)


def extract_shooting_features(
    landmarks_sequence: List[np.ndarray],
    joint_indices: Dict[str, int],
    is_3d: bool = True
) -> Optional[Dict[str, float]]:
    """
    Extract shooting-relevant features from a sequence of pose landmarks.

    Features include:
    - Joint angles (elbow, shoulder, knee, hip)
    - Body alignment metrics
    - Motion dynamics (velocity, smoothness)
    - Release point characteristics
    - 3D-specific features (depth, trajectory in 3D space)

    Args:
        landmarks_sequence: List of landmark arrays from video frames
        joint_indices: Mapping from joint names to indices
        is_3d: Whether landmarks include full 3D coordinates

    Returns:
        Dictionary of feature names to values
    """
    features = {}

    if len(landmarks_sequence) < 3:
        return None

    # Filter out None landmarks
    valid_landmarks = [lm for lm in landmarks_sequence if lm is not None]

    if len(valid_landmarks) < 3:
        return None

    # Get joint indices
    R_SHOULDER = joint_indices['right_shoulder']
    R_ELBOW = joint_indices['right_elbow']
    R_WRIST = joint_indices['right_wrist']
    R_HIP = joint_indices['right_hip']
    R_KNEE = joint_indices['right_knee']
    R_ANKLE = joint_indices['right_ankle']
    L_SHOULDER = joint_indices['left_shoulder']
    L_HIP = joint_indices['left_hip']

    # Initialize feature lists
    elbow_angles = []
    shoulder_angles = []
    knee_angles = []
    hip_angles = []
    wrist_heights = []
    wrist_depths = []  # 3D feature
    shoulder_tilts = []
    elbow_wrist_alignments = []
    body_lean_angles = []  # 3D feature
    wrist_positions_3d = []

    for lm in valid_landmarks:
        # Extract 3D coordinates (first 3 columns)
        coords = lm[:, :3]

        # ELBOW ANGLE (shoulder-elbow-wrist)
        elbow_angle = calculate_angle(
            coords[R_SHOULDER],
            coords[R_ELBOW],
            coords[R_WRIST]
        )
        elbow_angles.append(elbow_angle)

        # SHOULDER ANGLE (hip-shoulder-elbow)
        shoulder_angle = calculate_angle(
            coords[R_HIP],
            coords[R_SHOULDER],
            coords[R_ELBOW]
        )
        shoulder_angles.append(shoulder_angle)

        # KNEE ANGLE (hip-knee-ankle)
        knee_angle = calculate_angle(
            coords[R_HIP],
            coords[R_KNEE],
            coords[R_ANKLE]
        )
        knee_angles.append(knee_angle)

        # HIP ANGLE (shoulder-hip-knee)
        hip_angle = calculate_angle(
            coords[R_SHOULDER],
            coords[R_HIP],
            coords[R_KNEE]
        )
        hip_angles.append(hip_angle)

        # WRIST HEIGHT (y-coordinate)
        wrist_heights.append(coords[R_WRIST, 1])

        # WRIST DEPTH (z-coordinate) - 3D specific
        if is_3d:
            wrist_depths.append(coords[R_WRIST, 2])
            wrist_positions_3d.append(coords[R_WRIST].copy())

        # SHOULDER TILT (difference in y between shoulders)
        shoulder_tilts.append(coords[R_SHOULDER, 1] - coords[L_SHOULDER, 1])

        # ELBOW-WRIST ALIGNMENT (x-difference)
        elbow_wrist_alignments.append(abs(coords[R_ELBOW, 0] - coords[R_WRIST, 0]))

        # BODY LEAN ANGLE (vertical alignment) - 3D specific
        if is_3d:
            # Calculate lean angle using hip-shoulder vector vs vertical
            hip_mid = (coords[R_HIP] + coords[L_HIP]) / 2
            shoulder_mid = (coords[R_SHOULDER] + coords[L_SHOULDER]) / 2
            body_vector = shoulder_mid - hip_mid
            vertical = np.array([0, -1, 0])  # Pointing up
            lean = calculate_angle(vertical, np.zeros(3), body_vector)
            body_lean_angles.append(lean)

    # ============== AGGREGATE STATISTICS ==============

    # Elbow angle features
    features['elbow_angle_mean'] = np.mean(elbow_angles)
    features['elbow_angle_std'] = np.std(elbow_angles)
    features['elbow_angle_min'] = np.min(elbow_angles)
    features['elbow_angle_max'] = np.max(elbow_angles)
    features['elbow_angle_range'] = np.max(elbow_angles) - np.min(elbow_angles)

    # Shoulder angle features
    features['shoulder_angle_mean'] = np.mean(shoulder_angles)
    features['shoulder_angle_std'] = np.std(shoulder_angles)
    features['shoulder_angle_max'] = np.max(shoulder_angles)

    # Knee angle features
    features['knee_angle_mean'] = np.mean(knee_angles)
    features['knee_angle_std'] = np.std(knee_angles)
    features['knee_angle_min'] = np.min(knee_angles)

    # Hip angle features
    features['hip_angle_mean'] = np.mean(hip_angles)
    features['hip_angle_std'] = np.std(hip_angles)

    # Wrist trajectory features
    features['wrist_height_min'] = np.min(wrist_heights)  # Highest point
    features['wrist_height_range'] = np.max(wrist_heights) - np.min(wrist_heights)

    # Motion features (2D)
    wrist_velocity = np.diff(wrist_heights)
    features['wrist_velocity_mean'] = np.mean(wrist_velocity)
    features['wrist_velocity_max'] = np.min(wrist_velocity)  # Most negative = fastest upward
    features['motion_smoothness'] = np.std(np.diff(wrist_velocity))  # Jerk

    # Alignment features
    features['shoulder_tilt_mean'] = np.mean(shoulder_tilts)
    features['shoulder_tilt_std'] = np.std(shoulder_tilts)
    features['elbow_wrist_align_mean'] = np.mean(elbow_wrist_alignments)
    features['elbow_wrist_align_min'] = np.min(elbow_wrist_alignments)

    # Release point features (using last few frames)
    n_release = min(5, len(valid_landmarks))
    features['release_elbow_angle'] = np.mean(elbow_angles[-n_release:])
    features['release_shoulder_angle'] = np.mean(shoulder_angles[-n_release:])
    features['release_wrist_height'] = np.mean(wrist_heights[-n_release:])

    # ============== 3D-SPECIFIC FEATURES ==============
    if is_3d and wrist_depths:
        # Depth features
        features['wrist_depth_mean'] = np.mean(wrist_depths)
        features['wrist_depth_std'] = np.std(wrist_depths)
        features['wrist_depth_range'] = np.max(wrist_depths) - np.min(wrist_depths)

        # 3D trajectory features
        wrist_positions = np.array(wrist_positions_3d)
        if len(wrist_positions) > 1:
            # 3D velocity
            velocities_3d = np.diff(wrist_positions, axis=0)
            speeds = np.linalg.norm(velocities_3d, axis=1)
            features['wrist_speed_mean_3d'] = np.mean(speeds)
            features['wrist_speed_max_3d'] = np.max(speeds)

            # Trajectory straightness (ratio of direct distance to path length)
            direct_dist = np.linalg.norm(wrist_positions[-1] - wrist_positions[0])
            path_length = np.sum(speeds)
            features['trajectory_straightness'] = direct_dist / (path_length + 1e-8)

        # Body lean features
        if body_lean_angles:
            features['body_lean_mean'] = np.mean(body_lean_angles)
            features['body_lean_std'] = np.std(body_lean_angles)
            features['body_lean_at_release'] = np.mean(body_lean_angles[-n_release:])

    return features


def process_video(
    video_path: str,
    pose_extractor: Union[SAM3DBodyPoseExtractor, MediaPipePoseExtractor],
    n_frames: int = 16
) -> Optional[Dict[str, float]]:
    """
    Process a video and extract pose features.

    Args:
        video_path: Path to video file
        pose_extractor: Pose extraction model (SAM3D or MediaPipe)
        n_frames: Number of frames to sample

    Returns:
        Dictionary of features or None if processing fails
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    landmarks_sequence = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            landmarks = pose_extractor.extract_pose(frame)
            landmarks_sequence.append(landmarks)

    cap.release()

    # Determine if using 3D features based on extractor type
    is_3d = isinstance(pose_extractor, SAM3DBodyPoseExtractor)
    joint_indices = pose_extractor.get_joint_indices()

    # Extract features
    features = extract_shooting_features(landmarks_sequence, joint_indices, is_3d)

    return features


def extract_features_from_dataset(
    splits_dir: str,
    output_dir: str,
    use_sam3d: bool = True,
    n_frames: int = 16
):
    """
    Extract pose features from all videos in the dataset.

    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv
        output_dir: Directory to save feature CSVs
        use_sam3d: Whether to use SAM 3D Body (True) or MediaPipe (False)
        n_frames: Number of frames to sample per video
    """
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pose extractor
    if use_sam3d:
        print("Using SAM 3D Body for pose estimation")
        try:
            pose_extractor = SAM3DBodyPoseExtractor()
        except ImportError:
            print("SAM 3D Body not available, falling back to MediaPipe")
            pose_extractor = MediaPipePoseExtractor()
    else:
        print("Using MediaPipe for pose estimation")
        pose_extractor = MediaPipePoseExtractor()

    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        csv_path = splits_dir / f'{split}.csv'

        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping...")
            continue

        df = pd.read_csv(csv_path)

        all_features = []
        failed = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            features = process_video(row['video_path'], pose_extractor, n_frames)

            if features is not None:
                features['video_path'] = row['video_path']
                features['label'] = row['label']
                all_features.append(features)
            else:
                failed += 1

        if all_features:
            features_df = pd.DataFrame(all_features)
            features_df.to_csv(output_dir / f'{split}_pose_features.csv', index=False)
            print(f"  Processed: {len(all_features)}, Failed: {failed}")
        else:
            print(f"  No features extracted for {split} split")

    # Cleanup
    if isinstance(pose_extractor, MediaPipePoseExtractor):
        pose_extractor.close()

    print(f"\nFeatures saved to {output_dir}")


def train_pose_classifier(
    features_dir: str,
    output_dir: str = 'models'
):
    """
    Train XGBoost classifier on pose features.

    Args:
        features_dir: Directory containing pose feature CSVs
        output_dir: Directory to save trained model
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        print("Please install dependencies: pip install xgboost scikit-learn joblib")
        return

    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    train_df = pd.read_csv(features_dir / 'train_pose_features.csv')
    val_df = pd.read_csv(features_dir / 'val_pose_features.csv')
    test_df = pd.read_csv(features_dir / 'test_pose_features.csv')

    # Prepare data
    feature_cols = [c for c in train_df.columns if c not in ['video_path', 'label']]

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    # Handle missing values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Calculate class weight for imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"\nClass distribution - Make: {n_pos}, Miss: {n_neg}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Train XGBoost
    print("\nTraining XGBoost classifier on pose features...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    print("\n" + "="*50)
    print("POSE BASELINE RESULTS (SAM 3D Body)")
    print("="*50)

    for name, X, y in [('Train', X_train, y_train),
                       ('Val', X_val, y_val),
                       ('Test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)

        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC:  {auc:.4f}")

    print("\nTest Set Classification Report:")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test, target_names=['Miss', 'Make']))

    # Feature importance
    print("\nTop 10 Important Features:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    joblib.dump(model, output_dir / 'pose_xgboost.joblib')
    joblib.dump(scaler, output_dir / 'pose_scaler.joblib')
    importance.to_csv(output_dir / 'pose_feature_importance.csv', index=False)

    print(f"\nModel saved to {output_dir}")

    return model, scaler


def visualize_pose_detection(
    video_path: str,
    output_path: str,
    use_sam3d: bool = True,
    n_frames: int = 8
):
    """
    Visualize pose detection on sample frames from a video.

    Args:
        video_path: Path to video file
        output_path: Path to save visualization
        use_sam3d: Whether to use SAM 3D Body or MediaPipe
        n_frames: Number of frames to visualize
    """
    import matplotlib.pyplot as plt

    # Initialize pose extractor
    if use_sam3d:
        try:
            pose_extractor = SAM3DBodyPoseExtractor()
            # For visualization with SAM3D, we need the visualization utilities
            from sam_3d_body.tools.vis_utils import visualize_sample_together
            has_sam3d_vis = True
        except ImportError:
            print("SAM 3D Body not available, using MediaPipe")
            pose_extractor = MediaPipePoseExtractor()
            has_sam3d_vis = False
    else:
        pose_extractor = MediaPipePoseExtractor()
        has_sam3d_vis = False

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, n_frames // 2, figsize=(3 * n_frames // 2, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if isinstance(pose_extractor, MediaPipePoseExtractor):
                # MediaPipe visualization
                import mediapipe as mp
                mp_drawing = mp.solutions.drawing_utils

                results = pose_extractor.pose.process(rgb_frame)

                if results.pose_landmarks:
                    annotated = rgb_frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        pose_extractor.mp_pose.POSE_CONNECTIONS
                    )
                    axes[i].imshow(annotated)
                else:
                    axes[i].imshow(rgb_frame)
            else:
                # SAM 3D Body visualization
                if has_sam3d_vis:
                    outputs = pose_extractor.estimator.process_one_image(rgb_frame)
                    if outputs is not None:
                        rend_img = visualize_sample_together(
                            frame, outputs, pose_extractor.estimator.faces
                        )
                        axes[i].imshow(cv2.cvtColor(rend_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
                    else:
                        axes[i].imshow(rgb_frame)
                else:
                    axes[i].imshow(rgb_frame)

            axes[i].set_title(f'Frame {idx}', fontsize=10)
            axes[i].axis('off')

    cap.release()

    if isinstance(pose_extractor, MediaPipePoseExtractor):
        pose_extractor.close()

    plt.suptitle(f"Pose Detection: {'SAM 3D Body' if use_sam3d else 'MediaPipe'}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved pose visualization to: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pose-based baseline for free throw prediction')
    parser.add_argument('--action', type=str, required=True,
                        choices=['extract', 'train', 'visualize'],
                        help='Action to perform')
    parser.add_argument('--splits_dir', type=str, default='data/splits',
                        help='Directory with split CSVs')
    parser.add_argument('--features_dir', type=str, default='data/pose_features',
                        help='Directory for pose features')
    parser.add_argument('--video_path', type=str, help='Video path for visualization')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--use_mediapipe', action='store_true',
                        help='Use MediaPipe instead of SAM 3D Body')
    parser.add_argument('--n_frames', type=int, default=16,
                        help='Number of frames to sample per video')

    args = parser.parse_args()
    use_sam3d = not args.use_mediapipe

    if args.action == 'extract':
        extract_features_from_dataset(
            args.splits_dir,
            args.features_dir,
            use_sam3d=use_sam3d,
            n_frames=args.n_frames
        )

    elif args.action == 'train':
        train_pose_classifier(args.features_dir)

    elif args.action == 'visualize':
        if args.video_path and args.output:
            visualize_pose_detection(
                args.video_path,
                args.output,
                use_sam3d=use_sam3d
            )
        else:
            print("Please provide --video_path and --output for visualization")
