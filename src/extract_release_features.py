"""
Extract clean, lean features for basketball free throw prediction.

Key SOTA features:
1. Release point timing
2. Joint angles at release (elbow, shoulder, knee)
3. Ball trajectory features
4. Ball-hand distance at release

Creates a focused ~15 feature vector that matches SOTA approaches.
"""
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# SMPL joint indices
SMPL_JOINTS = {
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
    'neck': 12,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
}


def compute_joint_angle(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def extract_features_from_pkl(pkl_path):
    """Extract clean features from a single video's pkl file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    features = {}

    # Basic info
    n_frames = data.get('n_frames', 16)
    frame_size = data.get('frame_size', (320, 240))
    frame_w, frame_h = frame_size

    # --- Release Point Features ---
    release_idx = data.get('release_frame_idx', n_frames // 2)
    release_info = data.get('release_info', {})

    features['release_frame_ratio'] = release_idx / n_frames  # When in video
    features['release_ball_hand_dist'] = release_info.get('distance', 100) / max(frame_w, frame_h)  # Normalized
    features['release_velocity'] = release_info.get('velocity', 50) / max(frame_w, frame_h)  # Normalized
    features['ball_above_hand'] = float(release_info.get('ball_above_hand', True))

    # --- Joint Angles at Release ---
    frames_data = data.get('frames_data', [])

    if release_idx < len(frames_data):
        release_frame = frames_data[release_idx]
        pose_3d = release_frame.get('pose_3d', {})
        joints_3d = pose_3d.get('joints_3d', None)

        if joints_3d and len(joints_3d) >= 22:
            joints = np.array(joints_3d)

            # Right arm angles (most shooters are right-handed)
            r_shoulder = joints[SMPL_JOINTS['right_shoulder']]
            r_elbow = joints[SMPL_JOINTS['right_elbow']]
            r_wrist = joints[SMPL_JOINTS['right_wrist']]
            r_hip = joints[SMPL_JOINTS['right_hip']]

            features['r_elbow_angle'] = compute_joint_angle(r_shoulder, r_elbow, r_wrist)
            features['r_shoulder_angle'] = compute_joint_angle(r_hip, r_shoulder, r_elbow)

            # Left arm angles
            l_shoulder = joints[SMPL_JOINTS['left_shoulder']]
            l_elbow = joints[SMPL_JOINTS['left_elbow']]
            l_wrist = joints[SMPL_JOINTS['left_wrist']]
            l_hip = joints[SMPL_JOINTS['left_hip']]

            features['l_elbow_angle'] = compute_joint_angle(l_shoulder, l_elbow, l_wrist)
            features['l_shoulder_angle'] = compute_joint_angle(l_hip, l_shoulder, l_elbow)

            # Knee angles
            r_knee = joints[SMPL_JOINTS['right_knee']]
            r_ankle = joints[SMPL_JOINTS['right_ankle']]
            features['r_knee_angle'] = compute_joint_angle(r_hip, r_knee, r_ankle)

            l_knee = joints[SMPL_JOINTS['left_knee']]
            l_ankle = joints[SMPL_JOINTS['left_ankle']]
            features['l_knee_angle'] = compute_joint_angle(l_hip, l_knee, l_ankle)

            # Arm height at release (normalized)
            features['r_wrist_height'] = float(r_wrist[1])  # Y coordinate
            features['l_wrist_height'] = float(l_wrist[1])
        else:
            # Default angles if pose not available
            features['r_elbow_angle'] = 90.0
            features['r_shoulder_angle'] = 90.0
            features['l_elbow_angle'] = 90.0
            features['l_shoulder_angle'] = 90.0
            features['r_knee_angle'] = 160.0
            features['l_knee_angle'] = 160.0
            features['r_wrist_height'] = 0.0
            features['l_wrist_height'] = 0.0
    else:
        # Defaults
        features['r_elbow_angle'] = 90.0
        features['r_shoulder_angle'] = 90.0
        features['l_elbow_angle'] = 90.0
        features['l_shoulder_angle'] = 90.0
        features['r_knee_angle'] = 160.0
        features['l_knee_angle'] = 160.0
        features['r_wrist_height'] = 0.0
        features['l_wrist_height'] = 0.0

    # --- Ball Trajectory Features ---
    ball_positions = []
    for frame in frames_data:
        ball_pos = frame.get('ball_centroid')
        ball_positions.append(ball_pos)

    valid_balls = [(i, p) for i, p in enumerate(ball_positions) if p is not None]

    if len(valid_balls) >= 2:
        # Detection rate
        features['ball_detection_rate'] = len(valid_balls) / n_frames

        # Ball trajectory direction
        first_ball = np.array(valid_balls[0][1])
        last_ball = np.array(valid_balls[-1][1])
        traj_vec = last_ball - first_ball

        features['ball_traj_x'] = traj_vec[0] / frame_w  # Normalized
        features['ball_traj_y'] = traj_vec[1] / frame_h  # Normalized (negative = going up)

        # Arc height (lowest y = highest point in image coords)
        all_y = [p[1] for _, p in valid_balls]
        min_y = min(all_y)
        max_y = max(all_y)
        features['ball_arc_height'] = (max_y - min_y) / frame_h

        # Ball at release
        if release_idx < len(ball_positions) and ball_positions[release_idx]:
            release_ball = np.array(ball_positions[release_idx])
            features['ball_x_at_release'] = release_ball[0] / frame_w
            features['ball_y_at_release'] = release_ball[1] / frame_h
        else:
            # Use nearest valid ball position
            nearest = min(valid_balls, key=lambda x: abs(x[0] - release_idx))
            features['ball_x_at_release'] = nearest[1][0] / frame_w
            features['ball_y_at_release'] = nearest[1][1] / frame_h
    else:
        features['ball_detection_rate'] = 0.0
        features['ball_traj_x'] = 0.0
        features['ball_traj_y'] = 0.0
        features['ball_arc_height'] = 0.1
        features['ball_x_at_release'] = 0.5
        features['ball_y_at_release'] = 0.5

    return features


def extract_all_features(data_dir, output_dir):
    """Extract features for all splits."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = None

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Skipping {split} - directory not found")
            continue

        pkl_files = sorted(split_dir.glob('*.pkl'))
        print(f"\n{split}: {len(pkl_files)} files")

        all_features = []
        all_labels = []

        for pkl_path in tqdm(pkl_files, desc=f"Processing {split}"):
            try:
                # Extract features
                features = extract_features_from_pkl(pkl_path)

                # Get label from pkl
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                label = data.get('label', 0)

                if feature_names is None:
                    feature_names = list(features.keys())

                all_features.append([features[k] for k in feature_names])
                all_labels.append(label)

            except Exception as e:
                print(f"Error processing {pkl_path}: {e}")
                continue

        if all_features:
            features_arr = np.array(all_features, dtype=np.float32)
            labels_arr = np.array(all_labels, dtype=np.int64)

            np.save(output_dir / f'{split}_release_features.npy', features_arr)
            np.save(output_dir / f'{split}_labels.npy', labels_arr)

            print(f"  Saved: {features_arr.shape} features, {len(labels_arr)} labels")
            print(f"  Class distribution: {np.bincount(labels_arr)}")

    # Save feature names
    if feature_names:
        with open(output_dir / 'feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"\nFeature names ({len(feature_names)}): {feature_names}")


def main():
    parser = argparse.ArgumentParser(description="Extract release features from SAM3 data")
    parser.add_argument('--data_dir', type=str, default='data/sam3_extracted',
                        help='Directory with train/val/test pkl files')
    parser.add_argument('--output_dir', type=str, default='data/release_features',
                        help='Output directory for features')
    args = parser.parse_args()

    print("=" * 60)
    print("EXTRACTING CLEAN RELEASE FEATURES")
    print("=" * 60)
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    extract_all_features(args.data_dir, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
