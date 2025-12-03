"""
Extract clean release-frame features from SAM3 PKL files.

Pipeline:
1. Filter to samples with good ball detection (>40% frames)
2. Detect release frame (ball-hand separation)
3. Extract pose at release frame only
4. Discard unclear samples
"""

import pickle
import numpy as np
from glob import glob
from pathlib import Path
import os
from tqdm import tqdm

# SMPL joint indices
JOINTS = {
    'R_SHOULDER': 17, 'L_SHOULDER': 16,
    'R_ELBOW': 19, 'L_ELBOW': 18,
    'R_WRIST': 21, 'L_WRIST': 20,
    'R_HIP': 2, 'L_HIP': 1,
    'R_KNEE': 5, 'L_KNEE': 4,
    'R_ANKLE': 8, 'L_ANKLE': 7,
}


def angle_between(a, b, c):
    """Calculate angle at point b between vectors ba and bc."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def find_release_frame(frames_data):
    """
    Find the release frame: when ball separates from shooter's hand.

    Returns: (frame_idx, confidence) or (None, reason)
    """
    ball_data = []

    for i, frame in enumerate(frames_data):
        if frame["ball_detected"] and frame["ball_centroid"] is not None:
            ball_data.append({
                'frame': i,
                'pos': np.array(frame["ball_centroid"]),
                'area': frame["ball_area"]
            })

    if len(ball_data) < 4:
        return None, "insufficient_ball"

    # Calculate ball velocity between consecutive detections
    velocities = []
    for i in range(1, len(ball_data)):
        dt = ball_data[i]['frame'] - ball_data[i-1]['frame']
        if dt > 0:
            vel = np.linalg.norm(ball_data[i]['pos'] - ball_data[i-1]['pos']) / dt
            velocities.append({
                'frame': ball_data[i]['frame'],
                'vel': vel,
                'pos': ball_data[i]['pos']
            })

    if len(velocities) < 2:
        return None, "insufficient_velocity"

    # Find frame with sudden velocity increase (release)
    for i in range(1, len(velocities)):
        if velocities[i]['vel'] > velocities[i-1]['vel'] * 1.3:  # 30% acceleration
            # Also check ball is moving upward (y decreasing in image coords)
            if i + 1 < len(velocities):
                dy = velocities[i+1]['pos'][1] - velocities[i]['pos'][1]
                if dy < 0:  # Moving up
                    return velocities[i]['frame'], "release_detected"

    # Fallback: use middle of ball detections as approximate release
    mid_idx = len(ball_data) // 2
    return ball_data[mid_idx]['frame'], "estimated_release"


def extract_pose_features(frame_data, label):
    """Extract biomechanical features from pose at release frame."""

    if not frame_data.get("pose_3d") or not frame_data["pose_3d"].get("joints_3d"):
        return None

    joints = np.array(frame_data["pose_3d"]["joints_3d"])

    if len(joints) < 22:  # Need at least up to wrist
        return None

    j = JOINTS  # shorthand

    features = {
        # Arm angles
        'r_elbow_angle': angle_between(joints[j['R_SHOULDER']], joints[j['R_ELBOW']], joints[j['R_WRIST']]),
        'l_elbow_angle': angle_between(joints[j['L_SHOULDER']], joints[j['L_ELBOW']], joints[j['L_WRIST']]),
        'r_shoulder_angle': angle_between(joints[j['R_HIP']], joints[j['R_SHOULDER']], joints[j['R_ELBOW']]),
        'l_shoulder_angle': angle_between(joints[j['L_HIP']], joints[j['L_SHOULDER']], joints[j['L_ELBOW']]),

        # Leg angles
        'r_knee_angle': angle_between(joints[j['R_HIP']], joints[j['R_KNEE']], joints[j['R_ANKLE']]),
        'l_knee_angle': angle_between(joints[j['L_HIP']], joints[j['L_KNEE']], joints[j['L_ANKLE']]),

        # Heights (y-coordinate, lower = higher in image)
        'r_wrist_height': -joints[j['R_WRIST']][1],  # Negate so higher = bigger
        'l_wrist_height': -joints[j['L_WRIST']][1],
        'r_elbow_height': -joints[j['R_ELBOW']][1],

        # Body position
        'shoulder_width': np.linalg.norm(joints[j['R_SHOULDER']][:2] - joints[j['L_SHOULDER']][:2]),
        'torso_lean': joints[j['R_SHOULDER']][0] - joints[j['R_HIP']][0],  # Forward lean
    }

    # Ball info if available
    if frame_data["ball_detected"] and frame_data["ball_centroid"]:
        ball = np.array(frame_data["ball_centroid"])
        r_wrist = joints[j['R_WRIST']][:2]
        features['ball_hand_dist'] = np.linalg.norm(ball - r_wrist)
        features['ball_height'] = -ball[1]  # Higher = bigger
        features['ball_x'] = ball[0]
    else:
        features['ball_hand_dist'] = np.nan
        features['ball_height'] = np.nan
        features['ball_x'] = np.nan

    features['label'] = label

    return features


def process_dataset(pkl_dir, min_ball_rate=0.4):
    """Process all PKL files and extract clean release features."""

    pkl_files = sorted(glob(f"{pkl_dir}/*.pkl"))
    print(f"Processing {len(pkl_files)} files from {pkl_dir}")

    results = {
        'clean': [],
        'no_ball': 0,
        'no_release': 0,
        'no_pose': 0,
    }

    for pkl_path in tqdm(pkl_files):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        frames = data["frames_data"]
        label = data["label"]
        video_id = Path(pkl_path).stem

        # Check ball detection rate
        ball_rate = sum(1 for f in frames if f["ball_detected"]) / len(frames)
        if ball_rate < min_ball_rate:
            results['no_ball'] += 1
            continue

        # Find release frame
        release_idx, status = find_release_frame(frames)
        if release_idx is None:
            results['no_release'] += 1
            continue

        # Extract pose features
        features = extract_pose_features(frames[release_idx], label)
        if features is None:
            results['no_pose'] += 1
            continue

        features['video_id'] = video_id
        features['release_frame'] = release_idx
        features['release_status'] = status
        features['ball_rate'] = ball_rate

        results['clean'].append(features)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/sam3_extracted/train')
    parser.add_argument('--output_dir', default='data/clean_release_features')
    parser.add_argument('--min_ball_rate', type=float, default=0.4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process train
    results = process_dataset(args.input_dir, args.min_ball_rate)

    print(f"\nResults:")
    print(f"  Clean samples: {len(results['clean'])}")
    print(f"  Rejected - no ball: {results['no_ball']}")
    print(f"  Rejected - no release: {results['no_release']}")
    print(f"  Rejected - no pose: {results['no_pose']}")

    if results['clean']:
        # Convert to arrays
        feature_names = [k for k in results['clean'][0].keys()
                        if k not in ['video_id', 'release_frame', 'release_status', 'label']]

        X = np.array([[s[k] for k in feature_names] for s in results['clean']])
        y = np.array([s['label'] for s in results['clean']])

        print(f"\nFeature matrix: {X.shape}")
        print(f"Label balance: {y.mean():.1%} positive")

        # Save
        split = 'train' if 'train' in args.input_dir else 'test'
        np.save(f"{args.output_dir}/{split}_features.npy", X)
        np.save(f"{args.output_dir}/{split}_labels.npy", y)

        with open(f"{args.output_dir}/feature_names.txt", 'w') as f:
            f.write('\n'.join(feature_names))

        print(f"Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
