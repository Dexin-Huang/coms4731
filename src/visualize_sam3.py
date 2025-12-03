"""
Visualize SAM3 detections overlaid on video frames.
Draws ball detection (centroid) and skeleton joints.
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
import argparse

# SMPL joint indices for skeleton drawing
SMPL_SKELETON = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # pelvis -> head
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # spine -> left hand
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),  # spine -> right hand
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),  # pelvis -> left foot
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),  # pelvis -> right foot
]

JOINT_NAMES = {
    0: 'pelvis', 1: 'l_hip', 2: 'r_hip', 3: 'spine1', 4: 'l_knee', 5: 'r_knee',
    6: 'spine2', 7: 'l_ankle', 8: 'r_ankle', 9: 'spine3', 12: 'neck',
    15: 'head', 16: 'l_shoulder', 17: 'r_shoulder', 18: 'l_elbow', 19: 'r_elbow',
    20: 'l_wrist', 21: 'r_wrist', 22: 'l_hand', 23: 'r_hand',
}


def project_3d_to_2d(joints_3d, frame_w, frame_h):
    """
    Simple orthographic projection of 3D joints to 2D.
    Assumes joints are in normalized coordinates centered around origin.
    """
    if joints_3d is None or len(joints_3d) == 0:
        return None

    joints = np.array(joints_3d)

    # Use x and y coordinates, scale to frame size
    # The 3D coordinates seem to be normalized around 0
    # We need to map them to pixel coordinates

    # Find bounds
    x_coords = joints[:, 0]
    y_coords = joints[:, 1]

    # Normalize to 0-1 range
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Add padding
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)

    # Map to frame coordinates (center in frame)
    joints_2d = []
    for j in joints:
        # Normalize
        x_norm = (j[0] - x_min) / x_range
        y_norm = (j[1] - y_min) / y_range

        # Map to frame (with padding)
        margin = 0.1
        px = int((margin + x_norm * (1 - 2*margin)) * frame_w)
        py = int((margin + y_norm * (1 - 2*margin)) * frame_h)

        joints_2d.append((px, py))

    return joints_2d


def draw_skeleton(frame, joints_2d, color=(0, 255, 0)):
    """Draw skeleton on frame."""
    if joints_2d is None:
        return frame

    # Draw bones
    for i, j in SMPL_SKELETON:
        if i < len(joints_2d) and j < len(joints_2d):
            pt1 = joints_2d[i]
            pt2 = joints_2d[j]
            cv2.line(frame, pt1, pt2, color, 2)

    # Draw joints
    for idx, pt in enumerate(joints_2d[:24]):  # Only first 24 body joints
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)  # Red dots

        # Label key joints
        if idx in JOINT_NAMES:
            cv2.putText(frame, JOINT_NAMES[idx], (pt[0]+5, pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return frame


def draw_ball(frame, ball_pos, is_release=False):
    """Draw ball position on frame."""
    if ball_pos is None:
        return frame

    x, y = int(ball_pos[0]), int(ball_pos[1])

    # Ball circle
    color = (0, 255, 255) if is_release else (255, 165, 0)  # Yellow if release, orange otherwise
    cv2.circle(frame, (x, y), 15, color, 3)
    cv2.circle(frame, (x, y), 3, color, -1)  # Center dot

    label = "RELEASE" if is_release else "ball"
    cv2.putText(frame, label, (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def visualize_pkl(pkl_path, video_base_dir, output_dir, max_frames=None):
    """Create visualization for a single pkl file."""
    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pkl
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    video_path_str = data.get('video_path', '')
    frame_size = data.get('frame_size', (320, 240))
    frames_data = data.get('frames_data', [])
    frame_indices = data.get('frame_indices', [])
    release_idx = data.get('release_frame_idx', -1)
    label = data.get('label', -1)

    print(f"PKL: {pkl_path.name}")
    print(f"  Video: {video_path_str}")
    print(f"  Frame size: {frame_size}")
    print(f"  Frames: {len(frames_data)}")
    print(f"  Release frame: {release_idx}")
    print(f"  Label: {label} ({'make' if label == 1 else 'miss'})")

    # Try to find video
    video_path = None
    for base in [video_base_dir, Path(video_base_dir) / "Basketball_51 dataset"]:
        # Try different path constructions
        candidates = [
            Path(base) / video_path_str,
            Path(base) / Path(video_path_str).name,
            Path(video_path_str),
        ]
        for cand in candidates:
            if cand.exists():
                video_path = cand
                break
        if video_path:
            break

    if video_path is None or not video_path.exists():
        print(f"  WARNING: Video not found, creating blank frames")
        video_path = None

    # Open video if found
    cap = None
    if video_path:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  WARNING: Could not open video")
            cap = None

    frame_w, frame_h = frame_size

    # Process frames
    output_frames = []
    n_frames = len(frames_data) if max_frames is None else min(len(frames_data), max_frames)

    for i in range(n_frames):
        frame_data = frames_data[i]
        frame_idx = frame_indices[i] if i < len(frame_indices) else i

        # Get video frame
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        else:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Resize if needed
        if frame.shape[:2] != (frame_h, frame_w):
            frame = cv2.resize(frame, (frame_w, frame_h))

        # Draw ball
        ball_pos = frame_data.get('ball_centroid')
        is_release = frame_data.get('is_release', False)
        frame = draw_ball(frame, ball_pos, is_release)

        # Draw skeleton
        pose_3d = frame_data.get('pose_3d', {})
        joints_2d_raw = pose_3d.get('joints_2d') if pose_3d else None

        if joints_2d_raw and len(joints_2d_raw) > 0:
            # joints_2d should already be in pixel coordinates
            joints_2d = [(int(j[0]), int(j[1])) for j in joints_2d_raw[:24]]
            frame = draw_skeleton(frame, joints_2d)
        elif pose_3d:
            # Try projecting 3D joints
            joints_3d = pose_3d.get('joints_3d')
            if joints_3d:
                joints_2d = project_3d_to_2d(joints_3d, frame_w, frame_h)
                frame = draw_skeleton(frame, joints_2d, color=(255, 0, 255))  # Magenta for projected

        # Add frame info
        info_text = f"Frame {i}/{len(frames_data)-1} (idx:{frame_idx})"
        if is_release:
            info_text += " [RELEASE]"
        cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        label_text = f"Label: {'MAKE' if label == 1 else 'MISS'}"
        cv2.putText(frame, label_text, (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0) if label == 1 else (0, 0, 255), 2)

        output_frames.append(frame)

        # Save individual frame
        frame_path = output_dir / f"{pkl_path.stem}_frame{i:02d}.png"
        cv2.imwrite(str(frame_path), frame)

    if cap:
        cap.release()

    # Create composite image (grid of all frames)
    if output_frames:
        # Arrange in 4x4 grid
        n_cols = 4
        n_rows = (len(output_frames) + n_cols - 1) // n_cols

        composite = np.zeros((n_rows * frame_h, n_cols * frame_w, 3), dtype=np.uint8)
        for i, frame in enumerate(output_frames):
            row = i // n_cols
            col = i % n_cols
            composite[row*frame_h:(row+1)*frame_h, col*frame_w:(col+1)*frame_w] = frame

        composite_path = output_dir / f"{pkl_path.stem}_composite.png"
        cv2.imwrite(str(composite_path), composite)
        print(f"  Saved composite: {composite_path}")

    return output_frames


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3 detections")
    parser.add_argument('--pkl_dir', type=str, default='data/sam3_extracted/train',
                       help='Directory with pkl files')
    parser.add_argument('--video_dir', type=str, default='data',
                       help='Base directory for videos')
    parser.add_argument('--output_dir', type=str, default='visualizations/sam3',
                       help='Output directory for visualizations')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--sample_makes', type=int, default=None,
                       help='Number of makes to sample')
    parser.add_argument('--sample_misses', type=int, default=None,
                       help='Number of misses to sample')
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pkl files
    pkl_files = sorted(pkl_dir.glob('*.pkl'))
    print(f"Found {len(pkl_files)} pkl files in {pkl_dir}")

    # Sample some files
    if args.sample_makes or args.sample_misses:
        makes = []
        misses = []
        for pkl_path in pkl_files:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if data.get('label', 0) == 1:
                makes.append(pkl_path)
            else:
                misses.append(pkl_path)

        selected = []
        if args.sample_misses:
            selected.extend(misses[:args.sample_misses])
        if args.sample_makes:
            selected.extend(makes[:args.sample_makes])
        pkl_files = selected
    else:
        pkl_files = pkl_files[:args.n_samples]

    print(f"Visualizing {len(pkl_files)} samples...")

    for pkl_path in pkl_files:
        visualize_pkl(pkl_path, args.video_dir, output_dir)
        print()


if __name__ == '__main__':
    main()
