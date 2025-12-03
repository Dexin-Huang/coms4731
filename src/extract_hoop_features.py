"""
Extract hoop-relative trajectory features.
Key insight: the spatial relationship between ball trajectory and hoop
is what determines make/miss.
"""
import sys
sys.path.insert(0, "/workspace/sam3")

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import pandas as pd

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def detect_hoop(processor, pil_frame, state=None):
    """Detect basketball hoop/net position."""
    if state is None:
        state = processor.set_image(pil_frame)

    # Try multiple prompts
    for prompt in ["net", "basketball hoop"]:
        output = processor.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"]
        scores = output["scores"]

        if len(masks) > 0 and scores[0] > 0.3:
            mask = masks[0]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            if mask.ndim > 2:
                mask = mask.squeeze()
            if mask.ndim > 2:
                mask = mask[0]
            mask = (mask > 0.5).astype(np.uint8)

            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return (cx, cy), float(scores[0])

    return None, 0.0


def detect_ball(processor, pil_frame, state=None):
    """Detect basketball position."""
    if state is None:
        state = processor.set_image(pil_frame)

    output = processor.set_text_prompt(state=state, prompt="ball")
    masks = output["masks"]
    scores = output["scores"]

    if len(masks) > 0 and scores[0] > 0.3:
        mask = masks[0]
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim > 2:
            mask = mask[0]
        mask = (mask > 0.5).astype(np.uint8)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return (cx, cy), float(scores[0])

    return None, 0.0


def extract_hoop_features(video_path, processor, n_frames=16):
    """Extract ball trajectory features relative to hoop."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    ball_positions = []
    hoop_positions = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            ball_positions.append(None)
            hoop_positions.append(None)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        state = processor.set_image(pil_frame)

        ball_pos, _ = detect_ball(processor, pil_frame, state)
        hoop_pos, _ = detect_hoop(processor, pil_frame, state)

        ball_positions.append(ball_pos)
        hoop_positions.append(hoop_pos)

    cap.release()

    # Compute hoop-relative features
    features = {}

    # Get best hoop estimate (average of detections)
    valid_hoop = [h for h in hoop_positions if h is not None]
    if valid_hoop:
        hoop_x = np.mean([h[0] for h in valid_hoop])
        hoop_y = np.mean([h[1] for h in valid_hoop])
    else:
        # Default hoop position (upper right)
        hoop_x = frame_w * 0.75
        hoop_y = frame_h * 0.25

    features["hoop_x"] = hoop_x / frame_w
    features["hoop_y"] = hoop_y / frame_h
    features["hoop_detected"] = float(len(valid_hoop) > 0)

    # Get valid ball trajectory
    valid_ball = [(i, b) for i, b in enumerate(ball_positions) if b is not None]

    if len(valid_ball) < 3:
        # Not enough data - return default features
        return get_default_features(features)

    ball_traj = np.array([b for _, b in valid_ball])

    # Normalize by frame size
    ball_traj_norm = ball_traj.copy()
    ball_traj_norm[:, 0] /= frame_w
    ball_traj_norm[:, 1] /= frame_h

    # Ball-to-hoop distances
    hoop_norm = np.array([hoop_x / frame_w, hoop_y / frame_h])
    distances = np.linalg.norm(ball_traj_norm - hoop_norm, axis=1)

    features["ball_hoop_dist_start"] = float(distances[0])
    features["ball_hoop_dist_end"] = float(distances[-1])
    features["ball_hoop_dist_min"] = float(distances.min())
    features["ball_hoop_dist_mean"] = float(distances.mean())

    # Is ball approaching hoop?
    features["approaching_hoop"] = float(distances[-1] < distances[0])

    # Trajectory direction relative to hoop
    if len(ball_traj_norm) >= 2:
        # Overall trajectory vector
        traj_vec = ball_traj_norm[-1] - ball_traj_norm[0]

        # Vector from start ball to hoop
        to_hoop_vec = hoop_norm - ball_traj_norm[0]

        # Angle between trajectory and hoop direction
        dot = np.dot(traj_vec, to_hoop_vec)
        mag_traj = np.linalg.norm(traj_vec)
        mag_hoop = np.linalg.norm(to_hoop_vec)

        if mag_traj > 0 and mag_hoop > 0:
            cos_angle = dot / (mag_traj * mag_hoop)
            cos_angle = np.clip(cos_angle, -1, 1)
            features["traj_hoop_angle"] = float(np.degrees(np.arccos(cos_angle)))
        else:
            features["traj_hoop_angle"] = 90.0

        features["traj_x"] = float(traj_vec[0])
        features["traj_y"] = float(traj_vec[1])
    else:
        features["traj_hoop_angle"] = 90.0
        features["traj_x"] = 0.0
        features["traj_y"] = 0.0

    # Final segment features (last 3 ball positions)
    if len(ball_traj_norm) >= 3:
        final_seg = ball_traj_norm[-3:]
        final_vec = final_seg[-1] - final_seg[0]

        features["final_vel_x"] = float(final_vec[0])
        features["final_vel_y"] = float(final_vec[1])

        # Final approach to hoop
        final_to_hoop = hoop_norm - final_seg[-1]
        features["final_hoop_dist_x"] = float(final_to_hoop[0])
        features["final_hoop_dist_y"] = float(final_to_hoop[1])
    else:
        features["final_vel_x"] = 0.0
        features["final_vel_y"] = 0.0
        features["final_hoop_dist_x"] = 0.0
        features["final_hoop_dist_y"] = 0.0

    # Arc height relative to hoop
    min_y = ball_traj_norm[:, 1].min()  # Lowest y = highest point in image coords
    features["arc_above_hoop"] = float(min_y < features["hoop_y"])
    features["arc_height_vs_hoop"] = float(features["hoop_y"] - min_y)

    # Ball detection quality
    features["ball_detection_rate"] = float(len(valid_ball) / n_frames)

    return features


def get_default_features(base_features):
    """Return default features when trajectory is not detected.

    IMPORTANT: Feature order must match extract_hoop_features() exactly.
    """
    # Start with base_features (hoop_x, hoop_y, hoop_detected) to preserve order
    defaults = dict(base_features)  # Copy to avoid modifying original

    # Add remaining features in the EXACT same order as extract_hoop_features()
    defaults["ball_hoop_dist_start"] = 0.5
    defaults["ball_hoop_dist_end"] = 0.3
    defaults["ball_hoop_dist_min"] = 0.2
    defaults["ball_hoop_dist_mean"] = 0.3
    defaults["approaching_hoop"] = 1.0
    defaults["traj_hoop_angle"] = 45.0
    defaults["traj_x"] = 0.1
    defaults["traj_y"] = -0.1
    defaults["final_vel_x"] = 0.05
    defaults["final_vel_y"] = -0.05
    defaults["final_hoop_dist_x"] = 0.1
    defaults["final_hoop_dist_y"] = 0.0
    defaults["arc_above_hoop"] = 1.0
    defaults["arc_height_vs_hoop"] = 0.1
    defaults["ball_detection_rate"] = 0.0

    return defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="data/hoop_features")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("Loaded!")

    for split in ["train", "val", "test"]:
        split_csv = Path(args.splits_dir) / f"{split}.csv"
        if not split_csv.exists():
            print(f"Skipping {split} - no CSV found")
            continue

        df = pd.read_csv(split_csv)
        if args.limit:
            df = df.head(args.limit)

        print(f"\nProcessing {split} ({len(df)} videos)...")

        all_features = []
        all_labels = []
        feature_names = None

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            video_rel_path = row["video_path"]
            if video_rel_path.startswith("data/"):
                video_path = Path(video_rel_path)
            else:
                video_path = Path(args.data_dir) / video_rel_path

            if not video_path.exists():
                print(f"  Warning: {video_path} not found")
                continue

            features = extract_hoop_features(video_path, processor, args.n_frames)

            if feature_names is None:
                feature_names = list(features.keys())

            all_features.append(list(features.values()))
            all_labels.append(row["label"])

        # Save features
        features_arr = np.array(all_features, dtype=np.float32)
        labels_arr = np.array(all_labels)

        np.save(output_dir / f"{split}_hoop_features.npy", features_arr)
        np.save(output_dir / f"{split}_labels.npy", labels_arr)

        with open(output_dir / "feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")

        print(f"  Saved {len(features_arr)} samples with {len(feature_names)} features")
        print(f"  Feature shape: {features_arr.shape}")


if __name__ == "__main__":
    main()
