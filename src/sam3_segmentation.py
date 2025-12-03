"""
SAM3 Video Segmentation for Basketball Free Throw Analysis.
Segments basketball and player using text prompts.
"""
import sys
sys.path.insert(0, '/workspace/sam3')

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


class SAM3VideoSegmenter:
    def __init__(self):
        print("Loading SAM3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        print("Model loaded!")

    def segment_frame(self, frame_pil, ball_prompt="ball", player_prompt="basketball player"):
        """Segment a single frame."""
        state = self.processor.set_image(frame_pil)

        # Segment ball
        ball_output = self.processor.set_text_prompt(state=state, prompt=ball_prompt)
        ball_masks = ball_output["masks"]
        ball_scores = ball_output["scores"]

        # Segment player
        player_output = self.processor.set_text_prompt(state=state, prompt=player_prompt)
        player_masks = player_output["masks"]
        player_scores = player_output["scores"]

        return {
            "ball_masks": ball_masks,
            "ball_scores": ball_scores,
            "player_masks": player_masks,
            "player_scores": player_scores
        }

    def segment_video(self, video_path, n_frames=16):
        """Segment all frames in a video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        results = {
            "ball_centroids": [],
            "ball_areas": [],
            "ball_detected": [],
            "player_masks": []
        }

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                results["ball_centroids"].append((0, 0))
                results["ball_areas"].append(0)
                results["ball_detected"].append(False)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            seg_result = self.segment_frame(pil_frame)

            # Process ball mask
            if len(seg_result["ball_masks"]) > 0 and seg_result["ball_scores"][0] > 0.3:
                mask = seg_result["ball_masks"][0]
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                # Ensure mask is 2D
                if mask.ndim > 2:
                    mask = mask.squeeze()
                if mask.ndim > 2:
                    mask = mask[0]  # Take first channel if still multi-dim
                mask = (mask > 0.5).astype(np.uint8)

                # Compute centroid
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    results["ball_centroids"].append((cx, cy))
                    results["ball_areas"].append(M["m00"])
                    results["ball_detected"].append(True)
                else:
                    results["ball_centroids"].append((0, 0))
                    results["ball_areas"].append(0)
                    results["ball_detected"].append(False)
            else:
                results["ball_centroids"].append((0, 0))
                results["ball_areas"].append(0)
                results["ball_detected"].append(False)

            # Store best player mask
            if len(seg_result["player_masks"]) > 0:
                mask = seg_result["player_masks"][0]
                if torch.is_tensor(mask):
                    mask = mask.cpu().numpy()
                results["player_masks"].append(mask)
            else:
                results["player_masks"].append(None)

        cap.release()
        return results


def compute_trajectory_features(centroids, detected, frame_size=(320, 240), fps=30):
    """Compute physics features from ball trajectory."""
    features = {}

    # Filter valid detections
    valid_points = [(x, y) for (x, y), d in zip(centroids, detected) if d]

    if len(valid_points) < 3:
        # Return empty features
        return {f"traj_{i}": 0.0 for i in range(20)}

    trajectory = np.array(valid_points)

    # Normalize by frame size
    trajectory[:, 0] /= frame_size[0]
    trajectory[:, 1] /= frame_size[1]

    features["traj_length"] = len(trajectory)
    features["traj_x_range"] = np.ptp(trajectory[:, 0])
    features["traj_y_range"] = np.ptp(trajectory[:, 1])

    # Release point
    features["release_x"] = trajectory[0, 0]
    features["release_y"] = trajectory[0, 1]

    # Arc apex
    min_y_idx = np.argmin(trajectory[:, 1])
    features["apex_x"] = trajectory[min_y_idx, 0]
    features["apex_y"] = trajectory[min_y_idx, 1]
    features["arc_height"] = features["release_y"] - features["apex_y"]

    # Velocities
    dt = 1.0 / fps
    velocities = np.diff(trajectory, axis=0) / dt

    if len(velocities) > 0:
        features["release_vx"] = velocities[0, 0]
        features["release_vy"] = velocities[0, 1]
        features["release_speed"] = np.linalg.norm(velocities[0])

        if velocities[0, 0] != 0:
            features["release_angle"] = np.degrees(np.arctan2(-velocities[0, 1], velocities[0, 0]))
        else:
            features["release_angle"] = 90.0

        speeds = np.linalg.norm(velocities, axis=1)
        features["avg_speed"] = np.mean(speeds)
        features["max_speed"] = np.max(speeds)
        features["speed_std"] = np.std(speeds)
    else:
        features["release_vx"] = 0
        features["release_vy"] = 0
        features["release_speed"] = 0
        features["release_angle"] = 0
        features["avg_speed"] = 0
        features["max_speed"] = 0
        features["speed_std"] = 0

    # Parabola fit
    if len(trajectory) >= 3:
        try:
            coeffs = np.polyfit(trajectory[:, 0], trajectory[:, 1], 2)
            features["parabola_a"] = coeffs[0]
            features["parabola_b"] = coeffs[1]
            features["parabola_c"] = coeffs[2]
            y_pred = np.polyval(coeffs, trajectory[:, 0])
            features["parabola_residual"] = np.mean((trajectory[:, 1] - y_pred) ** 2)
        except:
            features["parabola_a"] = 0
            features["parabola_b"] = 0
            features["parabola_c"] = 0
            features["parabola_residual"] = 1.0
    else:
        features["parabola_a"] = 0
        features["parabola_b"] = 0
        features["parabola_c"] = 0
        features["parabola_residual"] = 1.0

    # Entry angle
    if len(velocities) >= 3:
        final_v = np.mean(velocities[-3:], axis=0)
        if final_v[0] != 0:
            features["entry_angle"] = np.degrees(np.arctan2(-final_v[1], final_v[0]))
        else:
            features["entry_angle"] = -90.0
    else:
        features["entry_angle"] = 0

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="data/sam3_features")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Limit videos for testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = SAM3VideoSegmenter()

    for split in ["train", "val", "test"]:
        split_csv = Path(args.splits_dir) / f"{split}.csv"
        if not split_csv.exists():
            print(f"Skipping {split} - no CSV found")
            continue

        df = pd.read_csv(split_csv)
        if args.limit:
            df = df.head(args.limit)

        print(f"\nProcessing {split} ({len(df)} videos)...")

        all_trajectories = []
        all_labels = []
        video_ids = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Handle path - CSV may already include 'data/' prefix
            video_rel_path = row["video_path"]
            if video_rel_path.startswith("data/"):
                video_path = Path(video_rel_path)
            else:
                video_path = Path(args.data_dir) / video_rel_path
            if not video_path.exists():
                continue

            # Segment video
            results = segmenter.segment_video(video_path, n_frames=args.n_frames)

            # Compute trajectory features
            features = compute_trajectory_features(
                results["ball_centroids"],
                results["ball_detected"]
            )

            all_trajectories.append(list(features.values()))
            all_labels.append(row["label"])
            video_ids.append(row["video_path"])

        # Save
        trajectories = np.array(all_trajectories, dtype=np.float32)
        labels = np.array(all_labels)

        np.save(output_dir / f"{split}_sam3_trajectories.npy", trajectories)
        np.save(output_dir / f"{split}_labels.npy", labels)

        with open(output_dir / f"{split}_video_ids.txt", "w") as f:
            for vid in video_ids:
                f.write(f"{vid}\n")

        print(f"  Saved {len(trajectories)} trajectories")
        print(f"  Trajectory shape: {trajectories.shape}")


if __name__ == "__main__":
    main()
