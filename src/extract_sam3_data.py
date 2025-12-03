"""
Extract SAM3 masks and SAM 3D Body poses for basketball free throw analysis.
Outputs:
- Ball masks (binary masks for ball detection)
- Player masks (binary masks for player/thrower)
- Ball centroids and trajectories
- 3D body pose from SAM 3D Body
- Labels
"""
import sys
sys.path.insert(0, '/workspace/sam3')
sys.path.insert(0, '/workspace/sam-3d-body')

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import pandas as pd
import pickle
import json

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# SAM 3D Body imports (conditional)
SAM3D_AVAILABLE = False
try:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    SAM3D_AVAILABLE = True
except ImportError:
    print("SAM 3D Body not available - will skip 3D pose extraction")


class SAM3Extractor:
    """Extract masks using SAM3."""

    def __init__(self):
        print("Loading SAM3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        print("SAM3 loaded!")

    def segment_frame(self, frame_pil):
        """Segment ball and player in a frame."""
        state = self.processor.set_image(frame_pil)

        # Segment ball
        ball_output = self.processor.set_text_prompt(state=state, prompt="ball")
        ball_masks = ball_output["masks"]
        ball_scores = ball_output["scores"]

        # Segment player/thrower
        player_output = self.processor.set_text_prompt(state=state, prompt="basketball player")
        player_masks = player_output["masks"]
        player_scores = player_output["scores"]

        return {
            "ball_masks": ball_masks,
            "ball_scores": ball_scores,
            "player_masks": player_masks,
            "player_scores": player_scores
        }


class SAM3DBodyExtractor:
    """Extract 3D body pose using SAM 3D Body."""

    def __init__(self, checkpoint_dir="/workspace/sam-3d-body/checkpoints/sam-3d-body-dinov3"):
        if not SAM3D_AVAILABLE:
            raise ImportError("SAM 3D Body not available")

        print("Loading SAM 3D Body model...")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model, self.model_cfg = load_sam_3d_body(
            str(self.checkpoint_dir / "model.ckpt"),
            device=self.device,
            mhr_path=str(self.checkpoint_dir / "assets" / "mhr_model.pt")
        )

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=None,  # We already have bboxes from SAM3
            human_segmentor=None,
            fov_estimator=None
        )
        print("SAM 3D Body loaded!")

    def extract_pose(self, frame, player_mask=None):
        """Extract 3D body pose from frame with optional mask prompt."""
        # Get bounding box from mask if provided
        bbox = None
        if player_mask is not None:
            if torch.is_tensor(player_mask):
                player_mask = player_mask.cpu().numpy()
            if player_mask.ndim > 2:
                player_mask = player_mask.squeeze()
            if player_mask.ndim > 2:
                player_mask = player_mask[0]

            # Get bounding box [x1, y1, x2, y2]
            ys, xs = np.where(player_mask > 0.5)
            if len(ys) > 0:
                bbox = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]])  # Shape: (1, 4)

        try:
            # Save frame to temp file (SAM 3D Body expects file path)
            temp_path = "/tmp/temp_frame.jpg"
            if isinstance(frame, np.ndarray):
                cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                frame.save(temp_path)

            # Run inference with pre-computed bbox
            outputs = self.estimator.process_one_image(
                temp_path,
                bboxes=bbox,
                bbox_thr=0.3,
                use_mask=False
            )

            if outputs and len(outputs) > 0:
                output = outputs[0]  # Take first person
                result = {
                    "joints_3d": output.get("pred_keypoints_3d"),
                    "joints_2d": output.get("pred_keypoints_2d"),
                    "pose_params": output.get("body_pose_params"),
                    "shape_params": output.get("shape_params"),
                    "global_orient": output.get("global_rot"),
                    "hand_pose_params": output.get("hand_pose_params"),
                    "joint_coords": output.get("pred_joint_coords"),
                    "valid": True
                }
            else:
                result = {"valid": False}

        except Exception as e:
            print(f"  Warning: SAM 3D Body failed: {e}")
            result = {"valid": False}

        return result


def process_video(video_path, sam3_extractor, sam3d_extractor, n_frames=16):
    """Process a single video and extract all data."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    results = {
        "video_path": str(video_path),
        "frame_size": (frame_w, frame_h),
        "n_frames": n_frames,
        "frame_indices": frame_indices.tolist(),
        "frames_data": []
    }

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        frame_data = {
            "frame_idx": int(idx),
            "ball_detected": False,
            "ball_centroid": None,
            "ball_area": 0,
            "ball_mask": None,
            "player_detected": False,
            "player_mask": None,
            "pose_3d": None
        }

        if not ret:
            results["frames_data"].append(frame_data)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # SAM3 segmentation
        seg_result = sam3_extractor.segment_frame(pil_frame)

        # Process ball mask
        if len(seg_result["ball_masks"]) > 0 and seg_result["ball_scores"][0] > 0.3:
            ball_mask = seg_result["ball_masks"][0]
            if torch.is_tensor(ball_mask):
                ball_mask = ball_mask.cpu().numpy()
            if ball_mask.ndim > 2:
                ball_mask = ball_mask.squeeze()
            if ball_mask.ndim > 2:
                ball_mask = ball_mask[0]
            ball_mask = (ball_mask > 0.5).astype(np.uint8)

            M = cv2.moments(ball_mask)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                frame_data["ball_detected"] = True
                frame_data["ball_centroid"] = (float(cx), float(cy))
                frame_data["ball_area"] = float(M["m00"])
                # Store compressed mask
                frame_data["ball_mask"] = cv2.resize(ball_mask, (80, 60), interpolation=cv2.INTER_NEAREST)

        # Process player mask
        player_mask = None
        if len(seg_result["player_masks"]) > 0 and seg_result["player_scores"][0] > 0.5:
            player_mask = seg_result["player_masks"][0]
            if torch.is_tensor(player_mask):
                player_mask = player_mask.cpu().numpy()
            if player_mask.ndim > 2:
                player_mask = player_mask.squeeze()
            if player_mask.ndim > 2:
                player_mask = player_mask[0]
            player_mask = (player_mask > 0.5).astype(np.uint8)

            frame_data["player_detected"] = True
            # Store compressed mask
            frame_data["player_mask"] = cv2.resize(player_mask, (80, 60), interpolation=cv2.INTER_NEAREST)

            # SAM 3D Body pose extraction
            pose_result = sam3d_extractor.extract_pose(frame_rgb, player_mask)
            if pose_result["valid"]:
                # Convert tensors to numpy for serialization
                pose_data = {}
                for k, v in pose_result.items():
                    if k == "valid":
                        continue
                    if v is not None:
                        if torch.is_tensor(v):
                            pose_data[k] = v.cpu().numpy().tolist()
                        elif isinstance(v, np.ndarray):
                            pose_data[k] = v.tolist()
                        else:
                            pose_data[k] = v
                frame_data["pose_3d"] = pose_data

        results["frames_data"].append(frame_data)

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--output_dir", type=str, default="data/sam3_extracted")
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_sam3d", action="store_true", help="Skip SAM 3D Body (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractors
    sam3_extractor = SAM3Extractor()

    if not args.skip_sam3d:
        try:
            sam3d_extractor = SAM3DBodyExtractor()
        except Exception as e:
            print(f"Warning: Could not load SAM 3D Body: {e}")
            print("Continuing without 3D pose extraction...")
            sam3d_extractor = None
    else:
        sam3d_extractor = None
        print("Skipping SAM 3D Body (--skip_sam3d)")

    for split in ["train", "val", "test"]:
        split_csv = Path(args.splits_dir) / f"{split}.csv"
        if not split_csv.exists():
            print(f"Skipping {split} - no CSV found")
            continue

        df = pd.read_csv(split_csv)
        if args.limit:
            df = df.head(args.limit)

        print(f"\nProcessing {split} ({len(df)} videos)...")

        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        all_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            video_rel_path = row["video_path"]
            if video_rel_path.startswith("data/"):
                video_path = Path(video_rel_path)
            else:
                video_path = Path(args.data_dir) / video_rel_path

            if not video_path.exists():
                print(f"  Warning: {video_path} not found")
                continue

            # Process video
            if sam3d_extractor:
                video_data = process_video(video_path, sam3_extractor, sam3d_extractor, args.n_frames)
            else:
                # Simplified processing without SAM 3D Body
                video_data = process_video_simple(video_path, sam3_extractor, args.n_frames)

            video_data["label"] = int(row["label"])
            video_data["video_id"] = video_rel_path

            all_data.append(video_data)

            # Save individual video data
            video_name = Path(video_rel_path).stem
            with open(split_output_dir / f"{video_name}.pkl", "wb") as f:
                pickle.dump(video_data, f)

        # Save summary
        summary = {
            "split": split,
            "n_videos": len(all_data),
            "n_frames_per_video": args.n_frames,
            "video_ids": [d["video_id"] for d in all_data],
            "labels": [d["label"] for d in all_data]
        }

        with open(output_dir / f"{split}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  Saved {len(all_data)} videos to {split_output_dir}")


def process_video_simple(video_path, sam3_extractor, n_frames=16):
    """Simplified processing without SAM 3D Body."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    results = {
        "video_path": str(video_path),
        "frame_size": (frame_w, frame_h),
        "n_frames": n_frames,
        "frame_indices": frame_indices.tolist(),
        "frames_data": []
    }

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        frame_data = {
            "frame_idx": int(idx),
            "ball_detected": False,
            "ball_centroid": None,
            "ball_area": 0,
            "ball_mask": None,
            "player_detected": False,
            "player_mask": None,
            "pose_3d": None
        }

        if not ret:
            results["frames_data"].append(frame_data)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        seg_result = sam3_extractor.segment_frame(pil_frame)

        # Process ball mask
        if len(seg_result["ball_masks"]) > 0 and seg_result["ball_scores"][0] > 0.3:
            ball_mask = seg_result["ball_masks"][0]
            if torch.is_tensor(ball_mask):
                ball_mask = ball_mask.cpu().numpy()
            if ball_mask.ndim > 2:
                ball_mask = ball_mask.squeeze()
            if ball_mask.ndim > 2:
                ball_mask = ball_mask[0]
            ball_mask = (ball_mask > 0.5).astype(np.uint8)

            M = cv2.moments(ball_mask)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                frame_data["ball_detected"] = True
                frame_data["ball_centroid"] = (float(cx), float(cy))
                frame_data["ball_area"] = float(M["m00"])
                frame_data["ball_mask"] = cv2.resize(ball_mask, (80, 60), interpolation=cv2.INTER_NEAREST)

        # Process player mask
        if len(seg_result["player_masks"]) > 0 and seg_result["player_scores"][0] > 0.5:
            player_mask = seg_result["player_masks"][0]
            if torch.is_tensor(player_mask):
                player_mask = player_mask.cpu().numpy()
            if player_mask.ndim > 2:
                player_mask = player_mask.squeeze()
            if player_mask.ndim > 2:
                player_mask = player_mask[0]
            player_mask = (player_mask > 0.5).astype(np.uint8)

            frame_data["player_detected"] = True
            frame_data["player_mask"] = cv2.resize(player_mask, (80, 60), interpolation=cv2.INTER_NEAREST)

        results["frames_data"].append(frame_data)

    cap.release()
    return results


if __name__ == "__main__":
    main()
