"""
Simple labeling interface for identifying the shooter in free throw videos.

Usage:
    python label_shooter.py --data_dir data/sam3_extracted/train --output labels/shooter_clicks.json

Controls:
    - Click on the shooter to mark them
    - Press 's' to save and go to next video
    - Press 'n' to skip (no shooter visible)
    - Press 'r' to reset click
    - Press 'q' to quit and save progress
"""
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import argparse

# Global state for mouse callback
click_point = None
current_frame = None


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks."""
    global click_point, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"  Clicked: ({x}, {y})")


def load_frame_from_pkl(pkl_path, frame_idx=0):
    """Load a specific frame from the pkl's source video."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    video_path = data.get('video_path', '')
    frame_indices = data.get('frame_indices', [])
    frame_size = data.get('frame_size', (320, 240))

    # Try to find video
    video_file = None
    for base in ['data', '.', 'data/Basketball_51 dataset']:
        candidates = [
            Path(base) / video_path,
            Path(base) / Path(video_path).name,
            Path(video_path),
        ]
        for cand in candidates:
            if cand.exists():
                video_file = cand
                break
        if video_file:
            break

    if video_file is None:
        print(f"  Video not found: {video_path}")
        return None, data

    # Read frame
    cap = cv2.VideoCapture(str(video_file))
    if frame_idx < len(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[frame_idx])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, data

    return frame, data


def draw_overlay(frame, click_pt=None, data=None):
    """Draw overlay with click point and info."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Draw click point
    if click_pt:
        cv2.circle(display, click_pt, 10, (0, 255, 0), 2)
        cv2.circle(display, click_pt, 3, (0, 255, 0), -1)
        cv2.putText(display, "SHOOTER", (click_pt[0] + 15, click_pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw instructions
    instructions = [
        "Click on SHOOTER",
        "S=Save, N=Skip, R=Reset, Q=Quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(display, text, (10, 25 + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw label info
    if data:
        label = data.get('label', -1)
        label_text = f"Label: {'MAKE' if label == 1 else 'MISS'}"
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.putText(display, label_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return display


def label_videos(data_dir, output_path, start_from=0):
    """Main labeling loop."""
    global click_point, current_frame

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing labels
    if output_path.exists():
        with open(output_path, 'r') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} existing labels")
    else:
        labels = {}

    # Get pkl files
    pkl_files = sorted(data_dir.glob('*.pkl'))
    print(f"Found {len(pkl_files)} videos to label")

    # Skip already labeled
    remaining = [p for p in pkl_files if p.stem not in labels]
    print(f"Remaining to label: {len(remaining)}")

    if start_from > 0:
        remaining = remaining[start_from:]

    # Create window
    cv2.namedWindow('Label Shooter', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Label Shooter', mouse_callback)

    idx = 0
    while idx < len(remaining):
        pkl_path = remaining[idx]
        print(f"\n[{idx+1}/{len(remaining)}] {pkl_path.name}")

        # Load frame (use frame 5 - usually a good shooting pose frame)
        frame, data = load_frame_from_pkl(pkl_path, frame_idx=5)

        if frame is None:
            print("  Skipping - could not load frame")
            labels[pkl_path.stem] = {"status": "no_video"}
            idx += 1
            continue

        current_frame = frame
        click_point = None

        # Display loop
        while True:
            display = draw_overlay(frame, click_point, data)
            cv2.imshow('Label Shooter', display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('s'):  # Save and next
                if click_point:
                    labels[pkl_path.stem] = {
                        "status": "labeled",
                        "click_x": click_point[0],
                        "click_y": click_point[1],
                        "frame_w": frame.shape[1],
                        "frame_h": frame.shape[0],
                        "label": data.get('label', -1)
                    }
                    print(f"  Saved: {click_point}")
                    idx += 1
                    break
                else:
                    print("  Click on shooter first!")

            elif key == ord('n'):  # Skip
                labels[pkl_path.stem] = {"status": "skipped"}
                print("  Skipped")
                idx += 1
                break

            elif key == ord('r'):  # Reset
                click_point = None
                print("  Reset click")

            elif key == ord('q'):  # Quit
                print("\nQuitting and saving...")
                with open(output_path, 'w') as f:
                    json.dump(labels, f, indent=2)
                print(f"Saved {len(labels)} labels to {output_path}")
                cv2.destroyAllWindows()
                return labels

            elif key == ord('b'):  # Back
                if idx > 0:
                    idx -= 1
                    break

        # Auto-save every 10 labels
        if len(labels) % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(labels, f, indent=2)

    # Final save
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"\nDone! Saved {len(labels)} labels to {output_path}")

    cv2.destroyAllWindows()
    return labels


def main():
    parser = argparse.ArgumentParser(description="Label shooter in free throw videos")
    parser.add_argument('--data_dir', type=str, default='data/sam3_extracted/sam3_extracted/train',
                       help='Directory with pkl files')
    parser.add_argument('--output', type=str, default='labels/shooter_clicks.json',
                       help='Output JSON file for labels')
    parser.add_argument('--start', type=int, default=0,
                       help='Start from this index (skip first N)')
    args = parser.parse_args()

    print("=" * 50)
    print("SHOOTER LABELING TOOL")
    print("=" * 50)
    print("Controls:")
    print("  Click    - Mark shooter location")
    print("  S        - Save and next")
    print("  N        - Skip (no clear shooter)")
    print("  R        - Reset click")
    print("  B        - Go back")
    print("  Q        - Quit and save")
    print("=" * 50)

    label_videos(args.data_dir, args.output, args.start)


if __name__ == '__main__':
    main()
