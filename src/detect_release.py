"""
Detect ball release frame from SAM3 + SAM 3D Body extracted data.
Release = moment ball leaves the shooter's hand.
"""
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# SAM 3D Body keypoint indices for wrists
# These are approximate - in joints_2d which has 70 keypoints
RIGHT_WRIST_IDX = 21
LEFT_WRIST_IDX = 20

def detect_release_frame(video_data, hand='right', distance_threshold=30, velocity_threshold=5):
    """
    Detect the release frame using ball-hand distance and velocity.

    Returns:
        release_idx: Index into frames_data of release frame (or None)
        release_info: Dict with detection details
    """
    frames_data = video_data.get('frames_data', [])
    frame_w, frame_h = video_data.get('frame_size', (320, 240))

    wrist_idx = RIGHT_WRIST_IDX if hand == 'right' else LEFT_WRIST_IDX

    # Collect ball positions and hand positions
    ball_positions = []
    hand_positions = []
    frame_indices = []

    for i, fd in enumerate(frames_data):
        ball_pos = fd.get('ball_centroid')
        pose_3d = fd.get('pose_3d')

        if ball_pos is not None and pose_3d is not None:
            joints_2d = pose_3d.get('joints_2d')
            if joints_2d is not None and len(joints_2d) > wrist_idx:
                hand_pos = joints_2d[wrist_idx]
                ball_positions.append(ball_pos)
                hand_positions.append(hand_pos)
                frame_indices.append(i)

    if len(ball_positions) < 3:
        return None, {'error': 'Not enough frames with ball and hand'}

    ball_positions = np.array(ball_positions)
    hand_positions = np.array(hand_positions)

    # Calculate ball-hand distances
    distances = np.linalg.norm(ball_positions - hand_positions, axis=1)

    # Calculate ball velocities
    velocities = np.linalg.norm(np.diff(ball_positions, axis=0), axis=1)
    velocities = np.concatenate([[0], velocities])

    # Detect release: distance increases AND velocity spikes
    release_candidates = []

    for i in range(1, len(distances)):
        dist_increasing = distances[i] > distances[i-1]
        dist_above_threshold = distances[i] > distance_threshold
        vel_above_threshold = velocities[i] > velocity_threshold
        ball_above_hand = ball_positions[i, 1] < hand_positions[i, 1]

        if dist_increasing and dist_above_threshold and vel_above_threshold:
            score = distances[i] + velocities[i] * 2
            release_candidates.append((frame_indices[i], score, {
                'distance': float(distances[i]),
                'velocity': float(velocities[i]),
                'ball_pos': ball_positions[i].tolist(),
                'hand_pos': hand_positions[i].tolist(),
                'ball_above_hand': bool(ball_above_hand)
            }))

    if not release_candidates:
        for i, dist in enumerate(distances):
            if dist > distance_threshold:
                return frame_indices[i], {
                    'method': 'distance_only',
                    'distance': float(dist),
                    'velocity': float(velocities[i])
                }
        return None, {'error': 'No release detected'}

    release_candidates.sort(key=lambda x: (-x[1], x[0]))
    best = release_candidates[0]

    return best[0], {'method': 'distance_velocity', **best[2]}


def process_all_extracted(data_dir='data/sam3_extracted'):
    """Add release detection to all extracted data files."""
    data_dir = Path(data_dir)

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        pkl_files = list(split_dir.glob('*.pkl'))
        print(f'Processing {split}: {len(pkl_files)} files...')

        release_stats = {'detected': 0, 'not_detected': 0}

        for pkl_path in tqdm(pkl_files):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            release_idx, release_info = detect_release_frame(data)

            data['release_frame_idx'] = release_idx
            data['release_info'] = release_info

            if release_idx is not None:
                release_stats['detected'] += 1
                for i, fd in enumerate(data['frames_data']):
                    fd['is_release'] = (i == release_idx)
            else:
                release_stats['not_detected'] += 1

            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)

        print(f'  Release detected: {release_stats["detected"]}/{len(pkl_files)}')
        print(f'  Not detected: {release_stats["not_detected"]}/{len(pkl_files)}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        process_all_extracted()
    else:
        # Test on one file
        test_files = list(Path('data/sam3_extracted_test/train').glob('*.pkl'))

        if test_files:
            print(f'Testing release detection on {test_files[0].name}...')

            with open(test_files[0], 'rb') as f:
                data = pickle.load(f)

            release_idx, release_info = detect_release_frame(data)

            print(f'Release frame index: {release_idx}')
            print(f'Release info: {release_info}')

            if release_idx is not None:
                fd = data['frames_data'][release_idx]
                print(f'Actual frame in video: {fd["frame_idx"]}')
                print(f'Ball centroid: {fd["ball_centroid"]}')
