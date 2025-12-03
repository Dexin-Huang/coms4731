"""
Kalman Filter for Basketball Trajectory Smoothing.

Research shows we detect ball in only ~50% of frames.
Kalman filter:
1. Predicts ball position in missing frames using physics model
2. Smooths noisy detections
3. Estimates velocity and acceleration

Physics model: projectile motion under gravity
State: [x, y, vx, vy]
Transition: constant velocity between frames (simple model)
"""
import numpy as np
from typing import List, Tuple, Optional


class BallTrajectoryKalman:
    """Kalman filter for basketball trajectory with projectile motion physics."""

    def __init__(self, dt: float = 1/30, process_noise: float = 0.1,
                 measurement_noise: float = 5.0, gravity: float = 9.8):
        """
        Initialize Kalman filter.

        Args:
            dt: Time between frames (1/fps)
            process_noise: Process noise (higher = more trust in measurements)
            measurement_noise: Measurement noise (pixel std dev)
            gravity: Gravity constant (pixels/frame^2, adjusted for image coords)
        """
        self.dt = dt
        self.gravity = gravity

        # State dimension: [x, y, vx, vy]
        self.n_state = 4
        self.n_meas = 2  # We only observe x, y

        # State transition matrix (constant velocity + gravity)
        # x' = x + vx*dt
        # y' = y + vy*dt + 0.5*g*dt^2  (gravity pulls down = positive y)
        # vx' = vx
        # vy' = vy + g*dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # Control input for gravity (applied to y and vy)
        self.B = np.array([
            [0],
            [0.5 * gravity * dt**2],
            [0],
            [gravity * dt]
        ], dtype=np.float64)

        # Measurement matrix (observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*2, 0],
            [0, 0, 0, q*2]
        ], dtype=np.float64)

        # Measurement noise covariance
        r = measurement_noise ** 2
        self.R = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float64)

        # Initial state and covariance
        self.x = None
        self.P = None

    def reset(self, initial_pos: Optional[Tuple[float, float]] = None,
              initial_vel: Tuple[float, float] = (0, 0)):
        """Reset filter with optional initial position."""
        if initial_pos is not None:
            self.x = np.array([
                initial_pos[0], initial_pos[1],
                initial_vel[0], initial_vel[1]
            ], dtype=np.float64)
        else:
            self.x = None

        # Large initial covariance (uncertain)
        self.P = np.eye(self.n_state) * 1000

    def predict(self):
        """Predict next state using physics model."""
        if self.x is None:
            return None

        # State prediction: x = F*x + B*u (u=1 for gravity)
        self.x = self.F @ self.x + self.B.flatten()

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:2].copy()

    def update(self, measurement: Tuple[float, float]):
        """Update state with new measurement."""
        z = np.array(measurement, dtype=np.float64)

        if self.x is None:
            # Initialize with first measurement
            self.x = np.array([z[0], z[1], 0, 0], dtype=np.float64)
            return self.x[:2].copy()

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I = np.eye(self.n_state)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2].copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state (position) and velocity."""
        if self.x is None:
            return None, None
        return self.x[:2].copy(), self.x[2:4].copy()


def smooth_trajectory(positions: List[Optional[Tuple[float, float]]],
                      dt: float = 1/30,
                      process_noise: float = 0.5,
                      measurement_noise: float = 10.0,
                      use_gravity: bool = False) -> np.ndarray:
    """
    Smooth a trajectory with missing values using Kalman filter.

    Args:
        positions: List of (x, y) tuples or None for missing frames
        dt: Time between frames
        process_noise: Higher = trust measurements more
        measurement_noise: Expected pixel noise in detections
        use_gravity: Whether to include gravity in physics model

    Returns:
        smoothed: (N, 2) array of smoothed positions
    """
    n_frames = len(positions)
    smoothed = np.zeros((n_frames, 2), dtype=np.float64)

    # Don't use gravity for now (works better without for free throws)
    gravity = 0.0 if not use_gravity else 9.8 * dt * dt * 30

    kf = BallTrajectoryKalman(
        dt=dt,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        gravity=gravity
    )
    kf.reset()

    # Forward pass
    for i, pos in enumerate(positions):
        if pos is not None:
            # Have measurement - predict then update
            kf.predict()
            smoothed[i] = kf.update(pos)
        else:
            # No measurement - just predict
            pred = kf.predict()
            if pred is not None:
                smoothed[i] = pred
            else:
                # No prior state, interpolate later
                smoothed[i] = np.nan

    # Fill any remaining NaN values with linear interpolation
    for dim in range(2):
        mask = np.isnan(smoothed[:, dim])
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            smoothed[:, dim] = np.interp(
                np.arange(n_frames),
                valid_idx,
                smoothed[valid_idx, dim]
            )

    return smoothed


def compute_velocity(trajectory: np.ndarray, dt: float = 1/30) -> np.ndarray:
    """
    Compute velocity from trajectory.

    Args:
        trajectory: (N, 2) positions
        dt: Time between frames

    Returns:
        velocity: (N, 2) velocity at each frame
    """
    velocity = np.zeros_like(trajectory)

    # Central difference for interior points
    velocity[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2 * dt)

    # Forward/backward difference for endpoints
    velocity[0] = (trajectory[1] - trajectory[0]) / dt
    velocity[-1] = (trajectory[-1] - trajectory[-2]) / dt

    return velocity


def compute_acceleration(trajectory: np.ndarray, dt: float = 1/30) -> np.ndarray:
    """
    Compute acceleration from trajectory.

    Args:
        trajectory: (N, 2) positions
        dt: Time between frames

    Returns:
        acceleration: (N, 2) acceleration at each frame
    """
    acceleration = np.zeros_like(trajectory)

    # Second derivative using central difference
    acceleration[1:-1] = (trajectory[2:] - 2*trajectory[1:-1] + trajectory[:-2]) / (dt**2)

    # Extend to endpoints
    if len(trajectory) > 2:
        acceleration[0] = acceleration[1]
        acceleration[-1] = acceleration[-2]

    return acceleration


def extract_trajectory_features_kalman(
    raw_positions: List[Optional[Tuple[float, float]]],
    hoop_position: Tuple[float, float],
    frame_size: Tuple[int, int],
    dt: float = 1/30
) -> dict:
    """
    Extract trajectory features using Kalman-smoothed trajectory.

    Args:
        raw_positions: List of (x, y) or None for each frame
        hoop_position: (x, y) of hoop
        frame_size: (width, height) for normalization
        dt: Time between frames

    Returns:
        Dictionary of trajectory features
    """
    frame_w, frame_h = frame_size
    hoop_x, hoop_y = hoop_position

    # Smooth trajectory
    smoothed = smooth_trajectory(raw_positions, dt=dt)

    # Normalize by frame size
    smoothed_norm = smoothed.copy()
    smoothed_norm[:, 0] /= frame_w
    smoothed_norm[:, 1] /= frame_h
    hoop_norm = np.array([hoop_x / frame_w, hoop_y / frame_h])

    # Compute velocity and acceleration
    velocity = compute_velocity(smoothed_norm, dt=1.0)  # dt=1 for per-frame
    acceleration = compute_acceleration(smoothed_norm, dt=1.0)

    features = {}

    # Ball-to-hoop distances over time
    distances = np.linalg.norm(smoothed_norm - hoop_norm, axis=1)
    features['ball_hoop_dist_start'] = float(distances[0])
    features['ball_hoop_dist_end'] = float(distances[-1])
    features['ball_hoop_dist_min'] = float(distances.min())
    features['ball_hoop_dist_mean'] = float(distances.mean())

    # Approaching hoop?
    features['approaching_hoop'] = float(distances[-1] < distances[0])

    # Trajectory direction to hoop
    traj_vec = smoothed_norm[-1] - smoothed_norm[0]
    to_hoop_vec = hoop_norm - smoothed_norm[0]

    dot = np.dot(traj_vec, to_hoop_vec)
    mag_traj = np.linalg.norm(traj_vec)
    mag_hoop = np.linalg.norm(to_hoop_vec)

    if mag_traj > 0 and mag_hoop > 0:
        cos_angle = np.clip(dot / (mag_traj * mag_hoop), -1, 1)
        features['traj_hoop_angle'] = float(np.degrees(np.arccos(cos_angle)))
    else:
        features['traj_hoop_angle'] = 90.0

    features['traj_x'] = float(traj_vec[0])
    features['traj_y'] = float(traj_vec[1])

    # Final segment (last 3 frames)
    final_seg = smoothed_norm[-3:]
    final_vec = final_seg[-1] - final_seg[0]
    features['final_vel_x'] = float(final_vec[0])
    features['final_vel_y'] = float(final_vec[1])

    final_to_hoop = hoop_norm - final_seg[-1]
    features['final_hoop_dist_x'] = float(final_to_hoop[0])
    features['final_hoop_dist_y'] = float(final_to_hoop[1])

    # Arc height relative to hoop
    min_y = smoothed_norm[:, 1].min()  # Lowest y = highest point
    features['arc_above_hoop'] = float(min_y < hoop_norm[1])
    features['arc_height_vs_hoop'] = float(hoop_norm[1] - min_y)

    # NEW: Velocity features from Kalman smoothing
    features['mean_vel_x'] = float(velocity[:, 0].mean())
    features['mean_vel_y'] = float(velocity[:, 1].mean())
    features['max_vel'] = float(np.linalg.norm(velocity, axis=1).max())

    # NEW: Acceleration features
    features['mean_acc_y'] = float(acceleration[:, 1].mean())  # Should be ~gravity
    features['acc_std'] = float(acceleration.std())  # Smoothness

    # NEW: Trajectory smoothness (deviation from parabola)
    # Fit parabola to y vs x and compute residual
    x_coords = smoothed_norm[:, 0]
    y_coords = smoothed_norm[:, 1]
    if len(np.unique(x_coords)) >= 3:
        try:
            coeffs = np.polyfit(x_coords, y_coords, 2)
            y_fit = np.polyval(coeffs, x_coords)
            residual = np.sqrt(((y_coords - y_fit) ** 2).mean())
            features['parabola_residual'] = float(residual)
        except:
            features['parabola_residual'] = 0.1
    else:
        features['parabola_residual'] = 0.1

    # Detection quality
    n_valid = sum(1 for p in raw_positions if p is not None)
    features['ball_detection_rate'] = float(n_valid / len(raw_positions))

    return features


if __name__ == '__main__':
    # Test with synthetic trajectory
    print("Testing Kalman filter...")

    # Simulate a parabolic trajectory with missing frames
    n_frames = 16
    t = np.linspace(0, 1, n_frames)

    # True trajectory: parabola
    true_x = 50 + 200 * t  # Moving right
    true_y = 100 - 150 * t + 100 * t**2  # Arc

    # Add noise
    np.random.seed(42)
    noisy_x = true_x + np.random.randn(n_frames) * 5
    noisy_y = true_y + np.random.randn(n_frames) * 5

    # Remove 50% of observations randomly
    positions = []
    for i in range(n_frames):
        if np.random.random() > 0.5:
            positions.append((noisy_x[i], noisy_y[i]))
        else:
            positions.append(None)

    print(f"Simulated {n_frames} frames, {sum(p is not None for p in positions)} observed")

    # Smooth
    smoothed = smooth_trajectory(positions)

    # Compute error
    error = np.sqrt(((smoothed[:, 0] - true_x)**2 + (smoothed[:, 1] - true_y)**2).mean())
    print(f"RMSE after Kalman smoothing: {error:.2f} pixels")

    # Extract features
    features = extract_trajectory_features_kalman(
        positions,
        hoop_position=(250, 80),
        frame_size=(320, 240)
    )
    print(f"\nExtracted {len(features)} features:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")
