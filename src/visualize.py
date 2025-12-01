"""
Visualization utilities for Basketball Free Throw Prediction.

Provides:
- GradCAM visualization for model interpretability
- Attention map visualization
- Sample predictions with overlays
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for video models.

    Shows which spatial-temporal regions the model focuses on for predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        Args:
            model: Trained video classification model
            target_layer: Name of layer to visualize (default: last conv layer)
        """
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Find target layer
        if target_layer is None:
            # For Video Swin, use the last stage
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = dict(model.named_modules())[target_layer]

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _find_target_layer(self):
        """Find the last convolutional layer."""
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv3d, torch.nn.Conv2d)):
                target = module
        return target

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Video tensor (1, C, T, H, W)
            target_class: Class to visualize (default: predicted class)

        Returns:
            Heatmap array of shape (T, H, W)
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (1, C, T', H', W')
        activations = self.activations  # (1, C, T', H', W')

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)

        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, T', H', W')
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()  # (T', H', W')
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize_on_video(
        self,
        video_frames: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay GradCAM heatmap on video frames.

        Args:
            video_frames: Original frames (T, H, W, C)
            cam: GradCAM heatmap (T', H', W')
            alpha: Overlay transparency

        Returns:
            Overlaid frames (T, H, W, C)
        """
        T, H, W, C = video_frames.shape
        T_cam, H_cam, W_cam = cam.shape

        overlaid = []
        for t in range(T):
            # Get corresponding CAM frame
            t_cam = int(t * T_cam / T)
            heatmap = cam[t_cam]

            # Resize heatmap to frame size
            heatmap = cv2.resize(heatmap, (W, H))

            # Convert to colormap
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Overlay
            frame = video_frames[t].astype(np.float32)
            overlay = (1 - alpha) * frame + alpha * heatmap_colored.astype(np.float32)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            overlaid.append(overlay)

        return np.stack(overlaid)


def plot_gradcam_grid(
    frames: np.ndarray,
    cam_frames: np.ndarray,
    prediction: str,
    confidence: float,
    true_label: str,
    save_path: Optional[str] = None,
    n_frames: int = 8
) -> plt.Figure:
    """
    Plot a grid of frames with GradCAM overlay.

    Args:
        frames: Original video frames (T, H, W, C)
        cam_frames: Frames with GradCAM overlay (T, H, W, C)
        prediction: Predicted class name
        confidence: Prediction confidence
        true_label: Ground truth label
        save_path: Path to save figure
        n_frames: Number of frames to display

    Returns:
        matplotlib Figure
    """
    T = frames.shape[0]
    indices = np.linspace(0, T-1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, n_frames, figsize=(2*n_frames, 5))

    for i, idx in enumerate(indices):
        # Original frame
        axes[0, i].imshow(frames[idx])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Frame {idx}', fontsize=8)

        # GradCAM overlay
        axes[1, i].imshow(cam_frames[idx])
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('GradCAM', fontsize=10)

    # Title with prediction info
    color = 'green' if prediction == true_label else 'red'
    fig.suptitle(
        f'Prediction: {prediction} ({confidence:.1%}) | True: {true_label}',
        fontsize=12,
        color=color
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved GradCAM visualization to: {save_path}")

    return fig


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    output_dir: str,
    num_samples: int = 10,
    device: str = 'cuda'
):
    """
    Generate GradCAM visualizations for sample predictions.

    Args:
        model: Trained model
        dataloader: DataLoader with video data
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        device: Device to run on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    class_names = ['Miss', 'Make']

    samples_done = 0

    for batch_idx, (videos, labels) in enumerate(dataloader):
        if samples_done >= num_samples:
            break

        videos = videos.to(device)

        for i in range(videos.shape[0]):
            if samples_done >= num_samples:
                break

            video = videos[i:i+1]
            label = labels[i].item()

            # Get prediction
            with torch.no_grad():
                output = model(video)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                confidence = probs[0, pred].item()

            # Generate GradCAM
            cam = gradcam.generate(video, target_class=pred)

            # Convert video to numpy for visualization
            # Denormalize
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).to(device)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).to(device)
            video_denorm = video * std + mean
            video_denorm = video_denorm.clamp(0, 1)

            # (C, T, H, W) -> (T, H, W, C)
            frames = video_denorm[0].permute(1, 2, 3, 0).cpu().numpy()
            frames = (frames * 255).astype(np.uint8)

            # Overlay GradCAM
            cam_frames = gradcam.visualize_on_video(frames, cam)

            # Plot and save
            plot_gradcam_grid(
                frames=frames,
                cam_frames=cam_frames,
                prediction=class_names[pred],
                confidence=confidence,
                true_label=class_names[label],
                save_path=str(output_dir / f'gradcam_sample_{samples_done}.png')
            )

            samples_done += 1

    print(f"Generated {samples_done} GradCAM visualizations in {output_dir}")


def plot_temporal_attention(
    attention_weights: np.ndarray,
    frame_indices: List[int],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal attention weights showing which frames the model focuses on.

    Args:
        attention_weights: Attention weights (T,)
        frame_indices: Frame numbers
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(frame_indices, attention_weights, color='steelblue', alpha=0.7)
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('Temporal Attention Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark key moments
    peak_idx = np.argmax(attention_weights)
    ax.axvline(x=frame_indices[peak_idx], color='red', linestyle='--',
               label=f'Peak attention (frame {frame_indices[peak_idx]})')
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_prediction_video(
    frames: np.ndarray,
    cam_frames: np.ndarray,
    prediction: str,
    confidence: float,
    output_path: str,
    fps: int = 10
):
    """
    Create a side-by-side video showing original and GradCAM overlay.

    Args:
        frames: Original frames (T, H, W, C)
        cam_frames: GradCAM overlay frames (T, H, W, C)
        prediction: Predicted class
        confidence: Prediction confidence
        output_path: Path to save video
        fps: Frames per second
    """
    T, H, W, C = frames.shape

    # Create side-by-side frames
    combined_width = W * 2 + 10  # 10px gap
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, H + 40))

    for t in range(T):
        # Create combined frame
        combined = np.ones((H + 40, combined_width, 3), dtype=np.uint8) * 255

        # Original
        combined[40:, :W] = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)

        # GradCAM
        combined[40:, W+10:] = cv2.cvtColor(cam_frames[t], cv2.COLOR_RGB2BGR)

        # Add text
        cv2.putText(combined, 'Original', (W//2 - 30, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(combined, 'GradCAM', (W + 10 + W//2 - 30, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(combined, f'Pred: {prediction} ({confidence:.1%})', (10, H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(combined)

    out.release()
    print(f"Saved prediction video to: {output_path}")


if __name__ == '__main__':
    print("Visualization module loaded.")
    print("Use visualize_predictions() to generate GradCAM visualizations.")
    print("Use GradCAM class for custom visualizations.")
