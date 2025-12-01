"""
Evaluation utilities for Basketball Free Throw Prediction.

Includes:
- Comprehensive metrics calculation
- Confusion matrix visualization
- ROC curve plotting
- Per-class analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from model import FreeThrowClassifier
from dataset import get_dataloaders


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, any]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with predictions, labels, probabilities, and metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            x, y = batch
            x = x.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of class 1 (make)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'metrics': metrics
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    }

    if y_prob is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auroc'] = 0.0

    # Per-class metrics
    metrics['miss_precision'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['miss_recall'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['make_precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['make_recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"{'Metric':<20} {'Value':>10}")
    print(f"{'-'*30}")
    print(f"{'Accuracy':<20} {metrics['accuracy']:>10.4f}")
    print(f"{'Precision':<20} {metrics['precision']:>10.4f}")
    print(f"{'Recall':<20} {metrics['recall']:>10.4f}")
    print(f"{'F1 Score':<20} {metrics['f1']:>10.4f}")
    print(f"{'Specificity':<20} {metrics['specificity']:>10.4f}")
    if 'auroc' in metrics:
        print(f"{'AUROC':<20} {metrics['auroc']:>10.4f}")
    print(f"{'-'*30}")
    print(f"\nPer-Class Metrics:")
    print(f"  Miss - Precision: {metrics['miss_precision']:.4f}, Recall: {metrics['miss_recall']:.4f}")
    print(f"  Make - Precision: {metrics['make_precision']:.4f}, Recall: {metrics['make_recall']:.4f}")
    print(f"{'='*50}\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with counts and percentages
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=['Miss', 'Make'],
        yticklabels=['Miss', 'Make'],
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Prediction Probability Distribution"
) -> plt.Figure:
    """
    Plot distribution of prediction probabilities by class.

    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate probabilities by true class
    miss_probs = y_prob[y_true == 0]
    make_probs = y_prob[y_true == 1]

    ax.hist(miss_probs, bins=50, alpha=0.6, label='True Miss', color='red')
    ax.hist(make_probs, bins=50, alpha=0.6, label='True Make', color='green')

    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')

    ax.set_xlabel('Predicted Probability of Make', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability distribution saved to: {save_path}")

    return fig


def generate_report(
    results: Dict[str, any],
    output_dir: str = 'results',
    model_name: str = 'model'
) -> str:
    """
    Generate comprehensive evaluation report with visualizations.

    Args:
        results: Output from evaluate_model()
        output_dir: Directory to save results
        model_name: Name for the model in reports

    Returns:
        Path to the report directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = results['labels']
    y_pred = results['predictions']
    y_prob = results['probabilities']
    metrics = results['metrics']

    # Print metrics
    print_metrics(metrics, title=f"Evaluation Results - {model_name}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / f'{model_name}_metrics.csv', index=False)

    # Generate plots
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=str(output_dir / f'{model_name}_confusion_matrix.png'),
        title=f'Confusion Matrix - {model_name}'
    )

    plot_roc_curve(
        y_true, y_prob,
        save_path=str(output_dir / f'{model_name}_roc_curve.png'),
        title=f'ROC Curve - {model_name}'
    )

    plot_probability_distribution(
        y_true, y_prob,
        save_path=str(output_dir / f'{model_name}_prob_dist.png'),
        title=f'Probability Distribution - {model_name}'
    )

    # Save classification report
    report = classification_report(
        y_true, y_pred,
        target_names=['Miss', 'Make'],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / f'{model_name}_classification_report.csv')

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability_make': y_prob
    })
    predictions_df.to_csv(output_dir / f'{model_name}_predictions.csv', index=False)

    print(f"\nResults saved to: {output_dir}")

    return str(output_dir)


def load_and_evaluate(
    checkpoint_path: str,
    data_dir: str,
    splits_dir: str,
    output_dir: str = 'results',
    device: str = 'cuda',
    batch_size: int = 8
):
    """
    Load a trained model and evaluate on test set.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing video data
        splits_dir: Directory containing split CSVs
        output_dir: Directory to save results
        device: Device for evaluation
        batch_size: Batch size for evaluation
    """
    from train import FreeThrowLightningModule

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = FreeThrowLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Create test dataloader
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=batch_size,
        num_workers=4
    )

    # Evaluate
    results = evaluate_model(model, test_loader, device=device)

    # Generate report
    model_name = Path(checkpoint_path).stem
    generate_report(results, output_dir=output_dir, model_name=model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Free Throw Prediction Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--splits_dir', type=str, default='data/splits', help='Splits directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    args = parser.parse_args()

    load_and_evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
