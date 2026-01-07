"""
Evaluation utilities for the Anomaly Detection project.
Provides metrics, threshold selection, visualization, and report generation.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Default partial AUC threshold (configurable)
DEFAULT_PAUC_MAX_FPR = 0.1


# =============================================================================
# Helper Functions
# =============================================================================


def to_numpy(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert torch.Tensor to numpy.ndarray.

    Args:
        arr: Input array (numpy or torch tensor)

    Returns:
        numpy.ndarray
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr


# =============================================================================
# Core Metric Functions
# =============================================================================


def compute_binary_metrics(
    y_true: np.ndarray | torch.Tensor, y_pred: np.ndarray | torch.Tensor
) -> dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0=normal, 1=anomaly)
        y_pred: Predicted binary labels

    Returns:
        Dictionary with precision, recall, f1, accuracy
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0.0),
        "recall": recall_score(y_true, y_pred, zero_division=0.0),
        "f1": f1_score(y_true, y_pred, zero_division=0.0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_roc_auc(y_true: np.ndarray | torch.Tensor, scores: np.ndarray | torch.Tensor) -> float:
    """
    Compute Area Under the ROC Curve.

    Args:
        y_true: Ground truth binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        ROC AUC score
    """
    y_true = to_numpy(y_true)
    scores = to_numpy(scores)
    return roc_auc_score(y_true, scores)


def compute_pauc(
    y_true: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    max_fpr: float = DEFAULT_PAUC_MAX_FPR,
) -> float:
    """
    Compute partial Area Under the ROC Curve at low FPR.

    This is the DCASE challenge standard metric, focusing on
    detection performance at low false positive rates.

    Args:
        y_true: Ground truth binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        max_fpr: Maximum FPR to consider (default 0.1 = 10%)

    Returns:
        Partial AUC score (normalized to [0, 1])
    """
    y_true = to_numpy(y_true)
    scores = to_numpy(scores)
    return roc_auc_score(y_true, scores, max_fpr=max_fpr)


def compute_precision_recall_auc(
    y_true: np.ndarray | torch.Tensor, scores: np.ndarray | torch.Tensor
) -> float:
    """
    Compute Area Under the Precision-Recall Curve (Average Precision).

    Better metric for imbalanced datasets.

    Args:
        y_true: Ground truth binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)

    Returns:
        Average Precision score
    """
    y_true = to_numpy(y_true)
    scores = to_numpy(scores)
    return average_precision_score(y_true, scores)


# =============================================================================
# Threshold Selection
# =============================================================================


def find_optimal_threshold(
    y_true: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    method: Literal["f1", "youden"] = "f1",
) -> float:
    """
    Find optimal threshold for converting scores to binary predictions.

    Args:
        y_true: Ground truth binary labels
        scores: Anomaly scores
        method: Selection criterion
            - 'f1': Maximize F1 score
            - 'youden': Maximize TPR - FPR (Youden's J statistic)

    Returns:
        Optimal threshold value
    """
    y_true = to_numpy(y_true)
    scores = to_numpy(scores)

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        j_statistic = tpr - fpr
        optimal_idx = np.argmax(j_statistic)
        return thresholds[optimal_idx]

    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        # F1 = 2 * (precision * recall) / (precision + recall)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores, nan=0.0)
        # thresholds has length len(precision) - 1
        optimal_idx = np.argmax(f1_scores[:-1])
        return thresholds[optimal_idx]

    raise ValueError(f"Unknown method: {method}. Use 'f1' or 'youden'.")


def apply_threshold(scores: np.ndarray | torch.Tensor, threshold: float) -> np.ndarray:
    """
    Convert continuous scores to binary predictions.

    Args:
        scores: Anomaly scores
        threshold: Classification threshold

    Returns:
        Binary predictions (0 or 1)
    """
    scores = to_numpy(scores)
    return (scores >= threshold).astype(np.int64)


# =============================================================================
# Stratified Analysis
# =============================================================================


def evaluate_by_snr(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    snr: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute metrics for each SNR value.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        snr: SNR values for each sample
        scores: Optional anomaly scores for AUC metrics

    Returns:
        Dictionary mapping SNR value to metrics dict
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    snr = to_numpy(snr)
    if scores is not None:
        scores = to_numpy(scores)

    results = {}
    for snr_val in np.unique(snr):
        mask = snr == snr_val
        metrics = compute_binary_metrics(y_true[mask], y_pred[mask])

        if scores is not None and np.sum(y_true[mask]) > 0 and np.sum(y_true[mask]) < np.sum(mask):
            metrics["roc_auc"] = compute_roc_auc(y_true[mask], scores[mask])
            metrics["pr_auc"] = compute_precision_recall_auc(y_true[mask], scores[mask])

        results[int(snr_val)] = metrics

    return results


def evaluate_by_anomaly_class(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    original_labels: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor | None = None,
) -> dict[int, dict[str, float]]:
    """
    Compute metrics for each anomaly subclass (6, 7, 8).

    Evaluates detection performance for normal samples vs each anomaly type.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        original_labels: Original class labels (0-8)
        scores: Optional anomaly scores for AUC metrics

    Returns:
        Dictionary mapping anomaly class to metrics dict
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    original_labels = to_numpy(original_labels)
    if scores is not None:
        scores = to_numpy(scores)

    results = {}

    # Get indices of normal samples (classes 0-5)
    normal_mask = original_labels < 6

    for anomaly_class in [6, 7, 8]:
        # Include all normal samples + this specific anomaly class
        anomaly_mask = original_labels == anomaly_class
        subset_mask = normal_mask | anomaly_mask

        if not np.any(anomaly_mask):
            continue

        subset_y_true = y_true[subset_mask]
        subset_y_pred = y_pred[subset_mask]

        metrics = compute_binary_metrics(subset_y_true, subset_y_pred)
        metrics["n_samples"] = int(np.sum(anomaly_mask))

        if scores is not None:
            subset_scores = scores[subset_mask]
            if np.sum(subset_y_true) > 0 and np.sum(subset_y_true) < len(subset_y_true):
                metrics["roc_auc"] = compute_roc_auc(subset_y_true, subset_scores)
                metrics["pr_auc"] = compute_precision_recall_auc(subset_y_true, subset_scores)

        results[anomaly_class] = metrics

    return results


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_roc_curve(
    y_true: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    ax: plt.Axes | None = None,
    label: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot ROC curve with AUC annotation.

    Args:
        y_true: Ground truth binary labels
        scores: Anomaly scores
        ax: Matplotlib axes (creates new if None)
        label: Legend label for this curve
        save_path: Optional path to save figure

    Returns:
        Matplotlib Axes
    """
    y_true = to_numpy(y_true)
    scores = to_numpy(scores)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    label = f"AUC = {roc_auc:.3f}" if label is None else f"{label} (AUC = {roc_auc:.3f})"

    ax.plot(fpr, tpr, label=label, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_precision_recall_curve(
    y_true: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    ax: plt.Axes | None = None,
    label: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot Precision-Recall curve with AUC annotation.

    Args:
        y_true: Ground truth binary labels
        scores: Anomaly scores
        ax: Matplotlib axes (creates new if None)
        label: Legend label for this curve
        save_path: Optional path to save figure

    Returns:
        Matplotlib Axes
    """
    y_true = to_numpy(y_true)
    scores = to_numpy(scores)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    label = f"AUC = {pr_auc:.3f}" if label is None else f"{label} (AUC = {pr_auc:.3f})"

    ax.plot(recall, precision, label=label, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_confusion_matrix(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    ax: plt.Axes | None = None,
    normalize: bool = False,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot confusion matrix as a heatmap.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        ax: Matplotlib axes (creates new if None)
        normalize: If True, show percentages instead of counts
        save_path: Optional path to save figure

    Returns:
        Matplotlib Axes
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    normalize_param = "true" if normalize else None
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Normal", "Anomaly"],
        normalize=normalize_param,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_score_distribution(
    scores: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot histogram of anomaly scores by class.

    Helps visualize separability between normal and anomalous samples.

    Args:
        scores: Anomaly scores
        y_true: Ground truth binary labels
        ax: Matplotlib axes (creates new if None)
        save_path: Optional path to save figure

    Returns:
        Matplotlib Axes
    """
    scores = to_numpy(scores)
    y_true = to_numpy(y_true)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    ax.hist(normal_scores, bins=50, alpha=0.7, label="Normal", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label="Anomaly", density=True)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor | None = None,
    snr: np.ndarray | torch.Tensor | None = None,
    original_labels: np.ndarray | torch.Tensor | None = None,
    model_name: str | None = None,
) -> str:
    """
    Generate formatted text report with all metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        scores: Optional anomaly scores for AUC metrics
        snr: Optional SNR values for stratified analysis
        original_labels: Optional original class labels for per-class analysis
        model_name: Optional model name for report header

    Returns:
        Formatted string report
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    lines = []
    if model_name:
        header = f"=== Evaluation Report: {model_name} ==="
    else:
        header = "=== Evaluation Report ==="
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    # Basic metrics
    metrics = compute_binary_metrics(y_true, y_pred)
    lines.append("Binary Classification Metrics:")
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall:    {metrics['recall']:.4f}")
    lines.append(f"  F1 Score:  {metrics['f1']:.4f}")
    lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
    lines.append("")

    # AUC metrics
    if scores is not None:
        scores = to_numpy(scores)
        lines.append("AUC Metrics:")
        lines.append(f"  ROC AUC:   {compute_roc_auc(y_true, scores):.4f}")
        lines.append(f"  pAUC@10%:  {compute_pauc(y_true, scores):.4f}")
        lines.append(f"  PR AUC:    {compute_precision_recall_auc(y_true, scores):.4f}")
        lines.append("")

    # SNR analysis
    if snr is not None:
        snr = to_numpy(snr)
        lines.append("Performance by SNR:")
        snr_metrics = evaluate_by_snr(y_true, y_pred, snr, scores)
        for snr_val in sorted(snr_metrics.keys()):
            m = snr_metrics[snr_val]
            p, r, f = m["precision"], m["recall"], m["f1"]
            line = f"  SNR {snr_val:2d} dB: P={p:.3f} R={r:.3f} F1={f:.3f}"
            if "roc_auc" in m:
                line += f" AUC={m['roc_auc']:.3f}"
            lines.append(line)
        lines.append("")

    # Per-anomaly-class analysis
    if original_labels is not None:
        original_labels = to_numpy(original_labels)
        lines.append("Performance by Anomaly Class:")
        class_metrics = evaluate_by_anomaly_class(y_true, y_pred, original_labels, scores)
        for cls in sorted(class_metrics.keys()):
            m = class_metrics[cls]
            line = f"  Class {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
            if "roc_auc" in m:
                line += f" AUC={m['roc_auc']:.3f}"
            line += f" (n={m['n_samples']})"
            lines.append(line)
        lines.append("")

    return "\n".join(lines)


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        results: Dictionary mapping model name to metrics dict
                 e.g., {"Model A": {"precision": 0.9, "recall": 0.8}}

    Returns:
        DataFrame with models as rows and metrics as columns
    """
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    return df.round(4)


if __name__ == "__main__":
    # Quick test with synthetic data
    np.random.seed(42)

    # Generate synthetic data
    n_normal, n_anomaly = 100, 50
    y_true = np.array([0] * n_normal + [1] * n_anomaly)
    scores = np.concatenate(
        [
            np.random.normal(0.3, 0.15, n_normal),  # Normal scores
            np.random.normal(0.7, 0.15, n_anomaly),  # Anomaly scores
        ]
    )
    scores = np.clip(scores, 0, 1)

    print("Testing eval_utils with synthetic data...")
    print(f"  Samples: {n_normal} normal + {n_anomaly} anomaly")
    print()

    # Find optimal threshold
    threshold = find_optimal_threshold(y_true, scores, method="f1")
    print(f"Optimal threshold (F1): {threshold:.3f}")

    # Apply threshold
    y_pred = apply_threshold(scores, threshold)

    # Compute metrics
    metrics = compute_binary_metrics(y_true, y_pred)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print()

    # AUC metrics
    print(f"ROC AUC:   {compute_roc_auc(y_true, scores):.3f}")
    print(f"pAUC@10%:  {compute_pauc(y_true, scores):.3f}")
    print(f"PR AUC:    {compute_precision_recall_auc(y_true, scores):.3f}")
