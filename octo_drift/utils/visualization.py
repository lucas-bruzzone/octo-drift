"""
Visualization utilities for octo-drift.
"""

from typing import List, Optional, Set
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ..evaluation.confusion_matrix import Metrics
from ..core.structures import Example


def plot_metrics(
    metrics_list: List[Metrics],
    stream: List[Example],  # NOVO: para identificar quando classes aparecem
    novelty_flags: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    evaluation_interval: int = 1000,
) -> None:
    """
    Plot accuracy and unknown rate with novelty markers.

    Args:
        metrics_list: List of Metrics objects
        stream: Full data stream to identify class appearances
        novelty_flags: Binary flags for novelty detection
        save_path: Path to save figure
        evaluation_interval: Interval between evaluations
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    timestamps = [m.timestamp for m in metrics_list]
    accuracies = [m.accuracy * 100 for m in metrics_list]
    unknown_rates = [m.unknown_rate * 100 for m in metrics_list]

    # Plot accuracy
    ax1.plot(timestamps, accuracies, "g-", linewidth=2, label="Accuracy")
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot unknown rate
    ax2.plot(timestamps, unknown_rates, "orange", linewidth=2, label="Unknown Rate")
    ax2.set_xlabel("Evaluation Moments", fontsize=12)
    ax2.set_ylabel("Unknown Rate (%)", fontsize=12)
    ax2.set_ylim([0, max(10, max(unknown_rates) * 1.1)])  # Escala dinâmica
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Identificar quando novas classes APARECEM no stream
    class_appearance_timestamps = find_class_appearances(stream, evaluation_interval)

    for ts in class_appearance_timestamps:
        ax1.axvline(
            x=ts,
            color="black",
            linestyle="-",
            linewidth=1.5,
            alpha=0.7,
            label="New Class Appears" if ts == class_appearance_timestamps[0] else "",
        )
        ax2.axvline(x=ts, color="black", linestyle="-", linewidth=1.5, alpha=0.7)

    # Marcar quando novidades são DETECTADAS
    if novelty_flags:
        for i, flag in enumerate(novelty_flags):
            if flag > 0:
                ts = timestamps[i]
                ax1.axvline(
                    x=ts,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    label="Novelty Detected" if i == 0 else "",
                )
                ax2.axvline(x=ts, color="gray", linestyle="--", alpha=0.5)

    # Adicionar legendas apenas uma vez
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys(), fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def find_class_appearances(
    stream: List[Example], evaluation_interval: int
) -> List[int]:
    """
    Identifica os momentos de avaliação onde novas classes aparecem pela primeira vez.

    Args:
        stream: Lista de exemplos do stream
        evaluation_interval: Intervalo entre avaliações

    Returns:
        Lista de timestamps onde novas classes aparecem
    """
    seen_classes: Set[float] = set()
    appearance_timestamps = []

    for i, example in enumerate(stream):
        if example.true_label not in seen_classes:
            seen_classes.add(example.true_label)
            # Mapear para o próximo momento de avaliação
            eval_moment = ((i // evaluation_interval) + 1) * evaluation_interval
            if eval_moment not in appearance_timestamps:
                appearance_timestamps.append(eval_moment)

    return sorted(appearance_timestamps)


def plot_all_metrics(
    metrics_list: List[Metrics], save_path: Optional[str] = None
) -> None:
    """Plot all metrics (accuracy, precision, recall, F1) in one figure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    timestamps = [m.timestamp for m in metrics_list]
    accuracies = [m.accuracy * 100 for m in metrics_list]
    precisions = [m.precision * 100 for m in metrics_list]
    recalls = [m.recall * 100 for m in metrics_list]
    f1_scores = [m.f1_score * 100 for m in metrics_list]

    ax.plot(timestamps, accuracies, "g-", linewidth=2, label="Accuracy")
    ax.plot(timestamps, precisions, "b-", linewidth=2, label="Precision")
    ax.plot(timestamps, recalls, "r-", linewidth=2, label="Recall")
    ax.plot(timestamps, f1_scores, "purple", linewidth=2, label="F1-Score")

    ax.set_xlabel("Evaluation Moments", fontsize=12)
    ax.set_ylabel("Metric Value (%)", fontsize=12)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_parameter_sensitivity(
    results: dict,
    param_name: str,
    metric_name: str = "accuracy",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot metric vs parameter values for sensitivity analysis.

    Args:
        results: Dict mapping param_value -> metrics_list
        param_name: Parameter name (e.g., 'phi', 'T', 'k_short')
        metric_name: Metric to plot ('accuracy', 'f1_score', etc.)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    param_values = sorted(results.keys())
    mean_metrics = []
    std_metrics = []

    for param_val in param_values:
        metrics_list = results[param_val]
        values = [getattr(m, metric_name) for m in metrics_list]
        mean_metrics.append(np.mean(values) * 100)
        std_metrics.append(np.std(values) * 100)

    ax.errorbar(
        param_values, mean_metrics, yerr=std_metrics, marker="o", linewidth=2, capsize=5
    )

    ax.set_xlabel(f"{param_name}", fontsize=12)
    ax.set_ylabel(f"{metric_name.capitalize()} (%)", fontsize=12)
    ax.set_title(f"{metric_name.capitalize()} vs {param_name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_confusion_heatmap(
    confusion_matrix: np.ndarray, labels: List[str], save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix as heatmap.

    Args:
        confusion_matrix: 2D array of confusion matrix
        labels: Class labels
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(confusion_matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(
                j, i, confusion_matrix[i, j], ha="center", va="center", color="black"
            )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
