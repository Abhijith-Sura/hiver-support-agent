"""
Intent classification metrics: accuracy, F1, confusion matrix.
"""
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

from src.config import INTENT_LABELS, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_intent_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict:
    """Compute all intent classification metrics."""
    labels = labels or INTENT_LABELS
    # Filter to only labels that appear in the data
    present_labels = sorted(set(y_true) | set(y_pred))
    present_labels = [l for l in labels if l in present_labels]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )

    per_intent = {}
    for i, label in enumerate(present_labels):
        per_intent[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    report = classification_report(y_true, y_pred, labels=present_labels, zero_division=0)

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_intent": per_intent,
        "classification_report": report,
        "n_samples": len(y_true),
    }


def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
    save_path: Path | None = None,
) -> Path:
    """Generate and save a confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = labels or INTENT_LABELS
    present_labels = sorted(set(y_true) | set(y_pred))
    present_labels = [l for l in labels if l in present_labels]

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_labels, yticklabels=present_labels, ax=ax)
    ax.set_xlabel("Predicted Intent")
    ax.set_ylabel("True Intent")
    ax.set_title("Intent Classification Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = save_path or (RESULTS_DIR / "confusion_matrix.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", save_path)
    return save_path
