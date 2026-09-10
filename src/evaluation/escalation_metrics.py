"""
Escalation decision metrics.
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_escalation_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute escalation decision metrics with 'escalate' as positive class."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["escalate"], average="binary",
        pos_label="escalate", zero_division=0,
    )

    # False negative rate: should-escalate but didn't
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "escalate" and p != "escalate")
    total_positives = sum(1 for t in y_true if t == "escalate")
    false_negative_rate = fn / total_positives if total_positives > 0 else 0.0

    # False positive rate: auto-handle but escalated
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "auto_handle" and p == "escalate")
    total_negatives = sum(1 for t in y_true if t == "auto_handle")
    false_positive_rate = fp / total_negatives if total_negatives > 0 else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_negative_rate": round(float(false_negative_rate), 4),
        "false_positive_rate": round(float(false_positive_rate), 4),
        "n_escalate_true": total_positives,
        "n_auto_handle_true": total_negatives,
        "n_samples": len(y_true),
    }
