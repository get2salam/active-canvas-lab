"""Evaluation metrics for active learning experiments.

All computation is pure Python — no external dependencies.
"""
from typing import Dict, List, Any, Tuple


def accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(
    y_true: list,
    y_pred: list,
    labels: Tuple[int, ...] = (0, 1),
) -> Dict[Tuple[int, int], int]:
    """Return {(true_label, predicted_label): count} dict."""
    matrix: Dict[Tuple[int, int], int] = {
        (t, p): 0 for t in labels for p in labels
    }
    for t, p in zip(y_true, y_pred):
        key = (t, p)
        if key in matrix:
            matrix[key] += 1
    return matrix


def auc_learning_curve(history: List[Dict[str, Any]]) -> float:
    """Area under the learning curve, normalised to the n_labeled range.

    Uses the trapezoidal rule so the result is bounded in [0, 1]
    regardless of how many rounds were run.
    """
    if not history:
        return 0.0
    if len(history) == 1:
        return history[0]["accuracy"]
    ns   = [h["n_labeled"] for h in history]
    accs = [h["accuracy"]  for h in history]
    span = ns[-1] - ns[0]
    if span == 0:
        return accs[-1]
    area = 0.0
    for i in range(1, len(history)):
        dx    = (ns[i] - ns[i - 1]) / span
        area += 0.5 * (accs[i] + accs[i - 1]) * dx
    return area


def summarise(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One-call summary of a completed experiment."""
    if not history:
        return {}
    accs = [h["accuracy"] for h in history]
    return {
        "initial_accuracy": history[0]["accuracy"],
        "final_accuracy":   history[-1]["accuracy"],
        "peak_accuracy":    max(accs),
        "n_labeled_final":  history[-1]["n_labeled"],
        "auc":              auc_learning_curve(history),
        "n_rounds":         len(history),
    }
