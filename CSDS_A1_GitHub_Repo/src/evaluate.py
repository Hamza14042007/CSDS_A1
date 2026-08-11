"""Evaluation metrics aligned with a rare-event, fixed-budget review queue.

Primary metric is PR-AUC (average precision) because the positive class is a
minority; ROC-AUC and Brier score add ranking and calibration checks; and the
budget metrics (precision/recall/lift at the top 20%) express operational value.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

import config


def budget_metrics(y_true, y_score, budget: float = config.REVIEW_BUDGET):
    """Precision, recall and lift when only the top ``budget`` share is reviewed."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = max(1, int(round(budget * n)))
    order = np.argsort(-y_score)               # highest score first
    top = order[:k]
    prevalence = y_true.mean()
    n_positive = y_true.sum()

    tp = y_true[top].sum()
    precision = tp / k
    recall = tp / n_positive if n_positive else float("nan")
    lift = precision / prevalence if prevalence else float("nan")
    return {
        "queue_size": k,
        "precision@budget": float(precision),
        "recall@budget": float(recall),
        "lift@budget": float(lift),
        "true_positives_in_queue": int(tp),
    }


def evaluate(y_true, y_score) -> dict:
    """All headline metrics for one model on one split."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    out.update(budget_metrics(y_true, y_score))
    return out
