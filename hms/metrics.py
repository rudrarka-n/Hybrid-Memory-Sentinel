"""Evaluation metrics shared by both channels."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, false-positive rate and ROC AUC.

    Label 1 means malicious.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    def div(a, b):
        return float(a / b) if b else 0.0

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    out = {
        "n": int(len(y_true)),
        "accuracy": div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "fpr": div(fp, fp + tn),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    out["auc"] = float(roc_auc_score(y_true, prob)) if len(set(y_true)) == 2 else None
    return out


def format_metrics(name: str, m: dict) -> str:
    auc = "n/a" if m["auc"] is None else f"{m['auc']:.4f}"
    return (
        f"{name:<22} acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
        f"rec={m['recall']:.4f} f1={m['f1']:.4f} fpr={m['fpr']:.4f} auc={auc}"
    )
