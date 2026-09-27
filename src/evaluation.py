import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true, prob):
    """Pick the threshold that maximizes F1 on the VALIDATION set only."""
    thresholds = np.linspace(0.05, 0.95, 181)

    scores = [
        f1_score(
            y_true,
            (prob >= t).astype(int),
            zero_division=0,
        )
        for t in thresholds
    ]

    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx])


def evaluate_model(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, prob)
        ),
        "ap": float(
            average_precision_score(y_true, prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }