import numpy as np
from typing import Dict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)


def compute_metrics(y_true, y_prob, threshold=0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    gmean = np.sqrt((tp / (tp + fn + 1e-12)) * (tn / (tn + fp + 1e-12)))
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc),
        'pr_auc': float(pr),
        'gmean': float(gmean),
        'mcc': float(mcc),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def expected_cost(y_true, y_pred, cost_fp: float, cost_fn: float) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return cost_fp * fp + cost_fn * fn


def find_best_threshold(y_true, y_prob, metric: str = 'f1', beta: float = 1.0, cost_fp: float = 1.0, cost_fn: float = 25.0):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_val = 0.5, -np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'f1':
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            if p == 0 and r == 0:
                val = 0
            else:
                val = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
        elif metric == 'cost':
            val = -expected_cost(y_true, y_pred, cost_fp, cost_fn)
        else:
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val:
            best_val, best_thr = val, t
    return best_thr, best_val
