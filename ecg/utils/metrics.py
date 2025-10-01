import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, f1_score
)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def per_class_auroc(y_true, y_prob):
    scores=[]
    for k in range(y_true.shape[1]):
        try:
            scores.append(roc_auc_score(y_true[:,k], y_prob[:,k]))
        except ValueError:
            scores.append(np.nan)
    return np.array(scores)

def per_class_auprc(y_true, y_prob):
    return np.array([
        average_precision_score(y_true[:,k], y_prob[:,k]) if len(np.unique(y_true[:,k]))>1 else np.nan
        for k in range(y_true.shape[1])
    ])

def macro_nanmean(x): return np.nanmean(x) if np.any(~np.isnan(x)) else np.nan

def optimal_pr_thresholds(y_true, y_prob):
    """Return per-class thresholds maximizing F1 on PR curve (val set)."""
    n_classes = y_true.shape[1]
    th = np.zeros(n_classes, dtype=np.float32)
    pr = np.zeros((n_classes,2), dtype=np.float32)
    for k in range(n_classes):
        p, r, t = precision_recall_curve(y_true[:,k], y_prob[:,k])
        # precision_recall_curve returns len(t)+1 points; align thresholds
        f1 = np.nan_to_num(2*p*r/(p+r+1e-12))
        idx = np.argmax(f1)
        # threshold index corresponds to idx-1 (except idx==0)
        th[k] = (t[idx-1] if idx>0 else (t[0]-1e-8)) if len(t)>0 else 0.5
        pr[k] = [p[idx], r[idx]]
    return th, pr

def f1_per_class(y_true, y_pred_bin):
    return np.array([
        f1_score(y_true[:,k], y_pred_bin[:,k], zero_division=0) for k in range(y_true.shape[1])
    ])
