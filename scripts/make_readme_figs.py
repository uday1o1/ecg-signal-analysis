#!/usr/bin/env python3
"""
Generate figures for README:
- Class distribution bar chart (train/val/test)
- Confusion matrices (val/test) for classifier
- Precision-Recall + ROC curves for anomaly detector (test)
- Sample beats panel

Saves PNGs to docs/figs/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc
)
from sklearn.ensemble import IsolationForest

# --------------------------
# Config / paths
# --------------------------
DATA_NPZ = Path("data/mitbih_beats_360Hz.npz")
OUT_DIR = Path("docs/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

AAMI_NAMES = {0:"N", 1:"S", 2:"V", 3:"F", 4:"Q"}

# --------------------------
# Features (same as baseline)
# --------------------------
def beat_morph_features(X):
    peak = X.max(1)
    trough = X.min(1)
    energy = (X**2).sum(1)
    l1 = np.abs(np.diff(X, axis=1)).sum(1)
    thr = (0.2 * (peak - trough) + trough)[:, None]
    above = (X >= thr)
    left = above.argmax(1)
    right = X.shape[1] - np.flip(above, axis=1).argmax(1)
    width = right - left
    return np.c_[peak, trough, energy, l1, width].astype(np.float32)

def load_data():
    assert DATA_NPZ.exists(), f"Missing {DATA_NPZ}. Run: python scripts/build_beats_mitbih.py"
    D = np.load(DATA_NPZ)
    Xtr, ytr = D["Xtr"], D["ytr"]
    Xva, yva = D["Xva"], D["yva"]
    Xte, yte = D["Xte"], D["yte"]
    return Xtr, ytr, Xva, yva, Xte, yte

# --------------------------
# Plots
# --------------------------
def plot_class_distribution(ytr, yva, yte):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for i, (name, y) in enumerate([("Train", ytr), ("Val", yva), ("Test", yte)]):
        classes, counts = np.unique(y, return_counts=True)
        labels = [AAMI_NAMES.get(int(c), str(c)) for c in classes]
        ax[i].bar(labels, counts)
        ax[i].set_title(f"{name} class counts")
        ax[i].set_xlabel("AAMI class")
        if i == 0:
            ax[i].set_ylabel("Count")
    plt.tight_layout()
    out = OUT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def plot_sample_beats(X, fs=360, n=5, title="Sample segmented beats"):
    n = min(n, len(X))
    if n == 0:
        return
    pre_s = 0.2
    t = np.arange(X.shape[1]) / fs - pre_s
    plt.figure(figsize=(10, 3))
    for i in range(n):
        plt.plot(t, X[i])
    plt.xlabel("Time (s)"); plt.ylabel("mV"); plt.title(title)
    plt.tight_layout()
    out = OUT_DIR / "sample_beats.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def train_classifier_and_plot(Xtr, ytr, Xva, yva, Xte, yte):
    Ftr, Fva, Fte = beat_morph_features(Xtr), beat_morph_features(Xva), beat_morph_features(Xte)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))  # OVR auto for binary multi-class
    ])
    clf.fit(Ftr, ytr)

    for split_name, F, y in [("val", Fva, yva), ("test", Fte, yte)]:
        yhat = clf.predict(F)
        cm = confusion_matrix(y, yhat, labels=[0,1,2,3,4], normalize=None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[AAMI_NAMES[i] for i in range(5)])
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix ({split_name.upper()})")
        plt.tight_layout()
        out = OUT_DIR / f"cm_{split_name}.png"
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        print("Saved:", out)

def train_anomaly_and_plot(Xtr, ytr, Xte, yte):
    # Train on NORMAL beats only
    Xtr_N = Xtr[ytr == 0]
    Ftr_N = beat_morph_features(Xtr_N)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0)
    iso.fit(Ftr_N)

    # Scores on test
    Fte = beat_morph_features(Xte)
    scores = -iso.score_samples(Fte)  # higher = more anomalous
    y_bin = (yte != 0).astype(int)

    # PR curve
    prec, rec, thr = precision_recall_curve(y_bin, scores)
    ap = np.trapz(prec[::-1], rec[::-1])  # (rough) area under PR via trapezoid

    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Anomaly PR (TEST). AP≈{ap:.3f}")
    plt.tight_layout()
    out_pr = OUT_DIR / "pr_anomaly_test.png"
    plt.savefig(out_pr, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved:", out_pr)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_bin, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"Anomaly ROC (TEST). AUC={roc_auc:.3f}")
    plt.tight_layout()
    out_roc = OUT_DIR / "roc_anomaly_test.png"
    plt.savefig(out_roc, dpi=180, bbox_inches="tight")
    plt.close()
    print("Saved:", out_roc)

def main():
    Xtr, ytr, Xva, yva, Xte, yte = load_data()
    # Figures
    plot_class_distribution(ytr, yva, yte)
    plot_sample_beats(Xtr)  # first few train beats
    train_classifier_and_plot(Xtr, ytr, Xva, yva, Xte, yte)
    train_anomaly_and_plot(Xtr, ytr, Xte, yte)
    print("\nAll figures written to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
