import argparse, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from ecg.utils.common import load_config, apply_overrides

ABN = ["1dAVb","RBBB","LBBB","SB","AF","ST"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--csv", default="data/code15/exams_part0.csv",
                    help="Ground truth CSV")
    ap.add_argument("--preds", default="outputs/preds.npy",
                    help="Predicted probabilities file")
    ap.add_argument("--labels", default="outputs/preds_labels.npy",
                    help="Binarized predictions file")
    ap.add_argument("--outdir", default="outputs/figures",
                    help="Where to save plots")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(args.csv)
    y_true = df[ABN].values.astype(int)
    y_prob = np.load(args.preds)
    y_pred = np.load(args.labels)

    # -----------------------------
    # PR Curves
    # -----------------------------
    for i, name in enumerate(ABN):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        plt.figure()
        plt.plot(recall, precision, lw=2, label=f"AUPRC={auc(recall, precision):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"pr_{name}.png"))
        plt.close()

    # -----------------------------
    # ROC Curves
    # -----------------------------
    for i, name in enumerate(ABN):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr, tpr):.3f}")
        plt.plot([0,1],[0,1],"--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"roc_{name}.png"))
        plt.close()

    # -----------------------------
    # Bar chart of F1 scores
    # -----------------------------
    f1s = [f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in range(len(ABN))]
    plt.figure(figsize=(8,5))
    bars = plt.bar(ABN, f1s, color="skyblue", edgecolor="black")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.ylim(0,1)
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "f1_scores.png"))
    plt.close()

    print(f"Saved visualizations to {args.outdir}")

if __name__ == "__main__":
    main()
