import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from ecg.utils.common import load_config, apply_overrides

ABN = ["1dAVb","RBBB","LBBB","SB","AF","ST"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("overrides", nargs="*", help="dot.notation=VALUE to override YAML")
    ap.add_argument("--csv", default="data/code15/exams_part0.csv",
                    help="Ground truth CSV with labels (filtered shard)")
    ap.add_argument("--preds", default="outputs/preds.npy",
                    help="Predicted probabilities file (.npy)")
    ap.add_argument("--labels", default="outputs/preds_labels.npy",
                    help="Binarized labels file (.npy)")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    # -----------------------------
    # Load ground truth
    # -----------------------------
    df = pd.read_csv(args.csv)
    if not all(a in df.columns for a in ABN):
        raise ValueError(f"CSV {args.csv} missing abnormality columns: {ABN}")
    y_true = df[ABN].values.astype(int)

    # -----------------------------
    # Load predictions
    # -----------------------------
    y_prob = np.load(args.preds)
    y_pred = np.load(args.labels)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")

    # -----------------------------
    # Metrics
    # -----------------------------
    print("=== Classification Report (per-class) ===")
    print(classification_report(y_true, y_pred, target_names=ABN, zero_division=0))

    print("\n=== AUROC per class ===")
    aurocs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(len(ABN))]
    for name, val in zip(ABN, aurocs):
        print(f"{name}: {val:.4f}")

    print("\n=== AUPRC per class ===")
    auprcs = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(len(ABN))]
    for name, val in zip(ABN, auprcs):
        print(f"{name}: {val:.4f}")

    print("\n=== Macro AUROC ===", roc_auc_score(y_true, y_prob, average="macro"))
    print("=== Macro AUPRC ===", average_precision_score(y_true, y_prob, average="macro"))

if __name__ == "__main__":
    main()
