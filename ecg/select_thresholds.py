import argparse, json, os
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

from ecg.utils.common import load_config, apply_overrides, device_from_cfg
from ecg.data.hdf5_dataset import HDF5ECG
from ecg.data.sharded_hdf5_dataset import ShardedHDF5ECG
from ecg.models.resnet1d import ResNet1D
from ecg.utils.metrics import sigmoid, optimal_pr_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("ckpt", nargs="?")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    ckpt_path = cfg.get("ckpt") or args.ckpt or "./checkpoints/best.ckpt"
    device = device_from_cfg(cfg)

    # -----------------------------
    # Dataset setup
    # -----------------------------
    if cfg["data"]["mode"] == "sharded":
        ds_val = ShardedHDF5ECG(
            cfg["data"]["index_csv"],
            scale_by_1000=cfg["data"]["scale_by_1000"],
            has_labels=True
        )
    else:  # single-file HDF5
        labels = pd.read_csv(cfg["data"]["labels_csv"]).values.astype(np.float32)
        n = labels.shape[0]
        n_train = int(round(n * (1 - cfg["split"]["val_fraction"])))
        val_start = n_train
        ds_val = HDF5ECG(
            cfg["data"]["hdf5"],
            cfg["data"]["dataset_name"],
            labels_arr=labels,
            start_idx=val_start,
            end_idx=n,
            scale_by_1000=cfg["data"]["scale_by_1000"]
        )

    loader = DataLoader(
        ds_val,
        batch_size=cfg["predict"]["batch_size"],
        shuffle=False,
        num_workers=cfg["predict"]["num_workers"],
        pin_memory=True
    )

    # -----------------------------
    # Load model
    # -----------------------------
    chk = torch.load(ckpt_path, map_location="cpu")
    mcfg = chk["cfg"]["model"]
    in_ch = chk["cfg"]["data"]["leads"]

    model = ResNet1D(
        n_classes=mcfg["n_classes"],
        in_ch=in_ch,
        stem=mcfg["stem_channels"],
        blocks=mcfg["blocks"],
        k=mcfg["kernel_size"],
        dropout=mcfg["dropout"],
        use_gap=mcfg["use_gap"]
    ).to(device)

    model.load_state_dict(chk["model"])
    model.eval()

    # -----------------------------
    # Collect predictions
    # -----------------------------
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            logits = model(x).cpu().numpy()
            y_prob_all.append(1 / (1 + np.exp(-logits)))
            y_true_all.append(y.numpy())

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    # -----------------------------
    # Compute thresholds
    # -----------------------------
    th, pr = optimal_pr_thresholds(y_true, y_prob)
    out = {"thresholds": th.tolist(), "precision_recall": pr.tolist()}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/thresholds.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved thresholds to outputs/thresholds.json")


if __name__ == "__main__":
    main()
