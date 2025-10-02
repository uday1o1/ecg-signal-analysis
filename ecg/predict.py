import argparse, json, os
import numpy as np, torch
from torch.utils.data import DataLoader

from ecg.utils.common import load_config, apply_overrides, device_from_cfg
from ecg.data.hdf5_dataset import HDF5ECG
from ecg.data.sharded_hdf5_dataset import ShardedHDF5ECG
from ecg.models.resnet1d import ResNet1D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("overrides", nargs="*", help="dot.notation=VALUE to override YAML")
    ap.add_argument("--ckpt", default="./checkpoints/best.ckpt", help="Checkpoint to load")
    ap.add_argument("--out.npy", dest="out_npy", default="./outputs/dnn_output.npy",
                    help="Path to save probabilities (.npy)")
    ap.add_argument("--thresholds_json", default=None,
                    help="Optional thresholds JSON from select_thresholds.py to also save binarized labels")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = device_from_cfg(cfg)

    # -----------------------------
    # Dataset (no labels needed for inference)
    # -----------------------------
    if cfg["data"]["mode"] == "sharded":
        ds = ShardedHDF5ECG(
            cfg["data"]["index_csv"],
            scale_by_1000=cfg["data"]["scale_by_1000"],
            has_labels=False
        )
    else:
        ds = HDF5ECG(
            cfg["data"]["hdf5"],
            cfg["data"]["dataset_name"],
            labels_arr=None,
            start_idx=0,
            end_idx=None,
            scale_by_1000=cfg["data"]["scale_by_1000"]
        )

    loader = DataLoader(
        ds,
        batch_size=cfg["predict"]["batch_size"],
        shuffle=False,
        num_workers=cfg["predict"]["num_workers"],
        pin_memory=True
    )

    # -----------------------------
    # Load model
    # -----------------------------
    chk = torch.load(args.ckpt, map_location="cpu")
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
    # Inference
    # -----------------------------
    probs = []
    with torch.no_grad():
        for x in loader:
            # x is just tensors when has_labels=False
            x = x.to(device)
            logits = model(x).cpu().numpy()
            p = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            probs.append(p)

    y_score = np.concatenate(probs, axis=0)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, y_score)
    print(f"Saved probabilities to {args.out_npy} (shape={y_score.shape})")

    # -----------------------------
    # Optional: apply thresholds -> binarized labels
    # -----------------------------
    if args.thresholds_json:
        with open(args.thresholds_json, "r") as f:
            th = np.array(json.load(f)["thresholds"], dtype=np.float32)
        if th.shape[-1] != y_score.shape[-1]:
            raise ValueError(f"Thresholds vector size {th.shape} != probs size {y_score.shape}")
        y_pred = (y_score > th).astype(np.int32)
        out_labels = args.out_npy.replace(".npy", "_labels.npy")
        np.save(out_labels, y_pred)
        print(f"Saved binarized labels to {out_labels} (shape={y_pred.shape})")


if __name__ == "__main__":
    main()
