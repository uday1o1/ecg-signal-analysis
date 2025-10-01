import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader
from .utils.common import load_config, apply_overrides, device_from_cfg
from .data.hdf5_dataset import HDF5ECG
from .models.resnet1d import ResNet1D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--ckpt", default="./checkpoints/best.ckpt")
    ap.add_argument("--out.npy", dest="out_npy", default="./outputs/dnn_output.npy")
    ap.add_argument("--thresholds_json", default=None, help="optional JSON from select_thresholds")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = device_from_cfg(cfg)

    # labels optional for predict; we pass None to Dataset
    ds = HDF5ECG(cfg["data"]["hdf5"], cfg["data"]["dataset_name"], labels_arr=None,
                 start_idx=0, end_idx=None, scale_by_1000=cfg["data"]["scale_by_1000"])
    loader = DataLoader(ds, batch_size=cfg["predict"]["batch_size"], shuffle=False,
                        num_workers=cfg["predict"]["num_workers"], pin_memory=True)

    chk = torch.load(args.ckpt, map_location="cpu")
    mcfg = chk["cfg"]["model"]
    in_ch = chk["cfg"]["data"]["leads"]
    model = ResNet1D(n_classes=mcfg["n_classes"], in_ch=in_ch, stem=mcfg["stem_channels"],
                     blocks=mcfg["blocks"], k=mcfg["kernel_size"],
                     dropout=mcfg["dropout"], use_gap=mcfg["use_gap"]).to(device)
    model.load_state_dict(chk["model"]); model.eval()

    probs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            p = 1/(1+np.exp(-logits))
            probs.append(p)
    y_score = np.concatenate(probs)
    np.save(args.out_npy, y_score)
    print(f"Saved probabilities to {args.out_npy}")

    if args.thresholds_json:
        with open(args.thresholds_json,"r") as f:
            th = np.array(json.load(f)["thresholds"])
        y_pred = (y_score > th).astype(np.int32)
        np.save(args.out_npy.replace(".npy","_labels.npy"), y_pred)
        print(f"Saved binarized labels to {args.out_npy.replace('.npy','_labels.npy')}")

if __name__ == "__main__":
    main()
