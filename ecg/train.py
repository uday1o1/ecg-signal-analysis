import argparse, numpy as np, pandas as pd, os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .utils.common import load_config, apply_overrides, set_seed, device_from_cfg, ensure_dir
from .data.hdf5_dataset import HDF5ECG
from .models.resnet1d import ResNet1D
from .utils.metrics import sigmoid, per_class_auprc, per_class_auroc, macro_nanmean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dot.notation=VALUE to override YAML")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg["split"]["seed"])
    device = device_from_cfg(cfg)

    # Load labels CSV
    labels = pd.read_csv(cfg["data"]["labels_csv"]).values.astype(np.float32)
    n = labels.shape[0]
    n_train = int(round(n * (1 - cfg["split"]["val_fraction"])))

    if cfg["split"]["shuffle"]:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    ds_train = HDF5ECG(cfg["data"]["hdf5"], cfg["data"]["dataset_name"], labels_arr=labels,
                       start_idx=train_idx.min(), end_idx=train_idx.max()+1,
                       scale_by_1000=cfg["data"]["scale_by_1000"])
    ds_val   = HDF5ECG(cfg["data"]["hdf5"], cfg["data"]["dataset_name"], labels_arr=labels,
                       start_idx=val_idx.min(), end_idx=val_idx.max()+1,
                       scale_by_1000=cfg["data"]["scale_by_1000"])

    # NOTE: For simplicity we slice contiguous ranges; if your CSV order isn't random,
    # pass explicit index lists or pre-shuffle your HDF5+CSV consistently.
    # (You can change Dataset to accept an explicit index list if needed.)

    train_loader = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(ds_val, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = ResNet1D(n_classes=cfg["model"]["n_classes"],
                     in_ch=cfg["data"]["leads"],
                     stem=cfg["model"]["stem_channels"],
                     blocks=cfg["model"]["blocks"],
                     k=cfg["model"]["kernel_size"],
                     dropout=cfg["model"]["dropout"],
                     use_gap=cfg["model"]["use_gap"]).to(device)

    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=cfg["train"]["patience_lr"], min_lr=cfg["train"]["lr"]/100)
    loss_fn = nn.BCEWithLogitsLoss()

    amp = cfg["train"]["amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    ensure_dir(cfg["train"]["checkpoint_dir"])
    best_score = -np.inf
    es_wait = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [train]")
        for batch in pbar:
            x, y = batch
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip_norm"] and cfg["train"]["grad_clip_norm"]>0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())
        tr_loss /= len(ds_train)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true_all, y_prob_all = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [val]"):
                x, y = batch
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * x.size(0)
                y_true_all.append(y.cpu().numpy())
                y_prob_all.append(sigmoid(logits.cpu().numpy()))
        val_loss /= len(ds_val)
        y_true = np.concatenate(y_true_all)
        y_prob = np.concatenate(y_prob_all)

        auprc_c = per_class_auprc(y_true, y_prob)
        auroc_c = per_class_auroc(y_true, y_prob)
        auprc_m = macro_nanmean(auprc_c)
        auroc_m = macro_nanmean(auroc_c)
        sel = auprc_m if cfg["eval"]["metric"]=="auprc_macro" else auroc_m

        print(f"Train loss {tr_loss:.4f} | Val loss {val_loss:.4f} | AUPRC {auprc_m:.4f} | AUROC {auroc_m:.4f}")

        # sched + early stop
        scheduler.step(val_loss)
        es_improved = sel > best_score + 1e-6
        if es_improved:
            best_score = sel
            es_wait = 0
            torch.save({"model": model.state_dict(),
                        "cfg": cfg}, os.path.join(cfg["train"]["checkpoint_dir"], "best.ckpt"))
        else:
            es_wait += 1
        torch.save({"model": model.state_dict(),
                    "cfg": cfg}, os.path.join(cfg["train"]["checkpoint_dir"], "last.ckpt"))

        if es_wait >= cfg["train"]["patience_es"]:
            print("Early stopping.")
            break

    # Optional ONNX export
    if cfg.get("export",{}).get("onnx", False):
        dummy = torch.randn(1, cfg["data"]["leads"], cfg["data"]["samples"]).to(device)
        onnx_path = cfg["export"]["onnx_path"]
        torch.onnx.export(model, dummy, onnx_path, input_names=["signal"], output_names=["logits"], opset_version=17)
        print(f"Exported ONNX to {onnx_path}")

if __name__ == "__main__":
    main()
