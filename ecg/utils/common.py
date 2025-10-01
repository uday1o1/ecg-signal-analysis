import os, random, yaml, math
import numpy as np
import torch

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def apply_overrides(cfg, overrides):
    # overrides like key.sub=val; supports dot paths
    for ov in overrides:
        if "=" not in ov: continue
        k, v = ov.split("=", 1)
        cursor = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            cursor = cursor.setdefault(kk, {})
        # best-effort type cast
        if v.lower() in ["true","false"]:
            v = v.lower() == "true"
        else:
            try:
                if "." in v: v = float(v);
                else: v = int(v)
            except: pass
        cursor[keys[-1]] = v
    return cfg

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_from_cfg(cfg):
    if cfg.get("device","auto") == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def num_batches(n_items, batch_size):
    return math.ceil(n_items / batch_size)
