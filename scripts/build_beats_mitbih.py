from pathlib import Path
import numpy as np
from ecg.io import list_records, load_record
from ecg.preprocess import ECGPreprocessor
from ecg.segment import BeatSegmenter
from ecg.align import BeatLabelAligner
from ecg.utils import patient_split

DB_DIR = Path("data/mitdb")
LEAD = 0
FS_OUT = 360
PRE_MS = 200
POST_MS = 400
OUT_NPZ = Path("data/mitbih_beats_360Hz.npz")


def build_split(record_ids):
    X_signals = []
    fs = None
    rlocs_list = []
    y_list = []

    for rid in record_ids:
        sig, fs0, rlocs, y = load_record(DB_DIR / rid, lead=LEAD)
        if fs is None:
            fs = fs0
        X_signals.append(sig)
        rlocs_list.append(rlocs)
        y_list.append(y)

    pre = ECGPreprocessor(fs_in=fs, fs_out=FS_OUT, bp_low=0.5, bp_high=40.0)
    Xf = pre.transform(X_signals)

    seg = BeatSegmenter(fs=FS_OUT, pre_ms=PRE_MS, post_ms=POST_MS, use_centers=rlocs_list)
    X_beats = seg.transform(Xf)
    counts = seg.counts_

    aligner = BeatLabelAligner()
    y_beats = aligner.align_by_counts(y_list, counts)

    expected_len = int(round((PRE_MS + POST_MS) * FS_OUT / 1000.0))
    assert X_beats.ndim == 2 and X_beats.shape[1] == expected_len, \
        f"Expected beat length {expected_len}, got {X_beats.shape}"
    assert len(X_beats) == len(y_beats), "Beats and labels must have same length"
    assert set(np.unique(y_beats)).issubset({0, 1, 2, 3, 4}), \
        f"Unexpected labels present: {set(np.unique(y_beats))}"

    return X_beats, y_beats


def summarize_split(name, X, y):
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    dist = " ".join([f"{int(c)}x{int(k)}" for k, c in zip(classes, counts)])
    print(f"[{name}] beats={n}, window_len={X.shape[1]}, class_dist: {dist}")


def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    records = list_records(str(DB_DIR))
    if not records:
        raise SystemExit(
            f"No records found under {DB_DIR}. "
            "Run: python -c \"import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mitdb')\""
        )
    print(f"Found {len(records)} records")

    train_ids, val_ids, test_ids = patient_split(records, seed=13, test_frac=0.2, val_frac=0.2)
    print(f"Split -> train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    print("Building TRAIN...")
    Xtr, ytr = build_split(train_ids)
    summarize_split("train", Xtr, ytr)

    print("Building VAL...")
    Xva, yva = build_split(val_ids)
    summarize_split("val", Xva, yva)

    print("Building TEST...")
    Xte, yte = build_split(test_ids)
    summarize_split("test", Xte, yte)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    print(f"Saved: {OUT_NPZ.resolve()}")


if __name__ == "__main__":
    main()
