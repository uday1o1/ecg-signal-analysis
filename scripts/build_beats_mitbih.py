# scripts/build_beats_mitbih.py
import numpy as np
from pathlib import Path
from ecg.io import list_records, load_record
from ecg.preprocess import ECGPreprocessor
from ecg.segment import BeatSegmenter
from ecg.utils import patient_split

DB="data/mitdb"
LEAD=0

def build_split(split_ids):
    X_recs=[]; fs=None; rpeaks_lists=[]; labels_lists=[]; seg_counts=[]
    for rid in split_ids:
        sig, fs0, rlocs, ysym = load_record(Path(DB)/rid, lead=LEAD)
        if fs is None: fs=fs0
        X_recs.append(sig); rpeaks_lists.append(rlocs); labels_lists.append(ysym)
    pre = ECGPreprocessor(fs_in=fs, fs_out=360)
    recs = pre.transform(X_recs)

    # segment per record, count kept beats, stack
    seg = BeatSegmenter(fs=360, pre_ms=200, post_ms=400)
    beats=[]; counts=[]
    for s, r in zip(recs, rpeaks_lists):
        # reuse the segmenter logic
        segs=[]
        for k in r:
            a=max(0, k - int(0.2*360)); b=min(len(s), k + int(0.4*360))
            if b-a == int(0.6*360):
                segs.append(s[a:b])
        counts.append(len(segs))
        if segs: beats.append(np.vstack(segs))
    X = np.vstack(beats)
    from ecg.align import BeatLabelAligner
    y = BeatLabelAligner().align(rpeaks_lists, labels_lists, counts)
    return X, y

if __name__ == "__main__":
    ids = list_records(DB)
    train_ids, val_ids, test_ids = patient_split(ids, seed=13, test_frac=0.2, val_frac=0.2)
    Xtr,ytr = build_split(train_ids)
    Xva,yva = build_split(val_ids)
    Xte,yte = build_split(test_ids)
    np.savez("data/mitbih_beats_360Hz.npz",
             Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)
