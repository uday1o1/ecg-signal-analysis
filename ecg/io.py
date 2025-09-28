from pathlib import Path
import numpy as np, wfdb

AAMI_MAP = {
    'N':0,'L':0,'R':0,'e':0,'j':0,'.':0,
    'A':1,'a':1,'J':1,'S':1,
    'V':2,'E':2,
    'F':3,
    'P':4,'/':4,'f':4,'Q':4
}

def load_record(record_path, lead=0):
    signals, fields = wfdb.rdsamp(str(record_path))
    ann = wfdb.rdann(str(record_path), extension='atr')
    sig = signals[:, lead].astype(np.float32)
    fs = fields['fs']
    r_locs = ann.sample.astype(int)
    symbols = np.array(ann.symbol)
    keep = np.array([s in AAMI_MAP for s in symbols])
    return sig, fs, r_locs[keep], np.array([AAMI_MAP[s] for s in symbols[keep]])

def list_records(db_root="data/mitdb"):
    p = Path(db_root)
    return sorted({f.stem for f in p.glob("*.hea")})
