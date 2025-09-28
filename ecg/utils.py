# ecg/utils.py
import numpy as np, random
def patient_split(record_ids, seed=13, test_frac=0.2, val_frac=0.2):
    ids = sorted(record_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids); n_test = int(n*test_frac); n_val = int(n*val_frac)
    test = set(ids[:n_test]); val = set(ids[n_test:n_test+n_val])
    train = [i for i in ids if i not in test|val]
    return train, list(val), list(test)
