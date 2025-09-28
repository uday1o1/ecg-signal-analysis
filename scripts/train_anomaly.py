# train_anomaly.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

D = np.load("data/mitbih_beats_360Hz.npz")
Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]

Xtr_norm = Xtr[ytr==0]  # AAMI N
feat = lambda X: np.c_[X.max(1), X.min(1), (X**2).sum(1), np.abs(np.diff(X,1)).sum(1)]
Ftr = feat(Xtr_norm)
clf = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(Ftr)

Fte = feat(Xte)
scores = -clf.score_samples(Fte)  # higher = more anomalous
y_bin = (yte!=0).astype(int)      # anomaly = non-N
print("AUROC:", roc_auc_score(y_bin, scores))
print("AUPRC:", average_precision_score(y_bin, scores))

# choose threshold at 95th percentile of normal validation scores if you want a binary alarm
