import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA = Path("data/mitbih_beats_360Hz.npz")

def main():
    assert DATA.exists(), f"Missing {DATA}. Run: python scripts/build_beats_mitbih.py"
    D = np.load(DATA)
    Xtr,ytr = D["Xtr"],D["ytr"]
    Xva,yva = D["Xva"],D["yva"]
    Xte,yte = D["Xte"],D["yte"]

    print("[load] shapes:",
          {k:D[k].shape for k in ["Xtr","ytr","Xva","yva","Xte","yte"]},
          flush=True)

    def feat(X):
        peak=X.max(1); trough=X.min(1)
        energy=(X**2).sum(1); l1=np.abs(np.diff(X,1)).sum(1)
        thr=(0.2*(peak-trough)+trough)[:,None]
        above=(X>=thr)
        left=above.argmax(1)
        right=X.shape[1]-np.flip(above,1).argmax(1)
        width=right-left
        return np.c_[peak,trough,energy,l1,width].astype(np.float32)

    Ftr, Fva, Fte = feat(Xtr), feat(Xva), feat(Xte)
    print("[feat] Ftr/Fva/Fte:", Ftr.shape, Fva.shape, Fte.shape, flush=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", multi_class="ovr"))
    ])
    pipe.fit(Ftr, ytr)
    print("[train] done", flush=True)

    print("\n[VAL] classification report")
    print(classification_report(yva, pipe.predict(Fva), digits=3), flush=True)

    print("\n[TEST] classification report")
    print(classification_report(yte, pipe.predict(Fte), digits=3), flush=True)

if __name__ == "__main__":
    main()
