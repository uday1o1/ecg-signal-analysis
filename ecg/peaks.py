# ecg/peaks.py
import neurokit2 as nk
from sklearn.base import BaseEstimator, TransformerMixin

class RPeakDetector(BaseEstimator, TransformerMixin):
    def __init__(self, fs): self.fs=fs
    def fit(self, X, y=None): return self
    def transform(self, X):
        res=[]
        for s in X:
            _,info = nk.ecg_peaks(s, sampling_rate=self.fs, method='neurokit')
            res.append({"signal": s, "rpeaks": info["ECG_R_Peaks"]})
        return res
