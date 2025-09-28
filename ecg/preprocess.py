# ecg/preprocess.py
import numpy as np
from scipy.signal import butter, filtfilt, resample
from sklearn.base import BaseEstimator, TransformerMixin

def bandpass(x, fs, low=0.5, high=40, order=4):
    b,a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b,a,x)

class ECGPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fs_in, fs_out=360, bp_low=0.5, bp_high=40):
        self.fs_in, self.fs_out = fs_in, fs_out
        self.bp_low, self.bp_high = bp_low, bp_high
    def fit(self, X, y=None): return self
    def transform(self, X):
        out=[]
        for s in X:
            s = bandpass(s, self.fs_in, self.bp_low, self.bp_high)
            if self.fs_out and self.fs_out!=self.fs_in:
                n = int(len(s)*self.fs_out/self.fs_in); s = resample(s, n)
            out.append(s.astype(np.float32))
        return out
