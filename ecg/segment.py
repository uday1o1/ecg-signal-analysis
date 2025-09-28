# ecg/segment.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BeatSegmenter(BaseEstimator, TransformerMixin):
    def __init__(self, fs, pre_ms=200, post_ms=400):
        self.fs=fs; self.pre=int(pre_ms*fs/1000); self.post=int(post_ms*fs/1000)
    def fit(self, X,y=None): return self
    def transform(self, X):
        beats=[]
        for rec in X:
            s=rec["signal"]; r=rec["rpeaks"]
            for k in r:
                a=max(0,k-self.pre); b=min(len(s),k+self.post)
                seg=s[a:b]
                if len(seg)==self.pre+self.post:
                    beats.append(seg)
        return np.vstack(beats)
