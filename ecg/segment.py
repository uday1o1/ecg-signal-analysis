import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BeatSegmenter(BaseEstimator, TransformerMixin):
    def __init__(self, fs, pre_ms=200, post_ms=400, use_centers=None):
        self.fs = fs
        self.pre = int(pre_ms * fs / 1000)
        self.post = int(post_ms * fs / 1000)
        self.use_centers = use_centers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        beats, counts = [], []
        if self.use_centers is not None:
            for s, centers in zip(X, self.use_centers):
                segs = []
                for c in centers:
                    a = max(0, c - self.pre); b = min(len(s), c + self.post)
                    if b - a == self.pre + self.post:
                        segs.append(s[a:b])
                counts.append(len(segs))
                if segs:
                    beats.append(np.vstack(segs))
        else:
            for rec in X:
                s = rec["signal"]; r = rec["rpeaks"]
                segs = []
                for k in r:
                    a = max(0, k - self.pre); b = min(len(s), k + self.post)
                    if b - a == self.pre + self.post:
                        segs.append(s[a:b])
                counts.append(len(segs))
                if segs:
                    beats.append(np.vstack(segs))
        if not beats:
            return np.empty((0, self.pre + self.post))
        self.counts_ = counts
        return np.vstack(beats)
