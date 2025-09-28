import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MorphologyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, beats):
        peak = beats.max(axis=1); trough = beats.min(axis=1)
        energy = (beats**2).sum(axis=1)
        l1 = np.abs(np.diff(beats,axis=1)).sum(axis=1)
        thr = (0.2*(peak - trough) + trough)[:,None]
        above = (beats >= thr)
        left = above.argmax(axis=1)
        right = beats.shape[1]-np.flip(above,axis=1).argmax(axis=1)
        width = (right-left)
        return np.c_[peak, trough, energy, l1, width]
