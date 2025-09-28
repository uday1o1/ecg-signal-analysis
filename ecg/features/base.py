from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureExtractorBase(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): raise NotImplementedError

class MorphologyFeatures(FeatureExtractorBase):
    def transform(self, beats):
        peak = beats.max(axis=1)
        trough = beats.min(axis=1)
        energy = (beats**2).sum(axis=1)
        width_proxy = (np.abs(np.diff(beats, axis=1)).sum(axis=1))
        return np.c_[peak, trough, energy, width_proxy]
