import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MorphologyFeatures(BaseEstimator, TransformerMixin):
    """
    Minimal, fast beat-level features that work well on MIT-BIH:
      - peak amplitude
      - trough amplitude
      - energy (L2)
      - total variation (L1 of first difference)
      - width proxy (samples above 20% dynamic range)
    These are intentionally simple; scaling happens later in the sklearn pipeline.
    """
    def __init__(self, width_threshold=0.2, enforce_fixed_len=True):
        self.width_threshold = width_threshold
        self.enforce_fixed_len = enforce_fixed_len

    def fit(self, X, y=None):
        return self

    def transform(self, beats):
        # beats: ndarray [n_beats, n_samples], fixed-length windows
        beats = np.asarray(beats)
        if self.enforce_fixed_len and beats.ndim != 2:
            raise ValueError("Expected 2D array [n_beats, n_samples].")

        peak = beats.max(axis=1)
        trough = beats.min(axis=1)
        energy = (beats ** 2).sum(axis=1)
        l1 = np.abs(np.diff(beats, axis=1)).sum(axis=1)

        # Width proxy: count consecutive samples above (trough + t*(peak-trough))
        thr = (self.width_threshold * (peak - trough) + trough)[:, None]
        above = beats >= thr
        # leftmost True index
        left = above.argmax(axis=1)
        # rightmost True index (flip and find first True)
        right = beats.shape[1] - np.flip(above, axis=1).argmax(axis=1)
        width = right - left

        return np.c_[peak, trough, energy, l1, width].astype(np.float32)
