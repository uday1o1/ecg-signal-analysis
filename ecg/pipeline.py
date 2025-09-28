# ecg/pipeline.py
from sklearn.pipeline import Pipeline
from .preprocess import ECGPreprocessor
from .peaks import RPeakDetector
from .segment import BeatSegmenter
from .features.base import MorphologyFeatures
from .models.classify import make_baseline_classifier

def make_classification_workflow(fs_in, fs_out=250):
    return Pipeline([
        ("pre", ECGPreprocessor(fs_in=fs_in, fs_out=fs_out)),
        ("peaks", RPeakDetector(fs=fs_out)),
        ("seg", BeatSegmenter(fs=fs_out)),
        ("feat", MorphologyFeatures()),
        ("clf", make_baseline_classifier())
    ])
