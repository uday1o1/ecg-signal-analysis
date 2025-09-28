# ecg/align.py
import numpy as np

class BeatLabelAligner:
    """
    Given original R locations & labels for each record, align them to the
    segmented beats produced by BeatSegmenter (same R order assumed).
    """
    def align(self, rpeaks_list, labels_list, seg_counts):
        # rpeaks_list: list of arrays (one per record) in chronological order
        # labels_list: list of arrays (one per record), same length as rpeaks_list entries
        # seg_counts: list of ints: how many segments kept per record
        y=[]
        for r,l,c in zip(rpeaks_list, labels_list, seg_counts):
            # We dropped segments near edges; keep only those that survived
            # Assume first c beats correspond to first c R-peaks for this record
            y.append(l[:c])
        return np.concatenate(y, axis=0)
