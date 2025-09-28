import numpy as np

class BeatLabelAligner:

    def align_by_counts(self, labels_list, counts):
        ys = []
        for y, c in zip(labels_list, counts):
            ys.append(y[:c])
        return np.concatenate(ys, axis=0)


    def align_detect_to_ann(self, detected_list, ann_list, labels_list, tol_samples):
        ys = []
        for det, ann, lab in zip(detected_list, ann_list, labels_list):
            i = j = 0
            matched = []
            while i < len(det) and j < len(ann):
                if abs(det[i] - ann[j]) <= tol_samples:
                    matched.append(lab[j]); i += 1; j += 1
                elif det[i] < ann[j]:
                    i += 1
                else:
                    j += 1
            ys.append(np.array(matched, dtype=int))
        return np.concatenate(ys, axis=0)
