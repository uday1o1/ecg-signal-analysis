from sklearn.ensemble import IsolationForest

class AnomalyIForest:
    def __init__(self, **kwargs):
        self.clf = IsolationForest(n_estimators=200, contamination='auto', **kwargs)

    def fit(self, X):
        self.clf.fit(X); return self

    def predict_scores(self, X):
        return -self.clf.score_samples(X)

    def predict(self, X, threshold=None):
        scores = self.predict_scores(X)
        if threshold is None:
            thr = np.percentile(scores, 95)
        else:
            thr = threshold
        return (scores >= thr).astype(int)
