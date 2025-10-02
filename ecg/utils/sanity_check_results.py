import numpy as np
y_prob = np.load("outputs/preds.npy")
y_pred = np.load("outputs/preds_labels.npy")

print("Shapes:", y_prob.shape, y_pred.shape)
print("First 5 probability rows:\n", y_prob[:5])
print("First 5 binarized rows:\n", y_pred[:5])
print("Positive counts per class:", y_pred.sum(axis=0))
