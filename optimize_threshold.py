import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve

# Load saved outputs
y_true = np.load("analysis/saved_outputs/y_true.npy")
scores = np.load("analysis/saved_outputs/anomaly_score.npy")

# Convert labels: 0 = normal, 1 = attack
y_binary = (y_true != 0).astype(int)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_binary, scores)

results = []

for thr in thresholds:
    y_pred = (scores >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_binary, y_pred, average="binary", zero_division=0
    )
    results.append((thr, p, r, f1))

results = np.array(results, dtype=object)

# Best thresholds
best_f1 = max(results, key=lambda x: x[3])
high_recall = max(results, key=lambda x: x[2])
low_fpr_idx = np.where(fpr <= 0.01)[0]
secure_thr = thresholds[low_fpr_idx[-1]
                        ] if len(low_fpr_idx) else thresholds[-1]

print("\n=== THRESHOLD OPTIONS ===")
print(
    f"Best F1 Threshold: {best_f1[0]:.6f} | P={best_f1[1]:.3f} R={best_f1[2]:.3f}")
print(f"Max Recall Threshold: {high_recall[0]:.6f} | R={high_recall[2]:.3f}")
print(f"Security Threshold (FPR<=1%): {secure_thr:.6f}")

np.save("analysis/saved_outputs/optimized_thresholds.npy", results)
