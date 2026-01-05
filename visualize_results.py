import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "analysis" / "saved_outputs"
FIG_DIR = BASE_DIR / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Attack-1", "Attack-2", "Attack-3", "Attack-4"]

# ===============================
# LOAD DATA
# ===============================
y_true = np.load(DATA_DIR / "y_true.npy")
y_pred = np.load(DATA_DIR / "y_pred.npy")
y_prob = np.load(DATA_DIR / "y_prob.npy")
anomaly_score = np.load(DATA_DIR / "anomaly_score.npy")

# ===============================
# 1. MULTICLASS CONFUSION MATRIX (SAFE)
# ===============================
labels = np.unique(np.concatenate([y_true, y_pred]))

class_names = [f"Class-{i}" for i in labels]

cm = confusion_matrix(y_true, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Multiclass Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix_multiclass.png", dpi=200)
plt.close()


# ===============================
# 2. ANOMALY ROC CURVE
# ===============================
y_true_bin = (y_true != 0).astype(int)

fpr, tpr, _ = roc_curve(y_true_bin, anomaly_score)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Anomaly Detection ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_anomaly.png", dpi=200)
plt.close()

# ===============================
# 3. PRECISION–RECALL CURVE
# ===============================
precision, recall, _ = precision_recall_curve(y_true_bin, anomaly_score)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Anomaly Detection Precision–Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIG_DIR / "pr_anomaly.png", dpi=200)
plt.close()

# ===============================
# 4. ANOMALY SCORE DISTRIBUTION
# ===============================
plt.hist(anomaly_score[y_true_bin == 0], bins=100, alpha=0.6, label="Normal")
plt.hist(anomaly_score[y_true_bin == 1], bins=100, alpha=0.6, label="Attack")
plt.xlabel("Anomaly Score (1 - P(normal))")
plt.ylabel("Count")
plt.title("Anomaly Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "anomaly_score_hist.png", dpi=200)
plt.close()

print("All figures saved to:", FIG_DIR)
