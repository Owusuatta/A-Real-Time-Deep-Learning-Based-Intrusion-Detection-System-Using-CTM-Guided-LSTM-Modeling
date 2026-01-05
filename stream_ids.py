import numpy as np
from collections import deque
import json
from datetime import datetime
from pathlib import Path

from ids.inference import predict_multiclass
from ids.thresholds import is_anomaly


# ===============================
# CONFIG
# ===============================
WINDOW = 30
N_FEATURES = 19
ALERT_LOG = Path("dashboard/alerts.jsonl")


# ===============================
# ALERT PERSISTENCE
# ===============================
def save_alert(result: dict):
    ALERT_LOG.parent.mkdir(exist_ok=True)

    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "class": int(result["class"]),
        "score": float(result["score"]),
    }

    with ALERT_LOG.open("a") as f:
        f.write(json.dumps(alert) + "\n")


# ===============================
# STREAMING IDS ENGINE
# ===============================
class SlidingWindowIDS:
    def __init__(self):
        self.buffer = deque(maxlen=WINDOW)

    def update(self, feature_row):
        """
        feature_row: ndarray shape (19,)
        Returns:
            None (if window not full yet)
            dict {class, score, anomaly}
        """

        # Validate input
        feature_row = np.asarray(feature_row, dtype="float32")
        if feature_row.shape != (N_FEATURES,):
            raise ValueError(
                f"Expected {(N_FEATURES,)}, got {feature_row.shape}"
            )

        # Append to sliding window
        self.buffer.append(feature_row)

        # Not enough data yet
        if len(self.buffer) < WINDOW:
            return None

        # Build sequence (30, 19)
        x_seq = np.stack(self.buffer)

        # ---- MODEL INFERENCE ----
        label, probs = predict_multiclass(x_seq)

        # ---- BINARY ANOMALY SCORE ----
        score = 1.0 - probs[0]  # class 0 = normal
        anomaly = is_anomaly(score)

        result = {
            "class": label,
            "score": float(score),
            "anomaly": anomaly
        }

        # ---- SAVE ALERT (SIDE EFFECT) ----
        if anomaly:
            save_alert(result)

        return result
