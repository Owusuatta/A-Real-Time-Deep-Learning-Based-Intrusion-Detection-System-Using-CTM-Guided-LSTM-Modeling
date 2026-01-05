import tensorflow as tf
import numpy as np
from pathlib import Path
from ids.model_def import build_ids_model


# ===============================
# CONFIG
# ===============================

THRESHOLD = 1e-4
WINDOW = 30
N_FEATURES = 19
N_CLASSES = 5


# ===============================
# PATHS
# ===============================
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "notebook" / "ctm_lstm_ids_model.keras"
CTM_PATH = BASE_DIR / "notebook" / "ctm_weights.npy"


# ===============================
# LOAD CTM
# ===============================
ctm_vector = np.load(CTM_PATH).astype("float32")  # (19,)


# ===============================
# BUILD MODEL & LOAD WEIGHTS
# ===============================
model = build_ids_model(WINDOW, N_FEATURES, N_CLASSES)

# Load weights only (bypasses Lambda completely)
model.load_weights(MODEL_PATH)


# ===============================
# PREPROCESS
# ===============================
def prepare_sequence(x_seq: np.ndarray) -> np.ndarray:
    if x_seq.shape != (WINDOW, N_FEATURES):
        raise ValueError(f"Expected {(WINDOW, N_FEATURES)}, got {x_seq.shape}")

    x = x_seq.astype("float32")
    x = x * ctm_vector          # CTM applied OUTSIDE model
    return np.expand_dims(x, axis=0)


# ===============================
# MULTICLASS PREDICTION
# ===============================
def predict_multiclass(x_seq: np.ndarray):
    x = prepare_sequence(x_seq)
    probs = model.predict(x, verbose=0)[0]
    label = int(np.argmax(probs))
    return label, probs


# ===============================
# ANOMALY DETECTION
# ===============================
def predict_anomaly(x_seq: np.ndarray, threshold=0.5):
    _, probs = predict_multiclass(x_seq)
    anomaly_prob = 1.0 - probs[0]
    return int(anomaly_prob >= threshold), anomaly_prob


def predict_anomaly(x):
    probs = model.predict(x, verbose=0)[0]
    score = 1.0 - probs[0]
    is_anomaly = int(score >= THRESHOLD)
    return is_anomaly, score
