import numpy as np
import time

from ids.stream_ids import SlidingWindowIDS


# ===============================
# SIMULATION CONFIG
# ===============================
N_FEATURES = 19
SLEEP_SEC = 0.05   # controls stream speed


# ===============================
# INIT IDS ENGINE
# ===============================
ids = SlidingWindowIDS()


# ===============================
# SIMULATED FEATURE STREAM
# ===============================
def generate_feature_row():
    """
    Simulate one network flow feature vector (19 features)
    Replace this later with real extractor output
    """
    return np.random.rand(N_FEATURES).astype("float32")


# ===============================
# STREAM LOOP
# ===============================
def run_stream():
    print("[INFO] Starting live IDS stream...")

    while True:
        feature_row = generate_feature_row()
        result = ids.update(feature_row)

        if result is not None and result["anomaly"]:
            print(
                f"[ALERT] score={result['score']:.6f} "
                f"class={result['class']}"
            )

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    run_stream()
