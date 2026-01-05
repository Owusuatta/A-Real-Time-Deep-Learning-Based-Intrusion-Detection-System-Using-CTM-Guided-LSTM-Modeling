import numpy as np
from ids.stream_ids import SlidingWindowIDS

X = np.load("analysis/saved_outputs/X_sequences.npy")  # (N, 30, 19)

ids = SlidingWindowIDS()

alerts = 0
for seq in X:
    for row in seq:
        result = ids.update(row)
        if result and result["anomaly"]:
            alerts += 1

print("Total alerts:", alerts)
