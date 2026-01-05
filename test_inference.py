import numpy as np
from ids.inference import predict_multiclass, predict_anomaly

# fake input (replace later with real stream)
x_dummy = np.random.rand(30, 19).astype("float32")

label, probs = predict_multiclass(x_dummy)
is_anom, score = predict_anomaly(x_dummy)

print("Class:", label)
print("Probabilities:", probs)
print("Anomaly:", is_anom, "Score:", score)
