# This value comes from your optimization output
ANOMALY_THRESHOLD = 0.000001


def is_anomaly(score: float) -> int:
    return int(score >= ANOMALY_THRESHOLD)


THRESHOLD = 0.20  # start conservative


def is_anomaly(score: float) -> bool:
    return score >= THRESHOLD
