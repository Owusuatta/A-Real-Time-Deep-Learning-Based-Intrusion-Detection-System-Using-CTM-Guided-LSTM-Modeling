from ids.stream_ids import SlidingWindowIDS

# Create IDS engine once
ids_engine = SlidingWindowIDS()


def process_feature_row(feature_row):
    """
    feature_row: ndarray shape (19,)
    """
    result = ids_engine.update(feature_row)

    # Window not full yet
    if result is None:
        return None

    # ALERT LOGIC (this is correct here)
    if result["anomaly"]:
        print(
            f"[ALERT] score={result['score']:.6f} "
            f"class={result['class']}"
        )

    return result
