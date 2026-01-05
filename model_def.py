import tensorflow as tf


def build_ids_model(window=30, n_features=19, n_classes=5):
    inputs = tf.keras.Input(shape=(window, n_features), name="input_sequence")

    x = tf.keras.layers.LSTM(64, name="lstm_1")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_4")(x)

    outputs = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="output"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    return model
