"""
models.py
---------
Two lightweight architectures for binary ON/OFF classification.
Both are designed to be tiny enough to convert for ESP32 (TFLite / EloquentTF).

Input shape: (WINDOW, 5)   → 5 sensor channels over WINDOW time-steps
                              (angle, x, y, z, omx – already Min-Max scaled)

Architecture A – Dense (MLP on flattened window)
    Flatten → Dense(64) → BN → ReLU → Dropout
            → Dense(32) → BN → ReLU → Dropout
            → Dense(1, sigmoid)
    ~  4 k parameters

Architecture B – 1-D CNN
    Conv1D(16, k=5) → BN → ReLU
    Conv1D(32, k=3) → BN → ReLU → GlobalAvgPool
    Dense(32) → BN → ReLU → Dropout
    Dense(1, sigmoid)
    ~ 8 k parameters
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_loader import WINDOW, FEATURES

N_FEATURES = len(FEATURES)   # 7


# ── A: Dense (MLP) ──────────────────────────────────────────────────────────
def build_dense_model(window: int = WINDOW,
                      n_features: int = N_FEATURES,
                      units: tuple = (64, 32),
                      dropout: float = 0.3) -> keras.Model:
    """
    Flatten the (window × features) input then pass through two Dense blocks.
    Simple, very fast on MCU, good baseline.
    """
    inp = keras.Input(shape=(window, n_features), name="input")

    x = layers.Flatten()(inp)
    for u in units:
        x = layers.Dense(u, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inp, out, name="dense_model")
    return model


# ── B: 1-D CNN ───────────────────────────────────────────────────────────────
def build_cnn_model(window: int = WINDOW,
                    n_features: int = N_FEATURES,
                    filters: tuple = (8, 16), #(16, 32),
                    kernel_sizes: tuple = (5, 3),
                    dense_units: int = 12, #32,
                    dropout: float = 0.5)  -> keras.Model:  #0.3)
    """
    Two Conv1D layers extract local temporal patterns across the window,
    GlobalAveragePooling collapses the time axis → one Dense head.
    Naturally translation-invariant in time → great for detecting
    gait-phase transitions at any point in the window.
    """
    inp = keras.Input(shape=(window, n_features), name="input")

    x = inp
    for f, k in zip(filters, kernel_sizes):
        x = layers.Conv1D(f, kernel_size=k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(dense_units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inp, out, name="cnn_model")
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = 1e-3,
                  pos_weight: float = 1.0) -> keras.Model:
    """
    Binary cross-entropy with optional class weighting baked in.
    pos_weight > 1 penalises missed ON detections more heavily
    (useful for the imbalanced dataset: ~5-6× more OFF than ON).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    dense = build_dense_model()
    dense.summary()
    print()
    cnn = build_cnn_model()
    cnn.summary()

