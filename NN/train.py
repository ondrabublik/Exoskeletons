"""
train.py
--------
Trains either the Dense or the CNN model (or both).
Saves the best checkpoint as a Keras .keras file and also exports
a TensorFlow SavedModel folder (needed for TFLite conversion).

Usage:
    python train.py --model dense
    python train.py --model cnn
    python train.py --model both          ← default
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loader import build_dataset, WINDOW
from models import build_dense_model, build_cnn_model, compile_model

# ── CONFIG ──────────────────────────────────────────────────────────────────
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
OUT_DIR = os.path.dirname(__file__)
# ────────────────────────────────────────────────────────────────────────────


def compute_class_weight(y_train: np.ndarray) -> dict:
    """Return {0: w0, 1: w1} so both classes contribute equally to the loss."""
    n_total = len(y_train)
    n_pos   = y_train.sum()
    n_neg   = n_total - n_pos
    # standard sklearn formula: total / (n_classes * count)
    w0 = n_total / (2 * n_neg) if n_neg > 0 else 1.0
    w1 = n_total / (2 * n_pos) if n_pos > 0 else 1.0
    print(f"Class weights  OFF={w0:.3f}  ON={w1:.3f}")
    return {0: w0, 1: w1}


def get_callbacks(name: str) -> list:
    ckpt_path = os.path.join(OUT_DIR, f"{name}_best.keras")
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(model: keras.Model, name: str,
                train_data, val_data, test_data,
                class_weight: dict) -> keras.callbacks.History:
    X_train, y_train = train_data
    X_val,   y_val   = val_data
    X_test,  y_test  = test_data

    compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=get_callbacks(name),
        verbose=1,
    )

    # ── Evaluate on held-out test set ──────────────────────────────────────
    print(f"\n── {name.upper()} TEST RESULTS ──")
    results = model.evaluate(X_test, y_test, verbose=0)
    for mname, val in zip(model.metrics_names, results):
        print(f"  {mname}: {val:.4f}")

    # ── Save final weights and SavedModel ─────────────────────────────────
    keras_path = os.path.join(OUT_DIR, f"{name}_final.keras")
    model.save(keras_path)
    print(f"Model saved → {keras_path}")

    saved_model_dir = os.path.join(OUT_DIR, f"{name}_saved_model")
    model.export(saved_model_dir)
    print(f"SavedModel  → {saved_model_dir}")

    return history


def main(which: str = "both"):
    print("=" * 60)
    print("Loading & windowing data …")
    print("=" * 60)
    train_data, val_data, test_data = build_dataset()
    X_train, y_train = train_data
    class_weight = compute_class_weight(y_train)

    histories = {}

    if which in ("dense", "both"):
        print("\n" + "=" * 60)
        print("Training DENSE model …")
        print("=" * 60)
        dense = build_dense_model(window=WINDOW)
        histories["dense"] = train_model(
            dense, "dense", train_data, val_data, test_data, class_weight)

    if which in ("cnn", "both"):
        print("\n" + "=" * 60)
        print("Training CNN model …")
        print("=" * 60)
        cnn = build_cnn_model(window=WINDOW)
        histories["cnn"] = train_model(
            cnn, "cnn", train_data, val_data, test_data, class_weight)

    return histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "cnn", "both"],
                        default="both",
                        help="Which architecture to train.")
    args = parser.parse_args()
    main(args.model)

