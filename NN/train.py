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
import matplotlib
matplotlib.use("Agg")   # no display needed – saves PNG files
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_loader import build_dataset, WINDOW
from models import build_dense_model, build_cnn_model, compile_model
from converter import convert_model

# ── CONFIG ──────────────────────────────────────────────────────────────────
EPOCHS        = 2000
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
BASE_DIR = os.path.dirname(__file__)   # NN/
# Each model gets its own subfolder: NN/cnn/  or  NN/dense/
# ────────────────────────────────────────────────────────────────────────────


def plot_training(history: keras.callbacks.History, name: str, out_dir: str):
    """
    Save two diagnostic plots after training:

    1. loss / accuracy / AUC  – train vs val  (spot under/over-fitting)
    2. train-val GAP over epochs               (early-stop / overfit indicator)
       • gap > 0 and growing → overfitting
       • both curves high     → underfitting
       • both curves low & close → good fit
    """
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    # ── Plot 1: three-panel metric curves ─────────────────────────────────
    metrics = [
        ("loss",     "Loss",     "lower is better"),
        ("accuracy", "Accuracy", "higher is better"),
        ("auc",      "AUC",      "higher is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title, note) in zip(axes, metrics):
        if key not in h:
            ax.set_visible(False)
            continue
        ax.plot(epochs, h[key],           lw=2, label="train")
        ax.plot(epochs, h[f"val_{key}"],  lw=2, label="val", linestyle="--")
        ax.set_title(f"{title}\n({note})")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name.upper()} – training curves", fontsize=13)
    fig.tight_layout()
    path1 = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"Plot saved → {path1}")

    # ── Plot 2: train-val gap (overfitting indicator) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    loss_gap = np.array(h["val_loss"]) - np.array(h["loss"])
    axes[0].plot(epochs, loss_gap, color="crimson", lw=2)
    axes[0].axhline(0, color="gray", linestyle="--", lw=0.8)
    axes[0].fill_between(epochs, loss_gap, 0,
                         where=(loss_gap > 0), alpha=0.15, color="crimson",
                         label="val > train (overfit risk)")
    axes[0].fill_between(epochs, loss_gap, 0,
                         where=(loss_gap <= 0), alpha=0.15, color="steelblue",
                         label="train > val (underfit / noise)")
    axes[0].set_title("Loss gap  (val_loss − train_loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if "auc" in h:
        auc_gap = np.array(h["auc"]) - np.array(h["val_auc"])
        axes[1].plot(epochs, auc_gap, color="darkorange", lw=2)
        axes[1].axhline(0, color="gray", linestyle="--", lw=0.8)
        axes[1].fill_between(epochs, auc_gap, 0,
                             where=(auc_gap > 0), alpha=0.15, color="darkorange",
                             label="train AUC > val (overfit risk)")
        axes[1].set_title("AUC gap  (train_auc − val_auc)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle(f"{name.upper()} – overfitting diagnostic", fontsize=13)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "overfit_gap.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"Plot saved → {path2}")


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


def get_callbacks(name: str, out_dir: str) -> list:
    ckpt_path    = os.path.join(out_dir, "best.keras")
    ckpt_path_h5 = os.path.join(out_dir, "best.h5")
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path_h5,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,          # silent – same checkpoint, just different format
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=100,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=30,
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

    out_dir = os.path.join(BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=get_callbacks(name, out_dir),
        verbose=1,
    )

    # ── Evaluate on held-out test set ──────────────────────────────────────
    print(f"\n── {name.upper()} TEST RESULTS ──")
    results = model.evaluate(X_test, y_test, verbose=0)
    for mname, val in zip(model.metrics_names, results):
        print(f"  {mname}: {val:.4f}")

    # ── Save final weights ─────────────────────────────────────────────────
    keras_path = os.path.join(out_dir, "final.keras")
    model.save(keras_path)
    print(f"Model saved → {keras_path}")

    h5_path = os.path.join(out_dir, "final.h5")
    model.save(h5_path)
    print(f"Model saved → {h5_path}  (HDF5)")

    # ── Convert final model → model.tflite + model.h ──────────────────────
    convert_model(h5_path, out_dir)

    # ── Also convert best checkpoint if present ───────────────────────────
    best_h5_path = os.path.join(out_dir, "best.h5")
    if os.path.exists(best_h5_path):
        best_out_dir = os.path.join(out_dir, "best")
        convert_model(best_h5_path, best_out_dir)

    # ── SavedModel export ─────────────────────────────────────────────────
    saved_model_dir = os.path.join(out_dir, "saved_model")
    model.export(saved_model_dir)
    print(f"SavedModel  → {saved_model_dir}")

    # ── Training plots ─────────────────────────────────────────────────────
    plot_training(history, name, out_dir)

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

