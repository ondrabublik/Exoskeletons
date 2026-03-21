"""
evaluate.py
-----------
Loads a saved model, runs it on the test set, prints detailed metrics
and saves diagnostic plots:
    - Training history  (loss, accuracy, AUC)
    - Confusion matrix
    - ROC curve
    - Precision-Recall curve
    - Per-sample prediction timeline (first test file)

Usage:
    python evaluate.py --model dense
    python evaluate.py --model cnn
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report,
)
import tensorflow as tf
from tensorflow import keras

from data_loader import build_dataset, WINDOW

OUT_DIR = os.path.dirname(__file__)
# plots and models live inside  NN/<name>/


# ── helpers ─────────────────────────────────────────────────────────────────
def plot_history(history, name: str, plot_dir: str):
    metrics = ["loss", "accuracy", "auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, m in zip(axes, metrics):
        if m not in history.history:
            continue
        ax.plot(history.history[m],      label="train")
        ax.plot(history.history[f"val_{m}"], label="val")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(f"{name} – training history")
    fig.tight_layout()
    path = os.path.join(plot_dir, "history.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confusion(y_true, y_pred_bin, name: str, plot_dir: str):
    cm = confusion_matrix(y_true, y_pred_bin)
    disp = ConfusionMatrixDisplay(cm, display_labels=["OFF", "ON"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} – confusion matrix (test set)")
    path = os.path.join(plot_dir, "confusion.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc_pr(y_true, y_score, name: str, plot_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap           = average_precision_score(y_true, y_score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{name} – ROC curve")
    ax1.legend()

    ax2.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{name} – Precision-Recall curve")
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(plot_dir, "roc_pr.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_predictions(y_true, y_score, name: str, plot_dir: str, n_samples: int = 300):
    """Show raw model output vs ground-truth label over time."""
    idx = np.arange(min(n_samples, len(y_true)))
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(idx, y_true[idx], alpha=0.3, color="green",
                    label="Ground truth ON")
    ax.plot(idx, y_score[idx], color="red", lw=1, label="Model output")
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, label="Threshold 0.5")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Window index")
    ax.set_ylabel("P(ON)")
    ax.set_title(f"{name} – prediction timeline (first {n_samples} test windows)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, "timeline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ── main ────────────────────────────────────────────────────────────────────
def evaluate(name: str):
    # Per-model folder: NN/cnn/  or  NN/dense/
    model_dir = os.path.join(OUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # Load data (same seed → identical test split)
    _, _, (X_test, y_test) = build_dataset()

    # Load model – prefer best checkpoint, fall back to final
    model_path = os.path.join(model_dir, "best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final.keras")
    print(f"Loading {model_path} …")
    model = keras.models.load_model(model_path)

    # Predict
    y_score = model.predict(X_test, batch_size=64, verbose=0).ravel()
    y_pred  = (y_score >= 0.5).astype(int)

    # Text report
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["OFF", "ON"],
                                 zero_division=0))

    # Plots – all go into the model folder
    plot_confusion(y_test, y_pred, name, model_dir)
    plot_roc_pr(y_test, y_score, name, model_dir)
    plot_predictions(y_test, y_score, name, model_dir)

    print(f"\nAll plots saved to: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "cnn"], default="cnn",
                        help="Which saved model to evaluate.")
    args = parser.parse_args()
    evaluate(args.model)

