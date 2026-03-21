"""
predict.py
----------
Minimal inference script – mimics what will run on the ESP32:
  1. Maintain a rolling window of the last WINDOW samples.
  2. Feed the window to the model and output ON / OFF.

Data must already be Min-Max normalised (run normalize_data.py first).
Input CSV columns (with header):
    angle, Pitch1, PitchRate1, Pitch2, PitchRate2, label

Run as a standalone demo:
    python predict.py --model cnn --file DATA_NORMALIZED/2026-03-21T09-57-02-714Z.csv

The script streams through the chosen file sample-by-sample,
exactly as the microcontroller would receive data in real time.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from data_loader import WINDOW, FEATURES, COLUMNS, RENAME_MAP

OUT_DIR = os.path.dirname(__file__)


def stream_predict(model_name: str, txt_file: str, threshold: float = 0.6):
    # Load model from  NN/<model_name>/  folder
    model_dir  = os.path.join(OUT_DIR, model_name)
    model_path = os.path.join(model_dir, "best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final.keras")
    model = keras.models.load_model(model_path)
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold}")
    print(f"Streaming: {txt_file}\n")
    print(f"{'Step':>6}  {'P(ON)':>7}  {'Pred':>5}  {'True':>5}  {'Match':>5}")
    print("-" * 40)

    # Rolling buffer
    buffer = np.zeros((WINDOW, len(FEATURES)), dtype=np.float32)
    buf_idx = 0
    correct = 0
    total   = 0

    # Read CSV – works with both raw files and already-normalised files
    df_raw = pd.read_csv(txt_file)
    df = df_raw.rename(columns=RENAME_MAP)   # raw headers → internal names
    df_features = df[COLUMNS].dropna()

    # Align index so we can write predictions back to df_raw
    valid_idx = df_features.index.tolist()
    predictions = np.zeros(len(df_raw), dtype=np.float32)   # 0 for warm-up rows

    for pos, (orig_idx, row) in enumerate(df_features.iterrows()):
        features   = row[FEATURES].values.astype(np.float32)
        true_label = int(row["label"])

        # Shift buffer and insert new sample
        buffer[:-1] = buffer[1:]
        buffer[-1]  = features
        buf_idx    += 1

        # Only predict once the buffer is full
        if buf_idx < WINDOW:
            continue

        prob = float(model.predict(
            buffer[np.newaxis, ...], verbose=0)[0, 0])
        predictions[orig_idx] = prob          # store raw probability

        pred = 1 if prob >= threshold else 0
        match = "✓" if pred == true_label else "✗"
        correct += (pred == true_label)
        total   += 1

        if total % 20 == 0 or pred == 1 or true_label == 1:
            print(f"{buf_idx:>6}  {prob:>7.3f}  "
                  f"{'ON' if pred else 'OFF':>5}  "
                  f"{'ON' if true_label else 'OFF':>5}  "
                  f"{match:>5}")

    acc = correct / total if total else 0
    print(f"\nAccuracy on streamed file: {acc*100:.1f}%  ({correct}/{total})")

    # Write predicted probabilities back into the Prediction column and save
    # Output file:  original_name_predicted.csv  (original file untouched)
    base, ext = os.path.splitext(txt_file)
    out_path = f"{base}_predicted{ext}"
    df_raw["Prediction"] = predictions
    df_raw.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Predictions saved → {out_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_path = f"{base}_predicted.png"
    _plot_results(df_raw, predictions, threshold, plot_path)
    print(f"Plot saved        → {plot_path}")


def _plot_results(df_raw: pd.DataFrame, predictions: np.ndarray,
                  threshold: float, plot_path: str):
    """Plot Angle, Pitch1, Pitch2, label and prediction in stacked subplots."""
    # Map raw column names if needed (df_raw may still have original headers)
    df = df_raw.rename(columns=RENAME_MAP)

    t = np.arange(len(df))

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    def _plot(ax, values, title, color, fill=False):
        ax.plot(t, values, color=color, lw=1)
        if fill:
            ax.fill_between(t, values, alpha=0.25, color=color)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)

    _plot(axes[0], df["angle"].values,   "Angle",      "steelblue")
    _plot(axes[1], df["Pitch1"].values,  "Pitch 1",    "darkorange")
    _plot(axes[2], df["Pitch2"].values,  "Pitch 2",    "green")
    _plot(axes[3], df["label"].values,   "Label (ON/OFF)", "crimson",  fill=True)

    # Prediction: raw probability + threshold line
    axes[4].plot(t, predictions, color="purple", lw=1, label="P(ON)")
    axes[4].axhline(threshold, color="gray", linestyle="--", lw=0.8,
                    label=f"threshold {threshold}")
    axes[4].fill_between(t, predictions, threshold,
                         where=(predictions >= threshold),
                         alpha=0.25, color="purple", label="predicted ON")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("Prediction", fontsize=9)
    axes[4].set_xlabel("Sample index")
    axes[4].legend(fontsize=8, loc="upper right")
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(os.path.basename(plot_path).replace("_predicted.png", ""),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "cnn"], default="cnn")
    parser.add_argument("--file",  default=os.path.join("DATA", "c:\\Users\\vaclav.heidler\\Downloads\\2026-03-21T15-10-41-489Z.csv"))
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    stream_predict(args.model, args.file, args.threshold)


