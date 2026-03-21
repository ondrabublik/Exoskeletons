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
    """
    4-panel stacked plot sharing the x-axis:
      1. Angle
      2. Pitch 1
      3. Pitch 2
      4. Overlay: Angle (left axis) + Label filled + Prediction probability
                  with threshold line (right axis)
    """
    df = df_raw.rename(columns=RENAME_MAP)
    t  = np.arange(len(df))

    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(4, 1, hspace=0.08)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)   # overlay panel

    # ── helper ────────────────────────────────────────────────────────────
    def _simple(ax, values, ylabel, color):
        ax.plot(t, values, color=color, lw=1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelbottom=False)
        ax.grid(True, alpha=0.25, linestyle=":")

    _simple(ax0, df["angle"].values,  "Angle",   "#2196F3")
    _simple(ax1, df["Pitch1"].values, "Pitch 1", "#FF9800")
    _simple(ax2, df["Pitch2"].values, "Pitch 2", "#4CAF50")

    # ── overlay panel ─────────────────────────────────────────────────────
    label_vals = df["label"].values.astype(float)
    angle_vals = df["angle"].values

    # background: soft green fill where label == ON
    ax3.fill_between(t, 0, 1,
                     where=(label_vals > 0.5),
                     transform=ax3.get_xaxis_transform(),
                     alpha=0.20, color="#4CAF50", label="Label ON (ground truth)")

    # angle on left axis
    ax3.plot(t, angle_vals, color="#2196F3", lw=1.2, alpha=0.9, label="Angle")
    ax3.set_ylabel("Angle", fontsize=9, color="#2196F3")
    ax3.tick_params(axis="y", colors="#2196F3")

    # prediction probability on right axis
    ax3_r = ax3.twinx()
    ax3_r.plot(t, predictions, color="#9C27B0", lw=1.5, alpha=0.9, label="P(ON) predicted")
    ax3_r.fill_between(t, 0, predictions, alpha=0.15, color="#9C27B0")
    ax3_r.set_ylim(-0.05, 1.05)
    ax3_r.set_ylabel("P(ON)", fontsize=9, color="#9C27B0")
    ax3_r.tick_params(axis="y", colors="#9C27B0")

    ax3.set_xlabel("Sample index", fontsize=9)
    ax3.grid(True, alpha=0.25, linestyle=":")

    # combined legend from both axes
    lines_l, labels_l = ax3.get_legend_handles_labels()
    lines_r, labels_r = ax3_r.get_legend_handles_labels()
    ax3.legend(lines_l + lines_r, labels_l + labels_r,
               fontsize=8, loc="upper right", framealpha=0.85)

    # hide x tick labels on top 3 panels
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    title = os.path.basename(plot_path).replace("_predicted.png", "")
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "cnn"], default="cnn")
    parser.add_argument("--file",  default=os.path.join("DATA", "c:\\Users\\vaclav.heidler\\Downloads\\2026-03-21T15-10-41-489Z_predicted.csv"))
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    stream_predict(args.model, args.file, args.threshold)


