"""
predict.py
----------
Minimal inference script – mimics what will run on the ESP32:
  1. Maintain a rolling window of the last WINDOW samples.
  2. Feed the window to the model and output ON / OFF.

Data must already be Min-Max normalised (run normalize_data.py first).
Input columns per row: angle, x, y, z, omx, label  (6 columns)

Run as a standalone demo:
    python predict.py --model cnn --file DATA/RPAPAMEO.TXT

The script streams through the chosen file sample-by-sample,
exactly as the microcontroller would receive data in real time.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loader import WINDOW, FEATURES

OUT_DIR = os.path.dirname(__file__)


def stream_predict(model_name: str, txt_file: str, threshold: float = 0.5):
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

    with open(txt_file, "r") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 6:   # angle, x, y, z, omx, label
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue

            features   = np.array(vals[:5], dtype=np.float32)  # already scaled
            true_label = int(vals[5])

            # Shift buffer and insert new sample
            buffer[:-1] = buffer[1:]
            buffer[-1]  = features
            buf_idx    += 1

            # Only predict once the buffer is full
            if buf_idx < WINDOW:
                continue

            prob = float(model.predict(
                buffer[np.newaxis, ...], verbose=0)[0, 0])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "cnn"], default="cnn")
    parser.add_argument("--file",  default=os.path.join("DATA", "RPAPAMEO.TXT"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    stream_predict(args.model, args.file, args.threshold)

