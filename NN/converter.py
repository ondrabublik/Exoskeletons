"""
converter.py
------------
Converts a saved Keras model (.h5) to:
  - <name>.tflite   – TensorFlow Lite flatbuffer
  - <name>.h        – C header for embedding on ESP32

Can be used as a module (convert_model) or run standalone:
    python converter.py --model cnn_final.h5 --out model
"""

import argparse
import os
import tensorflow as tf


def convert_model(h5_path: str, out_dir: str | None = None) -> tuple[str, str]:
    """
    Convert a .h5 Keras model to TFLite and a C header file.

    Parameters
    ----------
    h5_path : str
        Path to the source .h5 model file.
    out_dir : str, optional
        Directory where output files are written.
        Defaults to the same directory as h5_path.

    Returns
    -------
    tflite_path, header_path : tuple[str, str]
    """
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(h5_path))

    base = os.path.splitext(os.path.basename(h5_path))[0]   # e.g. "cnn_final"
    tflite_path = os.path.join(out_dir, f"{base}.tflite")
    header_path = os.path.join(out_dir, f"{base}.h")

    # ── 1. Convert to TFLite ───────────────────────────────────────────────
    print(f"Converting {h5_path} → {tflite_path} …")
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enable for int8
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite saved → {tflite_path}")

    # ── 2. Write C header ──────────────────────────────────────────────────
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("#pragma once\n\n")
        array_name = base.replace("-", "_")
        f.write(f"const unsigned char {array_name}_tflite[] = {{\n")
        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if i % 12 == 11:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int {array_name}_tflite_len = {len(data)};\n")
    print(f"C header saved → {header_path}")

    return tflite_path, header_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Keras .h5 → TFLite + C header")
    parser.add_argument("--model", default="cnn_final.h5",
                        help="Path to the .h5 model file (default: cnn_final.h5)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same as model file)")
    args = parser.parse_args()
    convert_model(args.model, args.out)
