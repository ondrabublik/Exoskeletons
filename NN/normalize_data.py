"""
normalize_data.py
-----------------
Reads all TXT files from DATA/, normalises every feature column to [0, 1]
using Min-Max scaling (fitted across ALL files combined so the scale is
consistent), keeps only the first 5 feature columns:

    angle, x, y, z, omx        ← kept  (columns 0-4)
    omy, omz                    ← dropped (columns 5-6)
    label (0/1)                 ← kept as-is (column 7 → new column 5)

Output files are written to DATA_NORMALIZED/ with the same filenames.
Each output row:  angle_n, x_n, y_n, z_n, omx_n, label

A 'scaler_minmax.pkl' is also saved so you can apply the exact same
scaling to live sensor data on the ESP32 (just store min/max arrays).
"""

import os
import numpy as np
import pandas as pd
import joblib

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "DATA")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "DATA_NORMALIZED")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "scaler_minmax.pkl")

# Original columns in the TXT files
ALL_COLUMNS  = ["angle", "x", "y", "z", "omx", "omy", "omz", "label"]

# Which feature columns to KEEP (first 5)
KEEP_FEATURES = ["angle", "x", "y", "z", "omx"]

# Output column order
OUT_COLUMNS  = KEEP_FEATURES + ["label"]
# ─────────────────────────────────────────────────────────────────────────────


def load_all(data_dir: str) -> list[tuple[str, pd.DataFrame]]:
    """Return list of (filename, DataFrame) for every TXT file."""
    result = []
    txt_files = sorted(f for f in os.listdir(data_dir) if f.upper().endswith(".TXT"))
    if not txt_files:
        raise FileNotFoundError(f"No .TXT files found in {data_dir}")
    for fname in txt_files:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, header=None, names=ALL_COLUMNS,
                         skipinitialspace=True)
        df = df.dropna()
        result.append((fname, df))
        print(f"  Read {fname}: {len(df)} rows, "
              f"ON={int(df['label'].sum())}, OFF={int((df['label']==0).sum())}")
    return result


def fit_minmax(all_data: list[tuple[str, pd.DataFrame]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute global min/max for KEEP_FEATURES across all files combined."""
    combined = pd.concat([df for _, df in all_data], ignore_index=True)
    col_min = combined[KEEP_FEATURES].min().values.astype(np.float32)
    col_max = combined[KEEP_FEATURES].max().values.astype(np.float32)
    return col_min, col_max


def minmax_scale(df: pd.DataFrame,
                 col_min: np.ndarray,
                 col_max: np.ndarray) -> pd.DataFrame:
    """Apply Min-Max normalisation to KEEP_FEATURES, clip to [0,1]."""
    out = df[KEEP_FEATURES].values.astype(np.float32)
    rng = col_max - col_min
    # Avoid division by zero for constant columns
    rng[rng == 0] = 1.0
    out = (out - col_min) / rng
    out = np.clip(out, 0.0, 1.0)
    result = pd.DataFrame(out, columns=KEEP_FEATURES)
    result["label"] = df["label"].values.astype(int)
    return result


def save_file(df_out: pd.DataFrame, fname: str, out_dir: str):
    """Write normalised DataFrame to CSV without header."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    df_out.to_csv(path, index=False, header=False,
                  float_format="%.6f")
    print(f"  Saved → {path}")


def main():
    print("=" * 60)
    print("Step 1 – Loading data files")
    print("=" * 60)
    all_data = load_all(DATA_DIR)

    print("\n" + "=" * 60)
    print("Step 2 – Computing global Min-Max across all files")
    print("=" * 60)
    col_min, col_max = fit_minmax(all_data)
    for name, mn, mx in zip(KEEP_FEATURES, col_min, col_max):
        print(f"  {name:>8}:  min={mn:.4f}  max={mx:.4f}")

    # Save scaler so ESP32 firmware can apply the same transform
    scaler_info = {"columns": KEEP_FEATURES, "min": col_min, "max": col_max}
    joblib.dump(scaler_info, SCALER_OUT)
    print(f"\nMinMax scaler saved → {SCALER_OUT}")

    print("\n" + "=" * 60)
    print(f"Step 3 – Normalising & saving to {OUT_DIR}/")
    print("=" * 60)
    for fname, df in all_data:
        df_out = minmax_scale(df, col_min, col_max)
        save_file(df_out, fname, OUT_DIR)

    print("\n" + "=" * 60)
    print("Done.  Output columns: " + ", ".join(OUT_COLUMNS))
    print(f"Dropped columns:       omy, omz")
    print("=" * 60)

    # Quick sanity check on one file
    sample_path = os.path.join(OUT_DIR, all_data[0][0])
    sample = pd.read_csv(sample_path, header=None, names=OUT_COLUMNS)
    print("\nSanity check – first 3 rows of", all_data[0][0])
    print(sample.head(3).to_string(index=False))
    print("\nFeature value ranges in normalised file:")
    print(sample[KEEP_FEATURES].agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()

