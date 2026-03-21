"""
data_loader.py
--------------
Loads all CSV files from the DATA folder, builds windowed
sequences and returns train/val/test splits ready for both Dense and CNN models.

Expected CSV format (with header):
    timestamp,Angle,Pitch 1,Pitch Rate 1,Pitch 2,Pitch Rate 2,Input,Prediction

Column mapping:
    Angle        → angle
    Pitch 1      → Pitch1
    Pitch Rate 1 → PitchRate1
    Pitch 2      → Pitch2
    Pitch Rate 2 → PitchRate2
    Input        → label   (0/1)
    timestamp, Prediction  → ignored
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "DATA")

# Raw CSV columns → internal names (timestamp and Prediction are dropped)
CSV_COLUMNS = ["timestamp", "Angle", "Pitch 1", "Pitch Rate 1",
               "Pitch 2", "Pitch Rate 2", "Input", "Prediction"]
RENAME_MAP  = {
    "Angle":        "angle",
    "Pitch 1":      "Pitch1",
    "Pitch Rate 1": "PitchRate1",
    "Pitch 2":      "Pitch2",
    "Pitch Rate 2": "PitchRate2",
    "Input":        "label",
}

COLUMNS    = ["angle", "Pitch1", "PitchRate1", "Pitch2", "PitchRate2", "label"]
FEATURES   = ["angle", "Pitch1", "PitchRate1", "Pitch2", "PitchRate2"]
WINDOW     = 10      # number of time-steps fed to the network at once
STEP       = 1       # sliding-window stride – use 1 for maximum samples (full overlap)
                     # increase (e.g. 5) only if you have a very large dataset and want faster training
# ────────────────────────────────────────────────────────────────────────────


def load_raw(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Read all CSV files and concatenate them into one DataFrame."""
    frames = []
    csv_files = [f for f in os.listdir(data_dir)
                 if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {data_dir}")

    for fname in sorted(csv_files):
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, skipinitialspace=True)
        # Rename to internal names and keep only needed columns
        df = df.rename(columns=RENAME_MAP)
        df = df[COLUMNS]
        df = df.dropna()
        df["source"] = fname
        frames.append(df)
        print(f"  Loaded {fname}: {len(df)} rows, "
              f"ON={int(df['label'].sum())}, "
              f"OFF={int((df['label'] == 0).sum())}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}, "
          f"ON={int(combined['label'].sum())}, "
          f"OFF={int((combined['label'] == 0).sum())}")
    return combined


def make_windows(X: np.ndarray, y: np.ndarray,
                 window: int = WINDOW, step: int = STEP):
    """
    Slide a window over a single continuous recording.
    Returns:
        Xw  – shape (N, window, n_features)
        yw  – shape (N,)  label = 1 if ANY step in window is ON
    """
    Xw, yw = [], []
    for i in range(0, len(X) - window, step):
        Xw.append(X[i: i + window])
        # label the window as ON if the majority of its steps are ON
        # (change to .any() for more sensitive detection)
        yw.append(1 if y[i: i + window].mean() >= 0.5 else 0)
    return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.int32)


def build_dataset(window: int = WINDOW, step: int = STEP,
                  val_size: float = 0.15, test_size: float = 0.15):
    """
    Full pipeline: load → window → split.
    Data must already be Min-Max normalised (run normalize_data.py first).

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    Each X has shape  (N, window, 5)
    Each y has shape  (N,)
    """
    df = load_raw()

    # ── 1. Extract features and labels (already scaled to [0,1]) ──────────
    X_all = df[FEATURES].values.astype(np.float32)
    y_all = df["label"].values.astype(np.int32)
    X_scaled = X_all   # no further scaling needed

    # ── 2. Build windows per source file to avoid cross-file contamination ─
    sources = df["source"].values
    all_Xw, all_yw = [], []
    for src in df["source"].unique():
        mask = sources == src
        Xw, yw = make_windows(X_scaled[mask], y_all[mask], window, step)
        all_Xw.append(Xw)
        all_yw.append(yw)

    X = np.concatenate(all_Xw, axis=0)
    y = np.concatenate(all_yw, axis=0)

    print(f"\nWindowed dataset: {X.shape}, "f"ON={y.sum()}, OFF={(y == 0).sum()}, "
          f"imbalance ratio = {(y==0).sum()/max(y.sum(),1):.1f}:1")

    # ── 3. Train / val / test split ────────────────────────────────────────
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_size / (1 - test_size),
        stratify=y_tv, random_state=42)

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    build_dataset()
