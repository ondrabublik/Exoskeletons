"""
data_loader.py
--------------
Loads all TXT files from the DATA folder, builds windowed sequences
and returns train/val/test splits ready for both Dense and CNN models.

Each TXT row: angle, x, y, z, omx, omy, omz, label
Sampling is done with a sliding window so the network sees a short
temporal context instead of a single snapshot.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "DATA")
COLUMNS    = ["angle", "x", "y", "z", "omx", "omy", "omz", "label"]
FEATURES   = ["angle", "x", "y", "z", "omx", "omy", "omz"]
WINDOW     = 20      # number of time-steps fed to the network at once
STEP       = 5       # sliding-window stride (smaller → more samples, more overlap)
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
# ────────────────────────────────────────────────────────────────────────────


def load_raw(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Read all TXT files and concatenate them into one DataFrame."""
    frames = []
    txt_files = [f for f in os.listdir(data_dir)
                 if f.upper().endswith(".TXT")]
    if not txt_files:
        raise FileNotFoundError(f"No .TXT files found in {data_dir}")

    for fname in sorted(txt_files):
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path, header=None, names=COLUMNS,
                         skipinitialspace=True)
        # Drop rows that couldn't be parsed (e.g. all-zero glitches)
        df = df.dropna()
        df["source"] = fname          # keep track of origin file
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
                  val_size: float = 0.15, test_size: float = 0.15,
                  fit_scaler: bool = True):
    """
    Full pipeline: load → scale → window → split.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    Each X has shape  (N, window, 7)
    Each y has shape  (N,)
    """
    df = load_raw()

    # ── 1. Scale features (fit on training data only, saved to disk) ──────
    scaler = StandardScaler()
    X_all = df[FEATURES].values.astype(np.float32)
    y_all = df["label"].values.astype(np.int32)

    # Fit scaler on everything for now (limited data).
    # With more data, fit only on train split.
    X_scaled = scaler.fit_transform(X_all)
    if fit_scaler:
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved → {SCALER_PATH}")

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

    print(f"\nWindowed dataset: {X.shape}, "
          f"ON={y.sum()}, OFF={(y == 0).sum()}, "
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

