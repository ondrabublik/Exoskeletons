"""
sensitivity_analysis.py
-----------------------
Sensitivity analysis of a trained neural network on training data.

Method:
  1) Load X_train from the same pipeline as training (build_dataset).
  2) Pick one fixed epsilon and perturb one feature at a time by epsilon * noise.
  3) For each training sample compute sensitivity:
        s_i = |y_pert_i - y_base_i| / epsilon
  4) Aggregate over all training samples: mean and std for each feature.
  5) Plot dependence on input feature (mean with std error bars).

Usage example:
  python sensitivity_analysis.py --model cnn
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from data_loader import build_dataset, FEATURES


OUT_DIR = os.path.dirname(__file__)


def _load_model(model_name: str):
    model_dir = os.path.join(OUT_DIR, model_name)
    final_path = os.path.join(model_dir, "final.keras")
    if not os.path.exists(final_path):
        raise FileNotFoundError(
            f"Model not found. Expected {final_path}"
        )
    model_path = final_path
    model = keras.models.load_model(model_path)
    print(f"Loaded model: {model_path}")
    return model


def _predict_in_batches(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    y = model.predict(X, batch_size=batch_size, verbose=0)
    y = np.asarray(y).reshape(-1)
    return y


def sensitivity_analysis(
    model_name: str = "cnn",
    epsilon: float = 0.2,
    seed: int = 42,
    batch_size: int = 256,
):
    model = _load_model(model_name)

    # Use the same split logic as during training; analysis is done on X_train.
    (X_train, _), _, _ = build_dataset()
    X_train = X_train.astype(np.float32)
    print(f"X_train shape: {X_train.shape}")

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    rng = np.random.default_rng(seed)
    n_features = X_train.shape[-1]

    y_base = _predict_in_batches(model, X_train, batch_size=batch_size)

    sample_sens_by_feature = np.zeros((n_features, len(y_base)), dtype=np.float32)

    for f_idx in range(n_features):
        feature_name = FEATURES[f_idx] if f_idx < len(FEATURES) else f"feature_{f_idx}"
        # Independent noise for each sample and each timestep.
        noise = rng.standard_normal(size=X_train.shape[:2]).astype(np.float32)
        X_pert = X_train.copy()
        X_pert[:, :, f_idx] += np.float32(epsilon) * noise
        y_pert = _predict_in_batches(model, X_pert, batch_size=batch_size)
        # Per-sample sensitivity so we can compute mean and std across train samples.
        sample_sens = np.abs(y_pert - y_base) / np.float32(epsilon)
        sample_sens_by_feature[f_idx] = sample_sens.astype(np.float32)
        print(f"Done feature: {feature_name} | mean={sample_sens.mean():.6f}")

    _save_outputs(model_name, epsilon, sample_sens_by_feature)


def _save_outputs(model_name, epsilon, sample_sens_by_feature):
    model_dir = os.path.join(OUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(model_dir, "sensitivity_analysis.csv")
    means = sample_sens_by_feature.mean(axis=1)
    feature_idx = np.arange(len(FEATURES), dtype=np.int32)

    header = "feature_idx,feature_name,epsilon,sensitivity_mean"
    rows = []
    for i, feat in enumerate(FEATURES):
        rows.append((feature_idx[i], feat, epsilon, means[i]))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for idx, feat, eps, mean_val in rows:
            f.write(f"{idx},{feat},{eps:.8f},{mean_val:.8f}\n")
    print(f"Saved CSV: {csv_path}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(FEATURES))
    ax.bar(
        x, means, color="#64B5F6", edgecolor="#1565C0", linewidth=1.2
    )
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=20, ha="right")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("|dy| / epsilon  (mean)")
    ax.set_title(f"Sensitivity on train set (epsilon = {epsilon:.4f})")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    plot_path = os.path.join(model_dir, "sensitivity_analysis.png")
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense", "cnn"], default="cnn")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    sensitivity_analysis(
        model_name=args.model,
        epsilon=args.epsilon,
        seed=args.seed,
        batch_size=args.batch_size,
    )
