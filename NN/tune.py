"""
tune.py
-------
Hyperparameter search for both Dense and CNN models using Optuna.

What gets tuned
───────────────
Dense
  • number of layers          (1 – 3)
  • units per layer           (16 – 256, powers of 2)
  • dropout rate              (0.1 – 0.6)
  • learning rate             (1e-4 – 1e-2, log-scale)
  • batch size                (16, 32, 64)

CNN
  • number of conv layers     (1 – 3)
  • filters per layer         (8 – 64, powers of 2)
  • kernel size               (3, 5, 7)
  • dense head units          (8 – 64, powers of 2)
  • dropout rate              (0.1 – 0.6)
  • learning rate             (1e-4 – 1e-2, log-scale)
  • batch size                (16, 32, 64)

Each trial trains for at most TRIAL_EPOCHS epochs with early stopping
(patience = EARLY_STOP_PATIENCE) and reports val_auc as the objective.

After the study the best hyper-parameters are:
  • Printed to the console
  • Saved to  <model>/best_params.json
  • The best model is re-trained with those params for FINAL_EPOCHS and saved
    (same artifacts as train.py: .keras, .h5, .tflite, model.h, plots)

Usage
─────
    python tune.py --model dense --trials 50
    python tune.py --model cnn   --trials 50
    python tune.py --model both  --trials 50    ← default
    python tune.py --model both  --trials 100 --final-epochs 2000
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from converter import convert_model
from data_loader import build_dataset, WINDOW, FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep console tidy

# ── CONFIG ──────────────────────────────────────────────────────────────────
TRIAL_EPOCHS         = 200    # max epochs per trial (early-stop kicks in sooner)
EARLY_STOP_PATIENCE  = 30     # patience inside each trial
FINAL_EPOCHS         = 1000   # epochs when re-training the winner
FINAL_ES_PATIENCE    = 100    # early-stop patience for the final training run
MAX_TRAINABLE_PARAMS = 1500  # hard cap – trials exceeding this are pruned
N_FEATURES           = len(FEATURES)
BASE_DIR             = os.path.dirname(__file__)

# ── Optuna persistent storage (enables Optuna Dashboard) ─────────────────────
# Studies are stored in a local SQLite file so you can inspect them with:
#   optuna-dashboard sqlite:///optuna_studies.db
OPTUNA_DB = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_studies.db')}"
# ────────────────────────────────────────────────────────────────────────────


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_class_weight(y: np.ndarray) -> dict:
    n_total = len(y)
    n_pos   = y.sum()
    n_neg   = n_total - n_pos
    w0 = n_total / (2 * n_neg) if n_neg > 0 else 1.0
    w1 = n_total / (2 * n_pos) if n_pos > 0 else 1.0
    return {0: w0, 1: w1}


def count_trainable_params(model: keras.Model) -> int:
    """Return the total number of trainable parameters in the model."""
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))


def _compile(model: keras.Model, lr: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ── model builders from trial params ─────────────────────────────────────────

def build_dense_from_trial(trial: optuna.Trial) -> tuple[keras.Model, int, float]:
    """Returns (model, batch_size, learning_rate)."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units    = [trial.suggest_categorical(f"units_l{i}",
                    [16, 32, 64, 128, 256]) for i in range(n_layers)]
    dropout  = trial.suggest_float("dropout", 0.1, 0.6)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch    = trial.suggest_categorical("batch_size", [16, 32, 64])

    inp = keras.Input(shape=(WINDOW, N_FEATURES), name="input")
    x   = layers.Flatten()(inp)
    for u in units:
        x = layers.Dense(u, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inp, out, name="dense_model")
    return model, batch, lr


def build_cnn_from_trial(trial: optuna.Trial) -> tuple[keras.Model, int, float]:
    """Returns (model, batch_size, learning_rate)."""
    n_conv      = trial.suggest_int("n_conv", 1, 3)
    filters     = [trial.suggest_categorical(f"filters_l{i}",
                       [8, 16, 32, 64]) for i in range(n_conv)]
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64])
    dropout     = trial.suggest_float("dropout", 0.1, 0.6)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch       = trial.suggest_categorical("batch_size", [16, 32, 64])

    inp = keras.Input(shape=(WINDOW, N_FEATURES), name="input")
    x   = inp
    for f in filters:
        x = layers.Conv1D(f, kernel_size=kernel_size,
                          padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inp, out, name="cnn_model")
    return model, batch, lr


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(arch: str, train_data, val_data, class_weight: dict):
    X_train, y_train = train_data
    X_val,   y_val   = val_data

    def objective(trial: optuna.Trial) -> float:
        # suppress TF rebuild warnings inside trials
        tf.keras.backend.clear_session()
        warnings.filterwarnings("ignore")

        if arch == "dense":
            model, batch, lr = build_dense_from_trial(trial)
        else:
            model, batch, lr = build_cnn_from_trial(trial)

        # ── parameter-count guard ────────────────────────────────────────
        n_params = count_trainable_params(model)
        trial.set_user_attr("trainable_params", n_params)
        if n_params > MAX_TRAINABLE_PARAMS:
            raise optuna.exceptions.TrialPruned(
                f"Model has {n_params:,} trainable params "
                f"(limit = {MAX_TRAINABLE_PARAMS:,})"
            )

        _compile(model, lr)

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max",
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRIAL_EPOCHS,
            batch_size=batch,
            class_weight=class_weight,
            callbacks=cb,
            verbose=0,
        )

        val_auc = max(history.history.get("val_auc", [0.0]))
        return val_auc

    return objective


# ── Optuna visualisation plots ────────────────────────────────────────────────

def save_optuna_plots(study: optuna.Study, out_dir: str, arch: str):
    """Save Optuna visualisation plots as PNG (static matplotlib versions)."""
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    completed = trials_df[trials_df["state"] == "COMPLETE"]

    if completed.empty:
        print("No completed trials – skipping Optuna plots.")
        return

    # ── 1. Objective value over trials ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(completed["number"], completed["value"],
               alpha=0.6, s=30, label="trial val_auc")
    # running best
    best_so_far = completed["value"].cummax()
    ax.plot(completed["number"], best_so_far,
            color="crimson", lw=2, label="best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("val_auc")
    ax.set_title(f"{arch.upper()} – Optuna optimisation history")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "optuna_history.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Plot → {path}")

    # ── 2. Hyperparameter importance (manual Spearman correlation) ────────
    param_cols = [c for c in completed.columns if c.startswith("params_")]
    if param_cols:
        correlations = {}
        for col in param_cols:
            try:
                from scipy.stats import spearmanr
                vals = pd.to_numeric(completed[col], errors="coerce")
                mask = vals.notna()
                if mask.sum() > 2:
                    rho, _ = spearmanr(vals[mask], completed["value"][mask])
                    correlations[col.replace("params_", "")] = abs(rho)
            except Exception:
                pass

        if correlations:
            imp = pd.Series(correlations).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(3, int(len(imp) * 0.4))))
            imp.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("|Spearman ρ| with val_auc")
            ax.set_title(f"{arch.upper()} – hyperparameter importance")
            ax.grid(True, alpha=0.3, axis="x")
            fig.tight_layout()
            path2 = os.path.join(out_dir, "optuna_param_importance.png")
            fig.savefig(path2, dpi=120)
            plt.close(fig)
            print(f"  Plot → {path2}")


# ── permutation feature importance ───────────────────────────────────────────

def permutation_feature_importance(model: keras.Model,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: list,
                                   n_repeats: int = 10,
                                   out_dir: str = ".",
                                   arch: str = "model") -> pd.DataFrame:
    """
    Measure how much each input signal matters to the trained model.

    For each feature:
      1. Shuffle that feature column across all samples (destroys its information).
      2. Evaluate the model → record the AUC drop vs. baseline.
      3. Repeat n_repeats times and average.

    A large drop  → the feature is *important* – removing it hurts the model.
    A small drop  → the feature has *little impact* – it might be safe to drop.

    Results are saved as:
      • <out_dir>/feature_importance.png  – horizontal bar chart (mean ± std)
      • <out_dir>/feature_importance.csv  – raw numbers
    """
    from sklearn.metrics import roc_auc_score

    # Baseline AUC with all features intact
    y_pred    = model.predict(X, verbose=0).ravel()
    base_auc  = roc_auc_score(y, y_pred)
    print(f"\n  Permutation feature importance (baseline AUC = {base_auc:.4f})")

    rng      = np.random.default_rng(seed=42)
    n_feat   = X.shape[2]          # X shape: (samples, window, features)
    results  = {}

    for fi in range(n_feat):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # shuffle the fi-th feature across the sample axis only
            idx = rng.permutation(len(X_perm))
            X_perm[:, :, fi] = X_perm[idx, :, fi]
            y_p   = model.predict(X_perm, verbose=0).ravel()
            try:
                auc_p = roc_auc_score(y, y_p)
            except ValueError:
                auc_p = 0.5
            drops.append(base_auc - auc_p)
        results[feature_names[fi]] = drops
        mean_drop = np.mean(drops)
        print(f"    {feature_names[fi]:>12s}:  mean AUC drop = {mean_drop:+.4f}")

    # ── build DataFrame & sort by importance ─────────────────────────────
    df_imp = pd.DataFrame(results)
    means  = df_imp.mean().sort_values(ascending=False)
    stds   = df_imp.std()[means.index]

    # ── bar chart ─────────────────────────────────────────────────────────
    colors = ["#d62728" if m > 0.01 else "#1f77b4" if m > 0 else "#aec7e8"
              for m in means]
    fig, ax = plt.subplots(figsize=(8, max(3, int(len(means) * 0.6) + 1)))
    y_pos   = np.arange(len(means))
    ax.barh(y_pos, means.values, xerr=stds.values,
            align="center", color=colors, alpha=0.85,
            error_kw=dict(ecolor="black", capsize=4, lw=1.5))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(means.index, fontsize=11)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean AUC drop when feature is shuffled\n"
                  "(larger = more important)", fontsize=10)
    ax.set_title(f"{arch.upper()} – Permutation Feature Importance\n"
                 f"(baseline AUC = {base_auc:.4f},  {n_repeats} repeats)",
                 fontsize=12)
    # annotate bars
    for i, (m, s) in enumerate(zip(means.values, stds.values)):
        label = f"{m:+.4f} ± {s:.4f}"
        ax.text(max(m, 0) + 0.001, i, label,
                va="center", ha="left", fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    img_path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(img_path, dpi=130)
    plt.close(fig)
    print(f"  Plot → {img_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "feature":   means.index,
        "mean_auc_drop": means.values,
        "std_auc_drop":  stds.values,
    })
    csv_path = os.path.join(out_dir, "feature_importance.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  CSV  → {csv_path}")

    return summary


# ── final training with best params ──────────────────────────────────────────

def plot_training(history: keras.callbacks.History, name: str, out_dir: str):
    """Same diagnostic plots as train.py."""
    h      = history.history
    epochs = range(1, len(h["loss"]) + 1)

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
        ax.plot(epochs, h[key],          lw=2, label="train")
        ax.plot(epochs, h[f"val_{key}"], lw=2, label="val", linestyle="--")
        ax.set_title(f"{title}\n({note})")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name.upper()} (tuned) – training curves", fontsize=13)
    fig.tight_layout()
    path1 = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"  Plot → {path1}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    loss_gap = np.array(h["val_loss"]) - np.array(h["loss"])
    axes[0].plot(epochs, loss_gap, color="crimson", lw=2)
    axes[0].axhline(0, color="gray", linestyle="--", lw=0.8)
    axes[0].fill_between(epochs, loss_gap, 0,
                         where=(loss_gap > 0), alpha=0.15, color="crimson",
                         label="val > train (overfit risk)")
    axes[0].fill_between(epochs, loss_gap, 0,
                         where=(loss_gap <= 0), alpha=0.15, color="steelblue",
                         label="train > val")
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
    fig.suptitle(f"{name.upper()} (tuned) – overfitting diagnostic", fontsize=13)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "overfit_gap.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"  Plot → {path2}")


def retrain_best(arch: str, best_params: dict,
                 train_data, val_data, test_data,
                 class_weight: dict, out_dir: str):
    """Re-build model with best params and train to convergence."""
    tf.keras.backend.clear_session()
    X_train, y_train = train_data
    X_val,   y_val   = val_data
    X_test,  y_test  = test_data

    # ── re-build model from saved params ────────────────────────────────
    lr    = best_params["lr"]
    batch = int(best_params["batch_size"])

    if arch == "dense":
        n_layers = int(best_params["n_layers"])
        units    = [int(best_params[f"units_l{i}"]) for i in range(n_layers)]
        dropout  = best_params["dropout"]

        inp = keras.Input(shape=(WINDOW, N_FEATURES), name="input")
        x   = layers.Flatten()(inp)
        for u in units:
            x = layers.Dense(u, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        model = keras.Model(inp, out, name="dense_model")

    else:  # cnn
        n_conv      = int(best_params["n_conv"])
        filters     = [int(best_params[f"filters_l{i}"]) for i in range(n_conv)]
        kernel_size = int(best_params["kernel_size"])
        dense_units = int(best_params["dense_units"])
        dropout     = best_params["dropout"]

        inp = keras.Input(shape=(WINDOW, N_FEATURES), name="input")
        x   = inp
        for f in filters:
            x = layers.Conv1D(f, kernel_size=kernel_size,
                              padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        model = keras.Model(inp, out, name="cnn_model")

    _compile(model, lr)
    n_params = count_trainable_params(model)
    print(f"\n  Trainable parameters: {n_params:,}  (limit = {MAX_TRAINABLE_PARAMS:,})")
    model.summary()

    os.makedirs(out_dir, exist_ok=True)

    ckpt_keras = os.path.join(out_dir, "best.keras")
    ckpt_h5    = os.path.join(out_dir, "best.h5")
    callbacks  = [
        keras.callbacks.ModelCheckpoint(
            ckpt_keras, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            ckpt_h5, monitor="val_auc", mode="max",
            save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=FINAL_ES_PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=30, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FINAL_EPOCHS,
        batch_size=batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Test evaluation ──────────────────────────────────────────────────
    print(f"\n── {arch.upper()} (tuned) TEST RESULTS ──")
    results = model.evaluate(X_test, y_test, verbose=0)
    for mname, val in zip(model.metrics_names, results):
        print(f"  {mname}: {val:.4f}")

    # ── Permutation feature importance ───────────────────────────────────
    permutation_feature_importance(
        model, X_test, y_test,
        feature_names=FEATURES,
        n_repeats=15,
        out_dir=out_dir,
        arch=arch,
    )

    # ── Save final artifacts ─────────────────────────────────────────────
    keras_path = os.path.join(out_dir, "final.keras")
    model.save(keras_path)
    print(f"  Saved → {keras_path}")

    h5_path = os.path.join(out_dir, "final.h5")
    model.save(h5_path)
    print(f"  Saved → {h5_path}")

    convert_model(h5_path, out_dir)

    best_h5 = os.path.join(out_dir, "best.h5")
    if os.path.exists(best_h5):
        best_sub = os.path.join(out_dir, "best")
        convert_model(best_h5, best_sub)

    saved_model_dir = os.path.join(out_dir, "saved_model")
    model.export(saved_model_dir)
    print(f"  SavedModel → {saved_model_dir}")

    plot_training(history, arch, out_dir)
    return history


# ── main study runner ─────────────────────────────────────────────────────────

def run_study(arch: str, n_trials: int,
              train_data, val_data, test_data,
              class_weight: dict):

    out_dir = os.path.join(BASE_DIR, arch)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Optuna study  –  {arch.upper()}  ({n_trials} trials)")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"{arch}_study",
        storage=OPTUNA_DB,
        load_if_exists=True,          # resume if the study already exists in the DB
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=10),
    )

    objective = make_objective(arch, train_data, val_data, class_weight)

    # progress bar via tqdm callback
    optuna.logging.disable_default_handler()
    from tqdm import tqdm
    with tqdm(total=n_trials, desc=f"{arch.upper()} trials",
              unit="trial") as pbar:
        def _cb(study, trial):
            pbar.update(1)
            if study.best_value is not None:
                pbar.set_postfix({"best_val_auc": f"{study.best_value:.4f}"})
        study.optimize(objective, n_trials=n_trials, callbacks=[_cb])

    best = study.best_trial
    print(f"\n  Best trial #{best.number}  val_auc = {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # ── persist best params ──────────────────────────────────────────────
    params_path = os.path.join(out_dir, "best_params.json")
    saved = dict(best.params)
    saved["trainable_params"] = best.user_attrs.get("trainable_params", "n/a")
    saved["max_trainable_params_limit"] = MAX_TRAINABLE_PARAMS
    with open(params_path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\n  Best params saved → {params_path}")

    # ── Optuna visualisation plots ───────────────────────────────────────
    save_optuna_plots(study, out_dir, arch)

    # ── Dashboard hint ───────────────────────────────────────────────────
    print(f"\n  >> To explore this study interactively, run:")
    print(f"       optuna-dashboard {OPTUNA_DB}")
    print( "     then open  http://127.0.0.1:8080  in your browser.")

    # ── Re-train with best params ────────────────────────────────────────
    print(f"\n  Re-training {arch.upper()} with best params ({FINAL_EPOCHS} epochs max) …")
    retrain_best(arch, best.params,
                 train_data, val_data, test_data,
                 class_weight, out_dir)

    return study


def main(which: str = "both", n_trials: int = 50, final_epochs: int = FINAL_EPOCHS):
    global FINAL_EPOCHS
    FINAL_EPOCHS = final_epochs

    print("=" * 60)
    print("Loading & windowing data …")
    print("=" * 60)
    train_data, val_data, test_data = build_dataset()
    X_train, y_train = train_data
    class_weight = compute_class_weight(y_train)

    if which in ("dense", "both"):
        run_study("dense", n_trials, train_data, val_data, test_data, class_weight)

    if which in ("cnn", "both"):
        run_study("cnn",   n_trials, train_data, val_data, test_data, class_weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for Dense / CNN models.")
    parser.add_argument("--model",   choices=["dense", "cnn", "both"],
                        default="cnn",
                        help="Which architecture to tune (default: both).")
    parser.add_argument("--trials",  type=int, default=300,
                        help="Number of Optuna trials per architecture (default: 20).")
    parser.add_argument("--final-epochs", type=int, default=FINAL_EPOCHS,
                        dest="final_epochs",
                        help=f"Max epochs for final re-training (default: {FINAL_EPOCHS}).")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch Optuna Dashboard (web UI) and exit. "
                             "Open http://127.0.0.1:8080 in your browser.")
    args = parser.parse_args()

    if args.dashboard:
        import subprocess, sys
        db_path = os.path.join(BASE_DIR, "optuna_studies.db")
        if not os.path.exists(db_path):
            print("No studies database found yet. Run tuning first to populate it.")
            sys.exit(1)
        # Resolve the optuna-dashboard executable that lives in the same venv
        # as this script's Python interpreter (avoids picking up a wrong venv).
        venv_scripts = os.path.dirname(sys.executable)
        dashboard_exe = os.path.join(venv_scripts, "optuna-dashboard.exe")
        if not os.path.exists(dashboard_exe):
            dashboard_exe = os.path.join(venv_scripts, "optuna-dashboard")
        if not os.path.exists(dashboard_exe):
            print("ERROR: optuna-dashboard executable not found in the current venv.")
            print("       Run:  pip install optuna-dashboard")
            sys.exit(1)
        print("Starting Optuna Dashboard ...  open http://127.0.0.1:8080")
        subprocess.run([dashboard_exe, OPTUNA_DB], check=True)
    else:
        main(args.model, args.trials, args.final_epochs)

