import os
import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


DATA_FILE = os.path.join("data", "DJYZLQAB.TXT")
RANDOM_SEED = 42
TEST_SIZE = 0.2


def load_data(path):
    data = np.loadtxt(path, delimiter=",")
    X = data[:, :7]
    y = data[:, 7].astype(int)
    return X, y


def build_model(input_dim, trial):
    units_1 = trial.suggest_int("units_1", 16, 128, step=16)
    units_2 = trial.suggest_int("units_2", 8, 64, step=8)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.05)
    epochs = trial.suggest_int("epochs", 5, 30)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(units_1, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units_2, activation="relu"),
        tf.keras.layers.Dropout(dropout / 2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, {"epochs": epochs, "batch_size": batch_size}


def objective(trial):
    X, y = load_data(DATA_FILE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model, fit_params = build_model(X_train.shape[1], trial)

    model.fit(
        X_train,
        y_train,
        epochs=fit_params["epochs"],
        batch_size=fit_params["batch_size"],
        verbose=0,
    )

    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(
       direction="maximize", 
       storage="sqlite:///optuna.db",
       load_if_exists=True,
    )
    study.optimize(objective, n_trials=20)

    print("\nBest trial:")
    best = study.best_trial
    print("  Value:", best.value)
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")


# ===== VÝSLEDEK PRVNÍHO POKUSU ======
'''
Best trial:
    PRVNÍ
        Value: 0.9825
        units_1 = 96
        units_2 = 32
        learning_rate = 0.0027522588082931057
        dropout = 0.2
        epochs = 21
        batch_size = 32

        -> RPAPAMEO -> test accuracy = 0.9635, TL = 0.0990
        -> KSTHKRWQ -> test accuracy = 0.9475, TL = 0.1255
        -> DJYZLQAB (trénovací) -> TA = 0.9700, TL = 0.0588

    DRUHÝ
        Value: 0.99
        units_1 = 64
        units_2 = 48
        learning_rate = 0.006702888624658378
        dropout = 0.15
        epochs = 27
        batch_size = 32

        -> RPAPAMEO -> test accuracy = 0.9635, TL = 0.0867
        -> KSTHKRWQ -> test accuracy = 0.9349, TL = 0.1442
        -> DJYZLQAB (trénovací) -> TA = 0.9700, TL = 0.0581
'''