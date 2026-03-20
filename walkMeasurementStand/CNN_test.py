import numpy as np
import tensorflow as tf

# ===============================
# PARAMETRY
# ===============================
T = 30              # délka časového okna
CHANNELS = 5
SAMPLES = 2000

# ===============================
# GENEROVÁNÍ DAT
# ===============================
def generate_data(samples, T, channels):
    X = np.zeros((samples, T, channels), dtype=np.float32)
    y = np.zeros((samples, 1), dtype=np.float32)

    for i in range(samples):
        t = np.linspace(0, 1, T)

        signal = np.zeros((T, channels))

        for ch in range(channels):
            freq = np.random.uniform(1, 5)
            phase = np.random.uniform(0, np.pi)
            noise = np.random.normal(0, 0.05, T)

            signal[:, ch] = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t + phase) + noise

        signal = np.clip(signal, 0, 1)

        # CÍLOVÁ HODNOTA:
        # např. průměr posledních 5 vzorků všech kanálů
        target = np.mean(signal[-5:, :])

        X[i] = signal
        y[i] = target

    return X, y

X, y = generate_data(SAMPLES, T, CHANNELS)

# ===============================
# ROZDĚLENÍ DAT
# ===============================
split = int(0.8 * SAMPLES)

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ===============================
# MODEL (ESP32 friendly)
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(T, CHANNELS)),

    tf.keras.layers.Conv1D(8, 3, activation="relu"),
    tf.keras.layers.Conv1D(8, 3, activation="relu"),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ===============================
# TRÉNOVÁNÍ
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

model.save("CNN_model_v1.h5")