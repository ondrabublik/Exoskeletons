import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# ===============================
# 1) Načtení scaleru a modelu
# ===============================
scaler = joblib.load("scaler.save")
model = tf.keras.models.load_model("lstm_model.h5")

# ===============================
# 2) Načtení dat
# ===============================
data = np.loadtxt("data/XXPVYLBK.TXT", delimiter=",")

X = data[:, :7]          # vstupy
y = data[:, 7].astype(int)

print("=== Původní trénovací data ===")
print(data)

# ===============================
# 3) Normalizace jako při trénování
# ===============================
X_scaled = scaler.transform(X)

# ===============================
# 4) Úprava tvaru pro LSTM
# LSTM očekává tvar: (samples, timesteps, features)
# timesteps = 7, features = 1
# ===============================
X_lstm = X_scaled.reshape(-1, 7, 1)

# ===============================
# 5) Predikce
# ===============================
y_pred_prob = model.predict(X_lstm).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== Predikované hodnoty (0/1) ===")
print(y_pred)

print("\n=== Pravděpodobnosti ===")
print(y_pred_prob)

print("\n=== Skutečné hodnoty ===")
print(y)

num_signals = 7

plt.figure(figsize=(10, 2.5 * (num_signals + 1)))

# --- 7 vstupních signálů ---
for i in range(num_signals):
    plt.subplot(num_signals + 1, 1, i + 1)
    plt.plot(X[:, i], label=f"Vstup {i+1}")
    plt.ylabel(f"X{i+1}")
    plt.grid(True)

# --- 8. subplot: skutečný vs. predikovaný výstup ---
plt.subplot(num_signals + 1, 1, num_signals + 1)
plt.plot(y, "k.-", label="Skutečný výstup (y)")     # černě
plt.plot(y_pred, "r.--", label="Predikce (červeně)")  # červeně
plt.ylabel("Výstup")
plt.xlabel("Index vzorku")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
