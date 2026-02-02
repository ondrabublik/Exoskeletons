import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# ===============================
# 1) Načtení dat
# ===============================
data = np.loadtxt("data/KSTHKRWQ.TXT", delimiter=",")

X = data[:, :7]    # 7 vstupních signálů
y = data[:, 7]     # výstup 0/1

print("=== Trénovací data ===")
print(data)

# ===============================
# 2) Normalizace
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 3) Train/test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# 4) MLP model
# ===============================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=150, verbose=0)

# ===============================
# 5) Predikce na trénovacích datech
# ===============================
y_pred_prob = model.predict(X_scaled).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== Predikce modelu ===")
print(y_pred)

# ===============================
# 6) Vykreslení subplotů
# ===============================
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
