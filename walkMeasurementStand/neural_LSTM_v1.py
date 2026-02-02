# train_lstm.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# ====== Parametry ======
fs = 10.0         # vzorkovací frekvence
T = 7             # délka sekvence = 7 vzorků
DATA_FILE = "data/KSTHKRWQ.TXT"

EPOCHS = 200
BATCH_SIZE = 8
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ====== Načtení dat ======
data = np.loadtxt(DATA_FILE, delimiter=",")

X = data[:, :7]          # 7 časových vzorků
y = data[:, 7].astype(int)

# ====== Normalizace ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Uložení scaleru
import joblib
joblib.dump(scaler, "scaler.save")

# ====== Převod na LSTM tvar ======
# LSTM očekává tvar: [samples, timesteps, features]
# Zde: 7 vzorků * 1 signál → features = 1
X_lstm = X_scaled.reshape(-1, 7, 1)

# ====== Train/test split ======
X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y, test_size=0.2, random_state=RANDOM_SEED
)

# ====== LSTM model ======
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7, 1)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ====== Trénování ======
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=2
)

# ====== Vyhodnocení ======
print("\n--- Vyhodnocení ---")
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")

# Predikce
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# ====== Uložení modelu ======
model.save("lstm_model.h5")
print("\nModel uložen jako lstm_model.h5")
