# train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ====== Parametry ======
DATA_FILE = "data/KSTHKRWQ.TXT"
RANDOM_SEED = 42
TEST_SIZE = 0.2        # podíl dat pro test
VAL_SPLIT = 0.1        # podíl z trénovací množiny pro validaci při fit()
BATCH_SIZE = 8
EPOCHS = 200
MODEL_FILE = "model.h5"

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ====== Načtení dat ======
# očekáváme: každý řádek má 8 hodnot: 7 vstupních + 1 cílovou (0/1)
data = np.loadtxt(DATA_FILE, delimiter=",")
X = data[:, :7]
y = data[:, 7].astype(int)

# ====== Rozdělení na trén/test ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y if len(np.unique(y))>1 else None
)

# ====== Normalizace (StandardScaler) ======
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# uložit scaler (pokud budete dělat inference později)
import joblib
joblib.dump(scaler, "scaler.save")

# ====== Návrh modelu (MLP) ======
def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")   # binární výstup
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

model = build_model(X_train.shape[1])
model.summary()

# ====== Callbacky ======
es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
mc = callbacks.ModelCheckpoint(MODEL_FILE, monitor="val_loss", save_best_only=True)

# ====== Řešení nerovnováhy tříd (pokud existuje) ======
# pokud je dataset nevyvážený, lze použít class_weight
unique, counts = np.unique(y_train, return_counts=True)
print("Train class distribution:", dict(zip(unique, counts)))
class_weight = None
if len(unique) == 2:
    # jednoduchý návrh class_weight (inverzní k četnosti)
    total = counts.sum()
    class_weight = {
        int(unique[0]): total / (2 * counts[0]),
        int(unique[1]): total / (2 * counts[1])
    }
    print("Using class_weight =", class_weight)

# ====== Trénování ======
history = model.fit(
    X_train, y_train,
    validation_split=VAL_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc],
    class_weight=class_weight,
    verbose=2
)

# ====== Vyhodnocení na testovací množině ======
print("\n--- Vyhodnocení na testovací množině ---")
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  AUC: {test_auc:.4f}")

# Predikce a report
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# uložit model (pokud ModelCheckpoint už neuložil)
model.save(MODEL_FILE)
print(f"\nModel uložen jako {MODEL_FILE}")
