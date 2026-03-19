import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------
# 1. Generování dat
# ---------------------------
N = 5000

X = np.random.rand(N, 5)  # hodnoty v intervalu [0,1]

# podmínka
y = ((X[:, 0] >= 1) | (X[:, 4] > 25)).astype(int)
# ---------------------------
# 2. Definice modelu
# ---------------------------
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(5,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# ---------------------------
# 3. Kompilace
# ---------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# 4. Trénování
# ---------------------------
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# ---------------------------
# 5. Testování
# ---------------------------
test_samples = np.array([
    [0.2, 0, 0, 0.3, 0],   # součet 0.5 -> 0
])

pred = model.predict(test_samples)

model.save("test_model_v1.h5")

print("\nTestování:")
for i in range(len(test_samples)):
    print(f"Vstup: {test_samples[i]}, výstup (pravděpodobnost): {pred[i][0]:.3f}, třída: {int(pred[i][0] > 0.5)}")