import re
import matplotlib.pyplot as plt
import numpy as np

# --- Seznam souborů ---
# Ondra
# filenames = [
#     "data_20251127_105241.txt",
#     "data_20251127_105327.txt",
#     "data_20251127_105413.txt"
# ]

# Martin
# filenames = [
#     "data_20251127_102309.txt",
#     "data_20251127_102502.txt",
#     "data_20251127_102737.txt"
# ]

# Honza
filenames = [
    "data_20251127_104156.txt",
    "data_20251127_104330.txt",
    "data_20251127_104511.txt"
]

# Low-pass filtr (EMA)
def lowpass_filter(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def drive(x, y, minAngleTreshold, window):
    out = np.zeros_like(x, dtype=float)
    activation = 0
    for i in range(len(x) - 1):
        if activation == 0:
            if x[i + 1] < x[i] and x[i] >= 0 >= x[i + 1] and y[i] > minAngleTreshold:
                activation = 1

        if activation == 1:
            start = max(0, i - window + 1)
            variace = 0
            for j in range(start, i + 1):
                if y[j + 1] <= y[j]:
                    variace += 1
            if variace >= window - 2:
                out[i] = 1
            else:
                activation = 0
    return out

# Počet subplotů = počet souborů
fig, axes = plt.subplots(len(filenames), 1, figsize=(10, 5*len(filenames)))
if len(filenames) == 1:
    axes = [axes]  # aby šlo indexovat stejným způsobem

for ax, filename in zip(axes, filenames):
    # --- Načtení dat ---
    data = {"x": [], "y": [], "z": [], "a": [], "b": [], "c": [], "p": []}

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            matches = re.findall(r"([xyzabcp])\s+(-?\d+(?:\.\d+)?)", line)
            for key, value in matches:
                data[key].append(float(value))

    # převedení na numpy
    p = np.array(data["p"])
    p -= p[0]

    c = np.array(data["c"])
    c = lowpass_filter(c, 0.4)

    ndif = drive(c, p, 70, 4)

    # --- Vykreslení ---
    ax.plot(p, color="blue", label="p", alpha=1)
    ax.plot(c, color="darkred", label="c (raw)", alpha=1)

    ax.plot(50 * ndif, color="black", label="kombinovana variace", alpha=0.5)

    ax.set_title(filename)
    ax.set_xlabel("Index")
    ax.set_ylabel("Hodnota")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
