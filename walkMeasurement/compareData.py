import re
import matplotlib.pyplot as plt
import numpy as np

# --- Seznam souborů ---
# filenames = [
#     "data_20251127_102309.txt",
#     "data_20251127_104156.txt",
#     "data_20251127_105241.txt"
# ]
filenames = [
    "data_20251127_105241.txt",
    "data_20251127_105327.txt",
    "data_20251127_105413.txt"
]

zeroValue = 530


# Low-pass parametr (0 < alpha ≤ 1)
alpha = 0.1

# Low-pass filtr (EMA)
def lowpass_filter(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


plt.figure()

for filename in filenames:
    # --- Načtení dat ---
    data = {"x": [], "y": [], "z": [], "a": [], "b": [], "c": [], "p": []}

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            matches = re.findall(r"([xyzabcp])\s+(-?\d+(?:\.\d+)?)", line)
            for key, value in matches:
                data[key].append(float(value))

    # výpočet úhlu
    angle = np.atan(np.array(data["y"] / np.array(data["x"]))) * 180 / np.pi
    angle = np.nan_to_num(angle, nan=0.0)
    angle = lowpass_filter(angle, alpha)
    # angle = angle - angle[0]

    # --- Vykreslení dat ---
    plt.plot(angle, label=filename)

plt.xlabel("Index")
plt.ylabel("Úhel (p - zeroValue)")
plt.legend()
plt.grid(True)
plt.show()
