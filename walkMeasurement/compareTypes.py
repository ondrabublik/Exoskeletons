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
    "data_20251127_105327.txt"
]

zeroValue = 530

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
    p = np.array(data["p"])
    c = np.array(data["c"])
    var = p*c

    # --- Vykreslení dat ---
    plt.plot(var, label=filename)

plt.xlabel("Index")
plt.ylabel("Úhel (p - zeroValue)")
plt.legend()
plt.grid(True)
plt.show()
