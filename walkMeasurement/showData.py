import re
import matplotlib.pyplot as plt

# --- Název souboru ---
filename = "data_20251127_105241.txt"   # uprav podle skutečného názvu souboru
# filename = "data_20251127_105327.txt"   # uprav podle skutečného názvu souboru
#filename = "data_20251112_181606.txt"   # uprav podle skutečného názvu souboru

# --- Načtení dat ---
data = {"x": [], "y": [], "z": [], "a": [], "b": [], "c": [], "p": []}

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        # vyhledá klíč-hodnota dvojice (např. x -0.20)
        matches = re.findall(r"([xyzabcp])\s+(-?\d+(?:\.\d+)?)", line)
        for key, value in matches:
            data[key].append(float(value))

# --- Kontrola načtení ---
for key in data:
    print(f"{key}: {len(data[key])} hodnot")

# --- Vykreslení dat ---
plt.figure(figsize=(10, 10))
for i, key in enumerate(data.keys(), 1):
    plt.subplot(7, 1, i)
    plt.plot(data[key], label=key)
    plt.ylabel(key)
    plt.grid(True)
    if i == 1:
        plt.title("Naměřená data ze souboru")

plt.xlabel("Vzorek")
plt.tight_layout()
plt.show()
