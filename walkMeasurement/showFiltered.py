import re
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import numpy as np

# --- Funkce pro Butterworth low-pass filtr ---
def lowpass_filter(data, cutoff=0.1, order=3):
    """
    cutoff – mezní frekvence (0 < cutoff < 0.5)
    order  – řád filtru
    """
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# --- Název souboru ---
#filename = "data_20251112_181227.txt"
#filename = "data_20251112_181514.txt"   # uprav podle skutečného názvu souboru
filename = "data_20251112_181606.txt"   # uprav podle skutečného názvu souboru

# --- Načtení dat ---
data = {"x": [], "y": [], "z": [], "a": [], "b": [], "c": [], "p": []}

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        matches = re.findall(r"([xyzabcp])\s+(-?\d+(?:\.\d+)?)", line)
        for key, value in matches:
            data[key].append(float(value))

alfa = []
for i in range(len(data['x'])):
    alfa.append(np.atan(data['x'][i] / data['y'][i]))
data['alfa'] = alfa

# --- Aplikace low-pass filtru ---
filtered_data = {}
for key, values in data.items():
    if len(values) > 10:  # filtr má smysl jen pokud je dost dat
        filtered_data[key] = lowpass_filter(values, cutoff=0.5, order=3)
    else:
        filtered_data[key] = values

# --- Vykreslení dat ---
plt.figure(figsize=(10, 10))
for i, key in enumerate(data.keys(), 1):
    plt.subplot(8, 1, i)
    plt.plot(data[key], label=f"{key} raw", alpha=0.5)
    plt.plot(filtered_data[key], label=f"{key} filtered", linewidth=2)
    plt.ylabel(key)
    plt.grid(True)
    if i == 1:
        plt.title("Naměřená data – Low-pass filtr")
    plt.legend(loc="upper right")

plt.xlabel("Vzorek")
plt.tight_layout()
plt.show()
