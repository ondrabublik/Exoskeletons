import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# --- Parametry ---
filename = "data_20251112_181227.txt"
SAMPLE_RATE = 10  # Hz – odhad vzorkovací frekvence, uprav podle svého měření
CUTOFF = 0.9      # normalizovaná mezní frekvence (0–0.5)

# --- Funkce pro Butterworth low-pass filtr ---
def lowpass_filter(data, cutoff=0.1, order=3):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# --- Numerický integrál (trapezoidální metoda) ---
def integrate(values, dt):
    return np.cumsum(np.array(values) * dt)

# --- Načtení dat ---
data = {"x": [], "y": [], "z": [], "a": [], "b": [], "c": [], "p": []}
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        matches = re.findall(r"([xyzabcp])\s+(-?\d+(?:\.\d+)?)", line)
        for key, value in matches:
            data[key].append(float(value))

# --- Převod na numpy pole ---
for key in data:
    data[key] = np.array(data[key])

# --- Aplikace filtru ---
filtered = {}
for key in data:
    if len(data[key]) > 10:
        filtered[key] = lowpass_filter(data[key], cutoff=CUTOFF, order=3)
    else:
        filtered[key] = data[key]

# --- Výpočet velikostí ---
acc_mag = np.sqrt(filtered["x"]**2 + filtered["y"]**2 + filtered["z"]**2)
gyro_mag = np.sqrt(filtered["a"]**2 + filtered["b"]**2 + filtered["c"]**2)

# --- Integrace ---
dt = 1.0 / SAMPLE_RATE
vel_x = integrate(filtered["x"], dt)
vel_y = integrate(filtered["y"], dt)
vel_z = integrate(filtered["z"], dt)
vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

ang_a = integrate(filtered["a"], dt)
ang_b = integrate(filtered["b"], dt)
ang_c = integrate(filtered["c"], dt)
ang_mag = np.sqrt(ang_a**2 + ang_b**2 + ang_c**2)

# --- Vykreslení ---
plt.figure(figsize=(12, 12))

# 1. Akcelerace
plt.subplot(4, 1, 1)
plt.plot(filtered["x"], label="x")
plt.plot(filtered["y"], label="y")
plt.plot(filtered["z"], label="z")
plt.plot(acc_mag, label="|a|", color="black", linewidth=2)
plt.title("Zrychlení (filtrované)")
plt.ylabel("m/s²")
plt.legend()
plt.grid(True)

# 2. Rychlosti (integrál akcelerace)
plt.subplot(4, 1, 2)
plt.plot(vel_x, label="vx")
plt.plot(vel_y, label="vy")
plt.plot(vel_z, label="vz")
plt.plot(vel_mag, label="|v|", color="black", linewidth=2)
plt.title("Rychlosti (integrál zrychlení)")
plt.ylabel("m/s")
plt.legend()
plt.grid(True)

# 3. Úhlové rychlosti
plt.subplot(4, 1, 3)
plt.plot(filtered["a"], label="a")
plt.plot(filtered["b"], label="b")
plt.plot(filtered["c"], label="c")
plt.plot(gyro_mag, label="|ω|", color="black", linewidth=2)
plt.title("Úhlové rychlosti (filtrované)")
plt.ylabel("°/s")
plt.legend()
plt.grid(True)

# 4. Úhly (integrál úhlových rychlostí)
plt.subplot(4, 1, 4)
#plt.plot(ang_a, label="angle a")
#plt.plot(ang_b, label="angle b")
plt.plot(ang_c, label="angle c")
#plt.plot(ang_mag, label="|angle|", color="black", linewidth=2)
plt.title("Úhly (integrál úhlových rychlostí)")
plt.ylabel("°")
plt.xlabel("Vzorek")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
