import re
import matplotlib.pyplot as plt
import numpy as np

# --- Seznam souborů ---
# Ondra
filenames = [
    "data_20251127_105241.txt",
    "data_20251127_105327.txt",
    "data_20251127_105413.txt"
]

# Martin
# filenames = [
#     "data_20251127_102309.txt",
#     "data_20251127_102502.txt",
#     "data_20251127_102737.txt"
# ]

# Honza
# filenames = [
#     "data_20251127_104156.txt",
#     "data_20251127_104330.txt",
#     "data_20251127_104511.txt"
# ]

zeroValue = 530

dt = 0.1

# Low-pass filtr (EMA)
def lowpass_filter(x, alpha):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def timeDerivative(x, dt):
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = (x[i] - x[i-1]) / dt
    return y

def kombinovana_variace_negativni(x, y, window):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        # součet růstů v okně
        for j in range(start, i):
            if y[j + 1] < y[j] and x[j + 1] < x[j]:
                out[i] += 1

    return out

def kombinovana_variace_pozitivni(x, y, window):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        # součet růstů v okně
        for j in range(start, i):
            if y[j + 1] > y[j] and x[j + 1] > x[j]:
                out[i] += 1

    return out

def kombinovana_variace_pozitivne_negativni(x, y, window):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        # součet růstů v okně
        for j in range(start, i):
            if y[j + 1] > y[j] and x[j + 1] < x[j]:
                out[i] += 1

    return out

def pozitivni_variace(x, window):
    # změny mezi vzorky
    dx = np.diff(x)
    dx = np.append(dx, 0)  # zarovnání délky

    # bereme jen kladné změny → růsty
    pos_dx = np.clip(dx, 0, None)  # nechá jen ≥ 0

    # výstupní pole
    out = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        start = max(0, i - window + 1)
        # součet růstů v okně
        out[i] = np.sum(pos_dx[start:i])

    return out


def negativni_variace(x, window):
    # změny mezi vzorky
    dx = np.diff(x)
    dx = np.append(dx, 0)  # zarovnání délky

    # bereme jen záporné změny → poklesy
    neg_dx = np.clip(dx, None, 0)  # nechá jen ≤ 0

    # výstupní pole
    out = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        start = max(0, i - window + 1)
        # součet absolutních hodnot poklesů
        out[i] = np.sum(-neg_dx[start:i])
    return out

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

def zero_negative(x):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)-1):
        if x[i + 1] < x[i] and x[i] >= 0 >= x[i + 1]:
            out[i] = 1

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

    window = 5
    cp = pozitivni_variace(c, window)
    cm = negativni_variace(c, window)
    pp = pozitivni_variace(p, window)
    pm = negativni_variace(p, window)

    ndif = drive(c, p, 70, 4)

    # cm[cm == 0] = 0.00001
    identn = kombinovana_variace_negativni(c,p,window)
    identp = kombinovana_variace_pozitivni(c, p, window)
    identpn = kombinovana_variace_pozitivne_negativni(c, p, window)
    identnp = kombinovana_variace_pozitivne_negativni(p, c, window)


    # --- Vykreslení ---
    ax.plot(p, color="blue", label="p", alpha=1)
    ax.plot(c, color="darkred", label="c (raw)", alpha=1)
    # ax.plot(c_f, color="darkred", label="c (low-pass)")

    # ax.plot(cp, color="black", label="c  + variace", alpha=0.5)
    # ax.plot(cm, color="green", label="c - variace", alpha=1)
    # ax.plot(pp, color="cyan", label="p + variace", alpha=0.5)
    # ax.plot(pm, color="magenta", label="p - variace", alpha=1)

    ax.plot(50 * ndif, color="black", label="kombinovana variace", alpha=0.5)
    # ax.plot(50 * identnp * identn, color="magenta", label="kombinovana variace", alpha=1)

    ax.set_title(filename)
    ax.set_xlabel("Index")
    ax.set_ylabel("Hodnota")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
