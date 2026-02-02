import numpy as np
import matplotlib.pyplot as plt

# ====== Parametry ======
fs = 10.0  # vzorkovací frekvence [Hz]
dt = 1.0 / fs  # vzorkovací perioda [s]

# ====== Načtení dat ======
# data = np.loadtxt("data/DJYZLQAB.TXT", delimiter=",")
# data = np.loadtxt("data/KQNVDOFY.TXT", delimiter=",")
data = np.loadtxt("data/KSTHKRWQ.TXT", delimiter=",")
# data = np.loadtxt("data/RPAPAMEO.TXT", delimiter=",")
# data = np.loadtxt("data/WVZRLCYU.TXT", delimiter=",")
# data = np.loadtxt("data/XXPVYLBK.TXT", delimiter=",")

# časová osa
N = data.shape[0]
t = np.arange(N) * dt

# počet signálů
num_signals = data.shape[1]

# ====== Subplots ======
plt.figure(figsize=(10, 2 * num_signals))

for i in range(num_signals):
    plt.subplot(num_signals, 1, i + 1)
    plt.plot(t, data[:, i])
    plt.grid(True)
    plt.ylabel(f"Sloupec {i + 1}")

    if i == 0:
        plt.title("Subploty jednotlivých signálů (10 Hz)")

# spodní subplot má osu X
plt.xlabel("Čas [s]")

plt.tight_layout()
plt.show()
