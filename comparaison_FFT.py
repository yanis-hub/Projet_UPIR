import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import cv2 as cv

# === Paramètres ===
csv_file = 'CSV/u=0.480.csv'
video_path = "Videos_croped/MP4/u=0.480.MP4"
u = 0.480

# === Calcul de la vitesse reduite ===
UR = u / (0.005 * 1.41)
print(f"UR : {UR:.4f}")

video = cv.VideoCapture(video_path)
fps = video.get(cv.CAP_PROP_FPS)
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

data = pd.read_csv(csv_file)
n, m = data.shape[1], data.shape[0]
A = np.zeros((n - 1, m))

for k in range(n - 1):
    A[k, :] = data.iloc[:, k + 1]

# on supprime les colonnes contenant -1 (erreur de captage)
valid_cols = ~np.any(A == -1, axis=0)
A = A[:, valid_cols]

for k in range(1, len(A[:, 0]), 2):
    A[k, :] = height - A[k, :]

for i in range(A.shape[0]):
    A[i, :] -= np.mean(A[i, :])

# === SVD ===
U, S, Vt = np.linalg.svd(A, full_matrices=False)
ns = A.shape[1]
time = np.linspace(0, ns / fps, ns)

# === Fonction amplitude RMS ===
def rms_amplitude(data):
    return np.sqrt(np.mean(np.square(data)))

# === Comparaison FFT normalisée / non normalisée  ===
mod = 4  # nombre de modes à comparer
fig, axs = plt.subplots(mod, 1, figsize=(12, 2.5 * mod), sharex=True)
fig.suptitle(f'Comparaison FFT normalisée vs non normalisée — u = {u:.3f} m/s', fontsize=18, weight='bold')

colors = ['#1f77b4', '#ff7f0e']  # bleu, orange

for i in range(mod):
    signal = Vt[i, :]
    fft_signal = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1 / fps)

    pos_mask = freq >= 0
    freq = freq[pos_mask]
    fft_magnitude = np.abs(fft_signal[pos_mask])

    mag_non_norm = fft_magnitude
    mag_norm = (S[i] / np.sum(S)) * fft_magnitude

    ax = axs[i]
    ax.plot(freq, mag_non_norm, linestyle='--', color=colors[0], label='Non normalisée')
    ax.plot(freq, mag_norm, linestyle='-', linewidth=2, color=colors[1], label='Normalisée')

    ax.set_xlim(0, 15)
    ax.set_title(f'Mode {i + 1}', fontsize=14, weight='semibold')
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

axs[-1].set_xlabel('Fréquence (Hz)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

