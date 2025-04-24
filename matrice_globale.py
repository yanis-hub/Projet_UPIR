import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import cv2 as cv

# === Paramètres ===
chemin_dossier = "CSV"
video_path = "Videos_croped/MP4/u=0.169.MP4"
video = cv.VideoCapture(video_path)
fps = video.get(cv.CAP_PROP_FPS)
modes_max = 4
composante = "_u"  # ou "_v" pour travailler sur l'autre direction

# === Fichiers CSV ===
fichiers = sorted([f for f in os.listdir(chemin_dossier) if f.endswith(".csv") and f.startswith("u=")])
vitesses = [float(re.findall(r"u=([0-9.]+)", f)[0].rstrip(".")) for f in fichiers]
vitesses, fichiers = zip(*sorted(zip(vitesses, fichiers)))  # trié par vitesse croissante

# === Matrice globale ===
matrice_globale = []

for fichier in fichiers:
    df = pd.read_csv(os.path.join(chemin_dossier, fichier))
    
    # Colonnes contenant "_u" ou "_v"
    colonnes_cibles = [col for col in df.columns if composante in col]
    A = df[colonnes_cibles].T.values  # shape: (nb_points, nb_frames)
    
    # Nettoyage
    A = A[:, ~np.any(A == -1, axis=0)]
    A -= np.mean(A, axis=1, keepdims=True)
    
    matrice_globale.append(A)

# === Assemblage final ===
M = np.concatenate(matrice_globale, axis=1)  # shape: (nb_points, total_frames)

print(f"Matrice finale : {M.shape}")

# === SVD ===
U, S, Vt = np.linalg.svd(M, full_matrices=False)

# === Analyse FFT ===
def rms_amplitude(signal):
    return np.sqrt(np.mean(np.square(signal)))

time_total = M.shape[1] / fps
time = np.linspace(0, time_total, M.shape[1])

for i in range(modes_max):
    signal = Vt[i, :]
    amplitude = rms_amplitude(signal)

    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/fps)
    magnitude = (S[i] / np.sum(S)) * np.abs(fft_vals)

    mask = freqs >= 0.5
    freqs = freqs[mask]
    magnitude = magnitude[mask]

    dominant_freq = freqs[np.argmax(magnitude)]

    # === Affichage ===
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Mode {i+1} — Fréquence dominante : {dominant_freq:.2f} Hz", fontsize=14)

    plt.subplot(2, 1, 1)
    plt.plot(time, signal, color='black')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal temporel')

    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitude, color='black')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Transformée de Fourier')
    plt.xlim(0, 15)
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
