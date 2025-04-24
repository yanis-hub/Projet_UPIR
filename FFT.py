import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.ticker as ticker

csv_file = 'CSV/u=0.169.csv'
video_path = "Videos_croped/MP4/u=0.169.MP4"

u = 0.169 # récupération vitesse 
UR = u / (0.005 * 1.41)  # D = 0.005 m / f = 1.41 Hz (papier d'Alexandre) 
print(f"UR : {UR:.4f}")

video = cv.VideoCapture(video_path)
data = pd.read_csv(csv_file)

# Informations vidéo
fps = video.get(cv.CAP_PROP_FPS)
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

n, m = data.shape[1], data.shape[0]  # nombre de points et nombre de snapshots

A = np.zeros((n-1, m))

# Transfert des colonnes du fichier CSV dans une matrice
for k in range(n-1):
    A[k, :] = data.iloc[:, k+1]

# Suppression des colonnes contenant -1 (valeurs manquantes)
cols_with_neg_one = np.any(A == -1, axis=0)
A = A[:, ~cols_with_neg_one]

# Ajustement de l'axe Y pour certaines données
for k in range(1, len(A[:, 0]), 2):
    A[k, :] = height - A[k, :]

B = np.array(A)  # sauvegarde des données pour plot

# Centrage des données autour de 0
for i in range(len(A[:, 0])):
    A[i, :] = A[i, :] - np.mean(A[i, :])  

# Analyse SVD    
U, S, Vt = np.linalg.svd(A, full_matrices=True)  # SVD pour séparer les modes de vibrations

h = n-1
ns = len(A[0, :])  # snapshots valides

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(S) + 1, 1), np.cumsum(S) / np.sum(S), '-o', color='k')
plt.xlabel('Modes')
plt.ylabel(r'$\lambda$ (Valeurs Propres)')
plt.title(f'Singular Values vs Modes, u = {u}')
plt.grid()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()

# FFT 
mod = 4  # nombre de modes affichés
freq_mode = np.zeros(h)  # pour récupérer la fréquence de chaque mode
time = np.linspace(0, ns/fps, ns)  # vecteur temporel
Amplitude = np.zeros(mod)

def rms_amplitude(data):
    return np.sqrt(np.mean(np.square(data)))

for i in range(mod):
    signal = Vt[i, :]
    Amplitude[i] = rms_amplitude(signal)

    # Appliquer la FFT
    signal_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/fps)

    # Calcul de la magnitude pondérée par la valeur propre relative (énergie)
    magnitude = (S[i] / np.sum(S)) * np.abs(signal_fft)

    # Filtrer les fréquences supérieures à 1 Hz
    mask = freq >= 0.5
    freq = freq[mask]
    magnitude = magnitude[mask]

    # Trouver les indices des 5 plus grandes magnitudes
    top_indices = np.argsort(magnitude)[-5:]

    # Extraire les fréquences correspondantes
    top_freqs = freq[top_indices]
    top_magnitudes = magnitude[top_indices]

    freq_mode[i] = freq[np.argmax(magnitude)]

    # Affichage du graphe FFT
    plt.figure(figsize=(12, 6))  

    plt.subplot(2, 1, 1)
    plt.plot(time, signal, color='black')
    plt.title(f'Signal temporel du mode {i+1}, u={u} m/s')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(freq, magnitude, color='black')
    plt.title(f'Transformée de Fourier du mode {i+1}, u={u} m/s')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(1, 15)
    plt.grid()

    # Annoter les 5 plus grandes fréquences avec une meilleure visibilité
    # for j in range(len(top_freqs)):
    #     plt.annotate(f'{top_freqs[j]:.2f} Hz', 
    #                  xy=(top_freqs[j], top_magnitudes[j]), 
    #                  xytext=(top_freqs[j] + 0.2, top_magnitudes[j] + 2),
    #                  arrowprops=dict(facecolor='red', edgecolor='black', shrink=0.05, headwidth=8, width=2),
    #                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
    #                  horizontalalignment='center',
    #                  fontsize=14,  # Augmenter la taille de la police
    #                  color='blue')

    plt.tight_layout()
    plt.show()