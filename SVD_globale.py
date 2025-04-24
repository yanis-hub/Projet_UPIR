import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# === Répertoire qui contient les CSV ===
chemin_dossier = "CSV"  
fichiers = sorted([f for f in os.listdir(chemin_dossier) if f.endswith(".csv") and f.startswith("u=")])

# Extraction des vitesses
vitesses = [float(re.findall(r"u=([0-9.]+)", f)[0].rstrip(".")) for f in fichiers]
vitesses, fichiers = zip(*sorted(zip(vitesses, fichiers)))
UR = [v / (0.005 * 1.41) for v in vitesses]
modes_max = 4

contributions = np.zeros((len(vitesses), modes_max))

for idx, fichier in enumerate(fichiers):
    data = pd.read_csv(os.path.join(chemin_dossier, fichier))
    n, m = data.shape[1], data.shape[0]

    A = np.zeros((n - 1, m))
    for k in range(n - 1):
        A[k, :] = data.iloc[:, k + 1]

    valid_cols = ~np.any(A == -1, axis=0)
    A = A[:, valid_cols]

    for i in range(A.shape[0]):
        A[i, :] -= np.mean(A[i, :])

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    energie_totale = np.sum(S[:modes_max]**2)
    for m in range(modes_max):
        contributions[idx, m] = (S[m]**2) / energie_totale


plt.figure(figsize=(10, 6))
markers = ['o', 's', 'd', '^']
colors = ['tab:blue', 'tab:red', 'tab:purple', 'black']
for i in range(modes_max):
    plt.plot(vitesses, contributions[:, i] * 1e6, marker=markers[i], color=colors[i], label=f"Mode {i+1}")

plt.xlabel("Vitesse (m/s)")
plt.ylabel("Contribution modale (×1e6)")
plt.title("Contribution énergétique des modes en fonction de U")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

markers = ['o', 's', 'd', '^']
colors = ['tab:blue', 'tab:red', 'tab:purple', 'black']
for i in range(modes_max):
    plt.plot(UR, contributions[:, i] * 1e6, marker=markers[i], color=colors[i], label=f"Mode {i+1}")

plt.xlabel("Vitesse réduite")
plt.ylabel("Contribution modale (×1e6)")
plt.title("Contribution énergétique des modes en fonction de U")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()