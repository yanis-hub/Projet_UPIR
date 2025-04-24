import cv2
import pandas as pd
import numpy as np

# --- Chemins des fichiers vidéo et CSV ---
video_path = "Videos_croped/MP4/u=0.076.MP4"
csv_path_1 = "CSV/u=0.076.csv"           
csv_path_2 = "CSV/u=0.076 (manuel)/data_u=0.076.csv"   

# Charger les deux jeux de données CSV
df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

# Charger la vidéo
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir la vidéo de sortie
output_path = "Superposition/comparaison-csv_u=0.076.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(df1) or frame_idx >= len(df2):
        break

    row1 = df1.iloc[frame_idx]
    row2 = df2.iloc[frame_idx]

    for i in range(5):  # Points 0 à 4
        # CSV 1 (rouge)
        x1, y1 = int(row1[f'{i}_u']), int(row1[f'{i}_v'])
        cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)  # Rouge

        # CSV 2 (bleu)
        x2, y2 = int(row2[f"point{i}_x"]), int(row2[f"point{i}_y"])
        cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)  # Bleu

    # --- Légende (coin en haut à gauche) ---
    overlay = frame.copy()
    legend_x, legend_y = 10, 10
    legend_width, legend_height = 220, 60

    # Rectangle semi-transparent pour la légende
    cv2.rectangle(overlay, (legend_x, legend_y), 
                         (legend_x + legend_width, legend_y + legend_height), 
                         (255, 255, 255), -1)
    alpha = 0.4  # Opacité
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Cercles de couleur + textes
    cv2.circle(frame, (legend_x + 15, legend_y + 15), 6, (0, 0, 255), -1)
    cv2.putText(frame, 'Tracking python', (legend_x + 30, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.circle(frame, (legend_x + 15, legend_y + 40), 6, (255, 0, 0), -1)
    cv2.putText(frame, 'Tracking manuel', (legend_x + 30, legend_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    out.write(frame)
    frame_idx += 1

# Libération des ressources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Vidéo enregistrée :", output_path)
