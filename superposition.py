import cv2
import pandas as pd
import numpy as np

# Paths to the video and CSV file (update with your actual paths)
video_path = "Videos_croped/MP4/u=0.169.MP4"  # Update this path
csv_path = "CSV/u=0.169.csv"  # Update this path

# Load tracking data
df = pd.read_csv(csv_path)

# Open video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video
output_path = "Superposition/superposition_u=0.169.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure the frame index does not exceed the number of rows in CSV
    if frame_idx < len(df):
        row = df.iloc[frame_idx]

        # Iterate over points using the new header format
        for i in range(5):  # Points 0 to 4
            x, y = int(row[f'{i}_u']), int(row[f'{i}_v'])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw red circles on tracked points

    out.write(frame)
    frame_idx += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as:", output_path)
