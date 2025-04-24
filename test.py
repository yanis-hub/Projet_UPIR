import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.ticker as ticker

csv_file = 'CSV/u=0.169.csv'
video_path = "Videos_croped/MP4/u=0.178.MP4"
video = cv.VideoCapture(video_path)

fps = video.get(cv.CAP_PROP_FPS)
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

print(height)