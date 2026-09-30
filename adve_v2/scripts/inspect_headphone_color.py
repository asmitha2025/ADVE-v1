import cv2
import numpy as np

video_path = "../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4"
cap = cv2.VideoCapture(video_path)

frame_idx = 0
saved = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Check frames around t = 1.5s to 4.5s (frames 35 to 100)
    if 35 <= frame_idx <= 100 and frame_idx % 5 == 0:
        cv2.imwrite(f"scratch/frame_{frame_idx:03d}.jpg", frame)
        saved += 1
    frame_idx += 1

cap.release()
print(f"Saved {saved} inspection frames to scratch/")
