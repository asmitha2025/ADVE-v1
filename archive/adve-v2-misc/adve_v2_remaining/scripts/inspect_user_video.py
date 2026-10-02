import cv2
from ultralytics import YOLO

video_path = "../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4"
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps if fps > 0 else 0

class_counts = {}
sample_frames = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 15 == 0:
        res = model(frame, verbose=False)[0]
        detected = []
        for box in res.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > 0.35:
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                detected.append(cls_name)
        if detected:
            sample_frames.append((frame_idx, detected))
    frame_idx += 1

cap.release()

print("=========================================================")
print("           REAL-WORLD VIDEO ANALYSIS REPORT              ")
print("=========================================================")
print(f"Video File  : {video_path}")
print(f"Resolution  : {width} x {height}")
print(f"Total Frames: {total_frames}")
print(f"FPS         : {fps:.2f}")
print(f"Duration    : {duration_sec:.2f} seconds")
print("---------------------------------------------------------")
print("Detected Object Distribution:")
for obj, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {obj}: {count} detections across sampled frames")
print("---------------------------------------------------------")
print("Frame Timeline Sample (Every ~0.5s):")
for f_num, objs in sample_frames[:15]:
    sec = f_num / fps if fps > 0 else 0
    print(f"  t={sec:4.1f}s (Frame {f_num:3d}): {', '.join(objs)}")
print("=========================================================")
