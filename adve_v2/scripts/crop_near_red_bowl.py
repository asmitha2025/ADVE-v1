import cv2
import glob
import torch
import clip
import numpy as np
from PIL import Image
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt")
clip_model, clip_prep = clip.load("ViT-B/32", device=device)

cap = cv2.VideoCapture("../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4")

frame_idx = 0
found_crops = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    res = yolo(frame, verbose=False)[0]
    for box in res.boxes:
        cls_name = yolo.names[int(box.cls[0])]
        # Look for bowl or items near bed
        if cls_name in ["bowl", "remote", "bed", "cell phone"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Expand bounding box to capture surrounding items (headphones next to bowl)
            h, w, _ = frame.shape
            ex_x1, ex_y1 = max(0, x1 - 100), max(0, y1 - 100)
            ex_x2, ex_y2 = min(w, x2 + 100), min(h, y2 + 100)
            crop = frame[ex_y1:ex_y2, ex_x1:ex_x2]
            if crop.size > 0:
                crop_path = f"scratch/crop_frame_{frame_idx:03d}_{cls_name}.jpg"
                cv2.imwrite(crop_path, crop)
                found_crops.append((frame_idx, cls_name, crop_path, crop))
    frame_idx += 1

cap.release()

print(f"Extracted {len(found_crops)} target crops around detected objects.")

colors = ["black", "white", "blue", "red", "pink", "silver or grey", "yellow", "green"]
text_tokens = clip.tokenize([f"a photo of a {c} object" for c in colors]).to(device)

for f_num, cname, cpath, crop_img in found_crops[:10]:
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_in = clip_prep(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = clip_model(img_in, text_tokens)
        probs = logits.softmax(dim=-1)[0]
    best_c = colors[torch.argmax(probs).item()]
    
    # Calculate dominant HSV color
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mean_h = np.mean(hsv[:, :, 0])
    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])
    print(f"Frame {f_num:3d} ({cname}): Main color near object = '{best_c}' | Mean HSV=({mean_h:.0f}, {mean_s:.0f}, {mean_v:.0f})")

