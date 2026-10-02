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
pen_stand_crops = []

queries = [
    "a photo of a pen stand with pens",
    "a photo of a pencil holder container with pens",
    "a photo of a cup holding pens on a desk",
    "a photo of a bed with items",
    "a photo of a table with items"
]
text_tokens = clip.tokenize(queries).to(device)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Sample every 5th frame
    if frame_idx % 5 == 0:
        h, w, _ = frame.shape
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_in = clip_prep(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = clip_model(img_in, text_tokens)
            probs = logits.softmax(dim=-1)[0]
        
        # Check probability for pen stand / container queries
        pen_stand_score = probs[0].item() + probs[1].item() + probs[2].item()
        
        # Also run YOLO to look for cups, bottles, vases, or small containers
        res = yolo(frame, verbose=False)[0]
        for box in res.boxes:
            cname = yolo.names[int(box.cls[0])]
            if cname in ["cup", "bottle", "vase", "dining table", "laptop"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ex_x1, ex_y1 = max(0, x1 - 50), max(0, y1 - 50)
                ex_x2, ex_y2 = min(w, x2 + 50), min(h, y2 + 50)
                crop = frame[ex_y1:ex_y2, ex_x1:ex_x2]
                if crop.size > 0:
                    crop_path = f"scratch/pen_stand_crop_{frame_idx:03d}_{cname}.jpg"
                    cv2.imwrite(crop_path, crop)
                    pen_stand_crops.append((frame_idx, cname, crop_path, pen_stand_score))

    frame_idx += 1

cap.release()

print(f"Extracted {len(pen_stand_crops)} potential pen stand crops across {frame_idx} frames.")

# Query count of pens on the top candidate crops
count_queries = [
    "a photo of 1 pen in a pen stand",
    "a photo of 2 pens in a pen stand",
    "a photo of 3 pens in a pen stand",
    "a photo of 4 pens in a pen stand",
    "a photo of 5 pens in a pen stand",
    "a photo of 6 pens in a pen stand",
    "a photo of 7 pens in a pen stand",
    "a photo of many pens in a pen stand",
    "a photo of an empty cup without pens"
]
count_tokens = clip.tokenize(count_queries).to(device)

sorted_crops = sorted(pen_stand_crops, key=lambda x: x[3], reverse=True)
for f_num, cname, cpath, score in sorted_crops[:10]:
    c_img = Image.open(cpath)
    img_in = clip_prep(c_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = clip_model(img_in, count_tokens)
        probs = logits.softmax(dim=-1)[0]
    best_cnt_idx = torch.argmax(probs).item()
    print(f"Frame {f_num:3d} ({cname}): Top match = '{count_queries[best_cnt_idx]}' ({probs[best_cnt_idx].item():.1%})")

