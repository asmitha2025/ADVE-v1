import cv2
import glob
import torch
import clip
from PIL import Image
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt")
clip_model, clip_prep = clip.load("ViT-B/32", device=device)

cap = cv2.VideoCapture("../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4")

frame_idx = 0
bowl_crops = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    res = yolo(frame, verbose=False)[0]
    for box in res.boxes:
        cls_name = yolo.names[int(box.cls[0])]
        if cls_name == "bowl":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = frame.shape
            # Crop 200px wider around the bowl to catch headphones lying nearby
            ex_x1, ex_y1 = max(0, x1 - 200), max(0, y1 - 200)
            ex_x2, ex_y2 = min(w, x2 + 200), min(h, y2 + 200)
            crop = frame[ex_y1:ex_y2, ex_x1:ex_x2]
            if crop.size > 0:
                crop_path = f"scratch/bowl_area_{frame_idx:03d}.jpg"
                cv2.imwrite(crop_path, crop)
                bowl_crops.append((frame_idx, crop_path, crop))
    frame_idx += 1

cap.release()

print(f"Extracted {len(bowl_crops)} bowl area crops.")

queries = [
    "black headphones near the red bowl",
    "white headphones near the red bowl",
    "blue headphones near the red bowl",
    "red headphones near the red bowl",
    "pink headphones near the red bowl",
    "yellow or orange headphones near the red bowl",
    "silver or grey headphones near the red bowl",
    "green headphones near the red bowl"
]

text_tokens = clip.tokenize(queries).to(device)

for f_num, cpath, crop_img in bowl_crops[:15]:
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_in = clip_prep(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = clip_model(img_in, text_tokens)
        probs = logits.softmax(dim=-1)[0]
    top_idx = torch.argmax(probs).item()
    print(f"Frame {f_num:3d} near red bowl: Top result = '{queries[top_idx]}' ({probs[top_idx].item():.1%})")

