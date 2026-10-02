import cv2
import glob
import torch
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_prep = clip.load("ViT-B/32", device=device)

cap = cv2.VideoCapture("../Testing videos/WhatsApp Video 2026-07-28 at 7.53.33 PM.mp4")

frame_idx = 0
table_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Inspect desk area frames around t = 11.5s to 14.5s (frames 260 to 320)
    if 260 <= frame_idx <= 320 and frame_idx % 2 == 0:
        cpath = f"scratch/desk_frame_{frame_idx:03d}.jpg"
        cv2.imwrite(cpath, frame)
        table_frames.append((frame_idx, cpath))
    frame_idx += 1

cap.release()

counts_prompts = [
    "no pen stand or cup visible",
    "an empty pen stand cup with 0 pens",
    "a pen stand with exactly 1 pen sticking out",
    "a pen stand with exactly 2 pens",
    "a pen stand with 3 or more pens"
]

tokens = clip.tokenize(counts_prompts).to(device)

print("=========================================================")
print("        DESK / PEN STAND COUNT EVALUATION               ")
print("=========================================================")

for f_num, cpath in table_frames:
    img = Image.open(cpath)
    img_in = clip_prep(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = clip_model(img_in, tokens)
        probs = logits.softmax(dim=-1)[0]
    best_idx = torch.argmax(probs).item()
    print(f"Frame {f_num:3d} (t={f_num/22.4:.1f}s): {counts_prompts[best_idx]} ({probs[best_idx].item():.1%})")

