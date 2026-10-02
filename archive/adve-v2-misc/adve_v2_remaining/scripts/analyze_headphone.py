import cv2
import glob
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

frame_paths = sorted(glob.glob("scratch/frame_*.jpg"))

colors = [
    "black headphones", 
    "white headphones", 
    "red headphones", 
    "blue headphones", 
    "silver headphones", 
    "pink headphones", 
    "green headphones", 
    "yellow headphones"
]

text_tokens = clip.tokenize(colors).to(device)

print("=========================================================")
print("          CLIP HEADPHONE COLOR CLASSIFICATION            ")
print("=========================================================")

for fpath in frame_paths:
    image = Image.open(fpath)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_per_image, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1)[0]
    
    top_idx = torch.argmax(probs).item()
    print(f"Frame {fpath}: Top candidate = '{colors[top_idx]}' ({probs[top_idx].item():.1%})")

