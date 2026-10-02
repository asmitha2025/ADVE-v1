import glob
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_prep = clip.load("ViT-B/32", device=device)

crops = sorted(glob.glob("scratch/bowl_area_*.jpg"))

colors = [
    "black headphones or black earphone wire",
    "white headphones or white earphone wire",
    "blue headphones",
    "red headphones",
    "pink headphones"
]

text_tokens = clip.tokenize(colors).to(device)

scores = {c: 0.0 for c in colors}
count = 0

for cpath in crops[:30]:
    pil_img = Image.open(cpath)
    img_in = clip_prep(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = clip_model(img_in, text_tokens)
        probs = logits.softmax(dim=-1)[0]
    for i, c in enumerate(colors):
        scores[c] += probs[i].item()
    count += 1

print("=========================================================")
print("  OVERALL AVERAGE CLIP CONFIDENCE AROUND RED BOWL")
print("=========================================================")
for c, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    avg_p = (score / count) * 100
    print(f"  - {c:40s}: {avg_p:.1f}% average probability")
print("=========================================================")
