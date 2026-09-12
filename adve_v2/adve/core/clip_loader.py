import clip
import torch

_clip_cache = {}

def load_clip_cached(model_name: str, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    key = (model_name, device)
    if key not in _clip_cache:
        print(f"[CLIP Loader] Loading CLIP model {model_name} on {device}...")
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        _clip_cache[key] = (model, preprocess)
    else:
        print(f"[CLIP Loader] Using cached CLIP model {model_name} on {device}.")
    return _clip_cache[key]


def load_clip_model(model_name: str = "ViT-B/32", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_clip_cached(model_name, device)

