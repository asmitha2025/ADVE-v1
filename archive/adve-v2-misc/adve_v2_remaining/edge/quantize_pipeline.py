import os
import torch
import argparse

from adve.core.reconstructor_v2 import DeltaReconstructor


def quantize_reconstructor(checkpoint_path: str, output_path: str):
    print(f"=== INT8 Dynamic Quantization for ADVE DeltaReconstructor ===")
    
    model = DeltaReconstructor(clip_dim=512, delta_dim=128, hidden_dim=512)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded weights from {checkpoint_path}")

    model.eval()

    # Apply INT8 dynamic quantization
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    
    orig_size = os.path.getsize(checkpoint_path) / 1e6 if os.path.exists(checkpoint_path) else 0.0
    quant_size = os.path.getsize(output_path) / 1e6

    print(f"[QUANTIZE] Successfully created INT8 model -> {output_path}")
    print(f"  Original FP32 Size : {orig_size:.2f} MB")
    print(f"  Quantized INT8 Size: {quant_size:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="training/checkpoints/reconstructor_v2.pt")
    parser.add_argument("--output", default="edge/reconstructor_int8.pt")
    args = parser.parse_args()

    quantize_reconstructor(args.checkpoint, args.output)
