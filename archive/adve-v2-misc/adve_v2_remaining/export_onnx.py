import os
import time
import torch
import argparse
import numpy as np

from adve.core.reconstructor_v2 import DeltaReconstructor


def export_and_benchmark(checkpoint_path, output_onnx, output_tensorrt, device="cuda"):
    print(f"=== Exporting & Benchmarking DeltaReconstructor ({checkpoint_path}) ===")

    # Load model
    model = DeltaReconstructor(clip_dim=512, delta_dim=128, hidden_dim=512)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded weights from {checkpoint_path}")

    model.to(device).eval()

    dummy_anchor = torch.randn(1, 512, device=device)
    dummy_delta  = torch.randn(1, 128, device=device)
    dummy_hprev  = torch.zeros(1, 128, device=device)

    # 1. PyTorch CUDA Latency Benchmark
    print("\n--- 1. PyTorch Latency ---")
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_anchor, dummy_delta, dummy_hprev)

        t0 = time.perf_counter()
        iters = 1000
        for _ in range(iters):
            _ = model(dummy_anchor, dummy_delta, dummy_hprev)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pytorch_ms = ((t1 - t0) / iters) * 1000.0
        print(f"PyTorch ({device.upper()}): {pytorch_ms:.4f} ms per inference")

    # 2. Export to ONNX
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_anchor, dummy_delta, dummy_hprev),
        output_onnx,
        input_names=["e_anchor", "delta_features", "h_prev"],
        output_names=["e_reconstructed", "h_next"],
        dynamic_axes={
            "e_anchor": {0: "batch_size"},
            "delta_features": {0: "batch_size"},
            "h_prev": {0: "batch_size"},
            "e_reconstructed": {0: "batch_size"},
            "h_next": {0: "batch_size"},
        },
        opset_version=14
    )
    print(f"[EXPORT] Successfully saved ONNX model -> {output_onnx}")

    # 3. ONNX Runtime Benchmark
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(output_onnx, providers=providers)

        inputs = {
            "e_anchor": dummy_anchor.cpu().numpy(),
            "delta_features": dummy_delta.cpu().numpy(),
            "h_prev": dummy_hprev.cpu().numpy(),
        }

        # Warmup
        for _ in range(100):
            _ = session.run(None, inputs)

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = session.run(None, inputs)
        t1 = time.perf_counter()
        onnx_ms = ((t1 - t0) / iters) * 1000.0
        print(f"ONNX Runtime: {onnx_ms:.4f} ms per inference")

    except ImportError:
        print("[ONNX Runtime] Package not installed (pip install onnxruntime-gpu). Skipping ORT benchmark.")

    print("\n=== Export Summary ===")
    print(f"Model format: ONNX opset 14")
    print(f"Artifact path: {output_onnx}")
    print(f"PyTorch Eager Latency : {pytorch_ms:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/reconstructor_v2.pt")
    parser.add_argument("--output_onnx", default="models/reconstructor_v2.onnx")
    parser.add_argument("--output_tensorrt", default="models/reconstructor_v2.trt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    export_and_benchmark(args.checkpoint, args.output_onnx, args.output_tensorrt, device=args.device)
