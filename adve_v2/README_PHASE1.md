# ADVE Phase 1 Learned Reconstructor & Optimization Suite

## Quick-Start Commands

### 1. Generate 100K Training Triples
```bash
python training/generate_training_data.py --target_samples 100000 --device cuda
```

### 2. Train DeltaReconstructor (MLP + GRU)
```bash
python training/train_reconstructor.py --data training/data/training_triples.pt --epochs 20 --device cuda
```

### 3. Export ONNX Model & Test Latency
```bash
python export_onnx.py --checkpoint training/checkpoints/best_model.pt --output_onnx models/reconstructor_v2.onnx --device cuda
```

### 4. Run End-to-End Benchmark
```bash
python benchmark.py --video test_video.mp4 --reconstructor training/checkpoints/best_model.pt --device cuda
```
