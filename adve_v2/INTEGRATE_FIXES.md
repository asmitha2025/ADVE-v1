# ADVE Enterprise Fix Pack Integration Guide

This guide details the integration steps for applying the Enterprise Fix Pack to achieve **$\ge 0.9850$ Mean CosSim** and **$\ge 0.8800$ Worst-Case Min CosSim**.

---

## 📦 Fix Pack Module Map

| Module | Location | What It Fixes |
| :--- | :--- | :--- |
| **`EgoMotionEstimator`** | `adve/core/ego_motion.py` | Camera pan/tilt crashes — ORB homography warps object centroids back to anchor frame |
| **`DeltaReconstructorV3`** | `adve/core/reconstructor_v3.py` | GRU temporal drift — hidden state clamped `[-3.0, 3.0]`, gate dampened to `0.50`, residual skip connection |
| **`ADVEEnterprisePipeline`**| `adve/core/pipeline_enterprise.py` | 9-factor gating (confidence, blur, object count, color hist), EMA smoothing ($\alpha=0.65$), empty scene handling |
| **`train_enterprise.py`** | `training/train_enterprise.py` | Ego-motion compensated triple generation + training loop for `DeltaReconstructorV3` |

---

## 🚀 Step-by-Step Execution Sequence

### Step 1: Generate Ego-Motion Compensated Triples
```bash
python training/train_enterprise.py generate \
    --video_dir demo_videos/ \
    --data training/data/enterprise_triples.pt \
    --target_samples 50000 \
    --device cuda
```

### Step 2: Train DeltaReconstructorV3
```bash
python training/train_enterprise.py train \
    --data training/data/enterprise_triples.pt \
    --output training/checkpoints/reconstructor_v3.pt \
    --epochs 20 \
    --batch_size 512 \
    --lr 3e-4 \
    --device cuda
```

### Step 3: Run Enterprise Audit Engine
```bash
python adve_realworld_test/scripts/enterprise_audit.py \
    --manifest adve_realworld_test/datasets/sample_manifest.json \
    --reconstructor training/checkpoints/reconstructor_v3.pt \
    --output results/enterprise_audit_v3.json \
    --device cuda
```
