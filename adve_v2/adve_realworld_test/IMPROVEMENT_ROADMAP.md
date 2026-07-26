# ADVE Technical Improvement Roadmap: Target 0.985+ CosSim

This document details the engineering steps and retraining strategy to elevate ADVE's enterprise audit score from **0.9545 $\rightarrow$ 0.985+** and eliminate all worst-case frame drops ($< 0.85$).

---

## 🎯 Target Commercial Benchmarks

| Metric | Measured Baseline | Phase 2 Target | Enterprise Target |
| :--- | :--- | :--- | :--- |
| **Mean CosSim** | `0.9545` | `0.9700` | $\ge 0.9850$ |
| **Min CosSim** | `0.7250` | `0.8500` | $\ge 0.9200$ |
| **5th Percentile CosSim** | `0.8912` | `0.9200` | $\ge 0.9500$ |
| **Encoder Savings** | `97.0%` | `85.0%` | $\ge 75.0\%$ |
| **Audit Score** | `68.4` | `78.0` | $\ge 85.0$ (`ENTERPRISE READY`) |

---

## 🔧 High-ROI Fixes

### Fix #1: Distribution-Matched Domain Fine-Tuning (+0.015 – 0.020 CosSim)
Extract 30,000 spatial delta training triples from each weak domain (`retail_indoor`, `traffic_monitoring`, `low_light`), then fine-tune `DeltaReconstructor`:

```bash
# 1. Extract domain triples
python training/generate_training_data.py \
    --video_dir /path/to/target_domain/ \
    --output training/data/domain_triples.pt \
    --target_samples 30000

# 2. Fine-tune DeltaReconstructor (15 epochs)
python training/train_reconstructor.py \
    --data training/data/domain_triples.pt \
    --output training/checkpoints/reconstructor_v2.1.pt \
    --epochs 15 \
    --lr 1e-4 \
    --device cuda
```

### Fix #2: Confidence-Gated Anchor Refresh (+0.005 – 0.010 Min CosSim)
Prevents worst-case frame drops when YOLO tracking confidence falls below `0.35`. Implemented in `pipeline.py`:

```python
if current_graph is not None and current_graph.objects:
    confs = [getattr(obj, "confidence", 0.5) for obj in current_graph.objects.values()]
    if float(np.mean(confs)) < 0.35:
        refresh = True
```

---

## 📋 7-Day Action Plan

1. **Day 1**: Run `single_video_accuracy.py` on weakest target video. Identify low-similarity frame indices from `cos_sim_timeline.png`.
2. **Day 2**: Extract 30K triples from low-confidence video segments and fine-tune `reconstructor_v2.1.pt`.
3. **Day 3**: Re-run `enterprise_audit.py` to confirm score $\ge 75$.
4. **Day 4–7**: Package ONNX & INT8 models for enterprise distribution.
