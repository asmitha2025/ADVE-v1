# ADVE Real-World Enterprise Validation Report

**Status**: `NEEDS IMPROVEMENT`  
**Generated At**: 2026-07-26 19:18:29

---

## 1. Executive Summary Metrics

| Metric | Measured Value | Enterprise Target | Status |
| :--- | :--- | :--- | :--- |
| **Overall Mean Cosine Similarity** | `0.9545` | $\ge 0.9850$ (or $\ge 0.9400$) | ✅ PASS |
| **Overall Min Cosine Similarity** | `0.7250` | $\ge 0.9200$ (or $\ge 0.8500$) | ❌ NEEDS RETRAINING |
| **Encoder Budget Savings** | `97.0%` | $\ge 70.0\%$ | ✅ PASS |
| **Effective Throughput** | `7.5 FPS` | $\ge 30.0\text{ FPS}$ | ⚠️ CPU / LIMITED |

---

## 2. Per-Domain Performance Breakdown

| Target Domain | Mean Cosine Similarity | Domain Status |
| :--- | :--- | :--- |
| `urban_surveillance` | `0.9571` | ✅ PASS |
| `retail_indoor` | `0.9632` | ✅ PASS |
| `traffic_monitoring` | `0.9404` | ✅ PASS |

---

## 3. Video-by-Video Validation Table

| Video Name | Domain | Total Frames | Encoder Calls | Savings % | Mean CosSim | Min CosSim | Effective FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `MOT17-02-SDP-raw` | `urban_surveillance` | 150 | 4 | 97.3% | `0.9777` | `0.9332` | 7.3 |
| `benchmark_test` | `retail_indoor` | 300 | 8 | 97.3% | `0.9632` | `0.8707` | 8.5 |
| `demo_clip_1bqQHEbVlFc` | `urban_surveillance` | 450 | 15 | 96.7% | `0.9365` | `0.7250` | 7.1 |
| `demo_clip_5sLYAQS9sWQ` | `traffic_monitoring` | 600 | 19 | 96.8% | `0.9404` | `0.7250` | 7.3 |

---

## 4. Recommendations & Commercial Next Steps

1. **Retraining & Fine-Tuning**: If domain mean CosSim drops below `0.9400`, incorporate additional domain clips into `generate_training_data.py` and run 15 epochs of `train_reconstructor.py`.
2. **Anchor Refresh Tuning**: For low-light or fast camera motion scenes, trigger local cluster refreshes or lower spatial thresholds to guarantee worst-case frame accuracy stay above `0.8500`.
