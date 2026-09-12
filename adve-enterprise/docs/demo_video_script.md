# ADVE 2-MINUTE LOOM DEMO VIDEO SCRIPT & STORYBOARD

**Format:** Silent Split-Screen Screen Recording (Text Overlays Only - Optimized for Enterprise Buyers)  
**Duration:** 2 Minutes (120 Seconds)  
**Target Audience:** CTOs, Heads of Video Analytics, VPs of AI Infrastructure (Tata Elxsi, Hikvision, L&T, Honeywell)

---

### VISUAL LAYOUT

```
+------------------------------------+------------------------------------+
|  LEFT SCREEN: BASELINE CLIP       |  RIGHT SCREEN: ADVE ENGINE         |
|  - Full Vision Transformer         |  - Anchor-Delta Video Embedding    |
|  - 100% Encoder Execution          |  - Keyframe Anchors + Delta Graph  |
|  - FPS: ~14 FPS                    |  - FPS: ~88 FPS                    |
|  - GPU VRAM Usage: 95%             |  - GPU VRAM Usage: 35%             |
+------------------------------------+------------------------------------+
|  BOTTOM LIVE TICKER:                                                    |
|  "Compute Saved Today: ₹47,000 | Cosine Similarity: 0.984 | 12 Streams"   |
+-------------------------------------------------------------------------+
```

---

### TIMELINE & OVERLAY SPECIFICATION

| Time | Visual Focus | On-Screen Text Overlay | Key Metric Highlighted |
|------|--------------|------------------------|------------------------|
| **0:00 - 0:15** | Both screens start processing identical 1080p CCTV feed | *"Standard Video Neural Encoding vs ADVE Engine"* | Left GPU @ 95%, Right GPU @ 35% |
| **0:15 - 0:40** | Left side struggles @ 14 FPS; Right side smoothly plays @ 88 FPS | *"Baseline runs heavy CLIP Vision Transformer on EVERY frame."* | FPS Counter: 14 vs 88 FPS |
| **0:40 - 1:05** | Right screen shows Green **[ANCHOR]** flash every ~25 frames and **[DELTA]** on intermediates | *"ADVE selectively calls CLIP only on Anchor Frames (68% savings)."* | CosSim Score updating: `0.984` |
| **1:05 - 1:30** | Vehicle enters CCTV scene; ADVE automatically triggers instant Anchor refresh | *"Sudden appearance / spatial shift triggers automatic Anchor Refresh."* | Tag switches to `[ANCHOR REFRESH]` |
| **1:30 - 1:50** | API Terminal window overlays showing Prometheus `/metrics` and OpenAPI docs `/docs` | *"Seamless REST API & Docker Integration (`docker-compose up`)."* | `/metrics` exposing `adve_encoder_savings_ratio: 0.68` |
| **1:50 - 2:00** | Final call to action slide | *"Reduce Video AI Compute Costs by 60%+ | Get Evaluation Report: sales@adve.ai"* | Ticker: ₹47,000 / day compute saved |

---

### PRODUCTION NOTES
1. Record using Loom or OBS Studio in 4K / 1080p 60 FPS resolution.
2. Use dark theme for API terminals and Grafana metrics dashboard.
3. No voiceover; ensure high-contrast bold white text overlays over translucent dark background banners.
