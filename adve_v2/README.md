# frameroute / ADVE

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/frameroute?color=blue&style=flat-square)](https://pypi.org/project/frameroute/)
[![Python Version](https://img.shields.io/pypi/pyversions/frameroute?style=flat-square)](https://pypi.org/project/frameroute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Website](https://img.shields.io/badge/Live_Site-frameroute.dev-purple?style=flat-square)](https://asmitha2025.github.io/ADVE-v1/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-asmitha2025%2FADVE--v1-181717?style=flat-square&logo=github)](https://github.com/asmitha2025/ADVE-v1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Asmitha_M-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/asmitha-m-ashh/)

**Budget-constrained frame selection and multimodal search for video AI pipelines.**  
*Stop paying your vision model to look at the same frame twice. Sample in change, not in time.*

[**🌐 Live Demo & Interactive Site**](https://asmitha2025.github.io/ADVE-v1/) • [**📊 Full Evidence & Benchmarks**](https://asmitha2025.github.io/ADVE-v1/evidence.html) • [**📦 PyPI Package**](https://pypi.org/project/frameroute/)

</div>

---

## ⚡ The Problem: Blind Uniform Sampling

Sending video to Vision-Language Models (GPT-4o, Gemini, Claude, Qwen-VL) or dense CLIP encoders is the single most expensive bottleneck in video AI pipelines:

* **1 hour of video at 1 fps** = **3,600 vision model calls**.
* At \$0.005 per model call, processing 1 hour of video costs **\$18.00**.
* Across 1,000 hours/month, that is **\$18,000/month** in inference bills.

Yet in lectures, surveillance, meeting recordings, and dashcam footage, **most consecutive frames are nearly identical**. Blindly sampling every 1 second wastes up to 90% of your budget re-encoding static backgrounds and unchanged slides.

---

## 💡 The Solution: `frameroute`

`frameroute` places an ultra-lightweight change-detection layer (1–2 ms/frame on CPU) upstream of your expensive models. It cuts the **cumulative-novelty curve** into equal-area segments and allocates calls where real visual changes occur:

```
┌─────────────────┐       120+ FPS CPU       ┌──────────────────────┐   Only 2-10% of Frames   ┌───────────────────────┐
│   Raw Video /   │ ───────────────────────> │  frameroute Router   │ ───────────────────────> │ Expensive VLM / CLIP  │
│  RTSP Stream    │   (160x90 Change Scan)   │  (Budget Allocation) │   (80–99% Calls Saved)   │ (GPT-4o, Claude, etc) │
└─────────────────┘                          └──────────────────────┘                          └───────────────────────┘
```

* **Zero GPU in the hot path:** Change detection runs on a 160×90 downscale using structural delta, ego-motion optical flow compensation, and edge metrics.
* **Hard Triggers:** Any frame whose novelty clears a sharp threshold is automatically preserved (never averaging away a 0.4s event).
* **Max-Gap Guarantees:** Ensures no extended timeframe is left un-sampled, even in static videos.
* **Grounded Savings:** Proven on real 23-minute lecture footage, reducing **40,984 frames to 46 keyframes (99.89% cost reduction)** while retaining all slide transitions.

---

## 📦 Installation

Install directly from PyPI:

```bash
# Core router (ultra-lightweight: numpy + opencv-python only)
pip install frameroute

# Optional: with full multimodal search stack (torch, CLIP, FAISS, EasyOCR, Whisper)
pip install "frameroute[search]"

# Optional: with benchmarking harness (Groq VLM, evaluators)
pip install "frameroute[bench]"

# Complete suite
pip install "frameroute[search,bench]"
```

---

## 🚀 Quickstart

### 1. Python SDK — Offline Batch Video Processing

```python
from frameroute import FrameRouter, analyze_video

# 1. Analyze video change signals (120+ FPS on CPU)
track = analyze_video("lecture.mp4", weights="static_cam")

# 2. Select keyframes under a target call budget
router = FrameRouter(min_gap_sec=2.0, max_gap_sec=60.0, hard_trigger=0.45)
selection = router.select(track, budget=50)

print(f"Selected {selection.n_calls} frames out of {track.n_frames_total}")
print(f"Cost reduction: {(1.0 - selection.n_calls / track.n_frames_total) * 100:.2f}%")

# 3. Call your expensive vision model ONLY on the selected frames
for pick in selection.picks:
    frame = get_frame(pick.idx)
    answer = your_expensive_vlm(frame)  # Save 80-99% on API bills!
```

### 2. Python SDK — Online Real-Time Streaming (CCTV / RTSP)

```python
import cv2
from frameroute import StreamingRouter

# Token-bucket adaptive router for continuous camera streams
sr = StreamingRouter(budget_per_hour=240, fps=30.0, max_gap_sec=30.0)

cap = cv2.VideoCapture("rtsp://camera_feed")
idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Check if this frame earns an expensive model call
    if sr.should_spend(frame, idx):
        print(f"Frame {idx} triggered model inference!")
        run_model_inference(frame)
        
    idx += 1
```

---

## 🛠️ CLI Reference

`frameroute` provides a unified command-line tool suite:

```bash
# 1. Run a one-command Frame Budget Audit (estimates savings & reports guarantees)
frameroute audit meeting.mp4 --price-per-call 0.005 --volume-hours 500

# 2. Select keyframes within a specific budget
frameroute route lecture.mp4 --budget-per-hour 240

# 3. Price different sampling policies without calling any model
frameroute cost lecture.mp4 --usd-per-call 0.005

# 4. View raw change signals
frameroute signals lecture.mp4

# 5. Two-tier cascade routing (cheap model for subtle motion + heavy VLM for large events)
frameroute cascade video.mp4

# 6. Run verification selftest (0 GPU required)
python -m bench.selftest
```

---

## 🤖 AI Agent Integration (MCP Server)

`frameroute` ships with a stdio Model Context Protocol (MCP) server for integration with **Claude Desktop**, **Cursor**, **ChatGPT**, and agentic coding workflows:

```bash
frameroute-mcp
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "frameroute": {
      "command": "frameroute-mcp"
    }
  }
}
```

**Exposed MCP Tools:**
* `cost_video`: Instant zero-model cost projection.
* `route_video`: Automated budget-constrained keyframe selector.
* `signals_video`: Frame-level change metrics extraction.
* `audit_video`: Complete one-page video efficiency audit.

---

## 📊 Measured Benchmarks & Empirical Proof

### Ground-Truth OCR Evaluation
Scored against slide text ground truth (*neither index generated*) across 20 grounded queries:

| Arm | Calls | hit@5 | precision@5 | Compute Savings |
| :--- | :---: | :---: | :---: | :---: |
| **Full Compute (Every Frame)** | 400 | 0.450 | 0.400 | Baseline (0%) |
| **frameroute (cities lecture)** | **104** | **0.750** | **0.320** | **74.0% fewer calls** |
| **frameroute (deep learning)** | **131** | **0.650** | **0.430** | **67.2% fewer calls** |

> **Key Finding:** At **67–74% fewer calls**, retrieval quality was *better* than sending every frame, because a sparse change-aware index avoids clustering all top results on redundant frames from a single static timestamp.

---

## 🔍 The Multimodal Search Stack (`adve`)

In addition to routing, `frameroute` includes the `adve` multimodal search engine:

| Modality | Module | Processing Engine |
| :--- | :--- | :--- |
| **Visual Search** | `adve.search` | CLIP (ViT-B/32) + FAISS Vector Index |
| **On-Screen Text** | `adve.vision` | EasyOCR + Full-Text Search Index |
| **Audio & Speech** | `adve.audio` | OpenAI Whisper Speech-to-Text |
| **Cross-Modal Fusion** | `adve.vision.unified_search` | Time-synchronized cross-modal retrieval |

---

## 📁 Repository Structure

```
.
├── website/                    # Live interactive 3D WebGL marketing & evidence site
├── run_frameroute_on_videoframe.py # Ready-to-run video extraction runner
├── adve_v2/
│   ├── frameroute/             # Core lightweight router & signals package
│   │   ├── signals.py          # 120 FPS CPU change detection & motion compensation
│   │   ├── router.py           # Equal-accumulated-novelty budget router & StreamingRouter
│   │   ├── cascade.py          # Two-tier multi-model cascade router
│   │   ├── mcp_server.py       # Stdio Model Context Protocol (MCP) server
│   │   ├── policies.py         # Baseline comparison samplers (Uniform, SceneCut)
│   │   └── cli.py              # Command-line interface
│   ├── bench/                  # Grounded evaluation & audit harness
│   │   ├── audit.py            # Customer-facing one-pager audit generator
│   │   ├── grounded_eval.py    # Ground-truth evaluation against OCR slide text
│   │   ├── cost_parity.py      # Wall-clock & dollar savings benchmark
│   │   └── selftest.py         # Self-contained zero-dependency verification suite
│   ├── adve/                   # Deep multimodal search stack (CLIP, OCR, Whisper)
│   └── tests/                  # Automated pytest test suites
└── setup.py                    # Root distribution packaging
```

---

## ⚖️ Honest Engineering Notes

1. **Guarantees are about signals, not semantics:** The router guarantees it never misses large measured visual changes. Whether a change is meaningful is an empirical question measured by `bench/`.
2. **Savings come from sending fewer frames:** Changing from uniform clock sampling to change-aware sampling avoids spending in idle periods while protecting high-velocity events.
3. **Downstream independence:** `frameroute` does not touch your prompts, model choices, or API keys. You stay with your existing model provider.

---

## 👤 Author & Contact

* **Author:** Asmitha M
* **Email:** [asmitha8825@gmail.com](mailto:asmitha8825@gmail.com)
* **LinkedIn:** [linkedin.com/in/asmitha-m-ashh](https://www.linkedin.com/in/asmitha-m-ashh/)
* **Live Website:** [https://asmitha2025.github.io/ADVE-v1/](https://asmitha2025.github.io/ADVE-v1/)
* **Repository:** [https://github.com/asmitha2025/ADVE-v1](https://github.com/asmitha2025/ADVE-v1)

---

## 📄 License

This project is licensed under the **MIT License** — free for personal, commercial, and research use.
