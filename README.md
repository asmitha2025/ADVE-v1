# frameroute / ADVE

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/frameroute?color=blue&style=flat-square)](https://pypi.org/project/frameroute/)
[![Python Version](https://img.shields.io/pypi/pyversions/frameroute?style=flat-square)](https://pypi.org/project/frameroute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Tests](https://github.com/asmitha2025/ADVE-v1/actions/workflows/tests.yml/badge.svg)](https://github.com/asmitha2025/ADVE-v1/actions/workflows/tests.yml)
[![Website](https://img.shields.io/badge/Live_Site-GitHub_Pages-purple?style=flat-square)](https://asmitha2025.github.io/ADVE-v1/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-asmitha2025%2FADVE--v1-181717?style=flat-square&logo=github)](https://github.com/asmitha2025/ADVE-v1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Asmitha_M-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/asmitha-m-ashh/)

**Budget-constrained frame selection and multimodal search for video AI pipelines.**  
*Stop paying your vision model to look at the same frame twice. Sample in change, not in time.*

[**🌐 Live Demo & Interactive Site**](https://asmitha2025.github.io/ADVE-v1/) • [**📊 Full Evidence & Benchmarks**](https://asmitha2025.github.io/ADVE-v1/evidence.html) • [**📦 PyPI Package**](https://pypi.org/project/frameroute/)

</div>

---

## The Problem: Blind Uniform Sampling

Sending video to Vision-Language Models (GPT-4o, Gemini, Claude, Qwen-VL) or dense CLIP encoders is the single most expensive bottleneck in video AI pipelines:

* **1 hour of video at 1 fps** = **3,600 vision model calls**.
* At \$0.005 per model call, processing 1 hour of video costs **\$18.00**.
* Across 1,000 hours/month, that is **\$18,000/month** in inference bills. *(Illustrative: list price x volume, not a customer invoice.)*

Yet in lectures, surveillance, meeting recordings, and dashcam footage, **most consecutive frames are nearly identical**. Blindly sampling every 1 second wastes up to 90% of your budget re-encoding static backgrounds and unchanged slides.

---

## The Solution: `frameroute`

`frameroute` places an ultra-lightweight change-detection layer (1–2 ms/frame on CPU) upstream of your expensive models. It cuts the **cumulative-novelty curve** into equal-area segments and allocates calls where real visual changes occur:

```
┌─────────────────┐       120+ FPS CPU       ┌──────────────────────┐   Only Changed Frames    ┌───────────────────────┐
│   Raw Video /   │ ───────────────────────> │  frameroute Router   │ ───────────────────────> │ Expensive VLM / CLIP  │
│  RTSP Stream    │   (160x90 Change Scan)   │  (Budget Allocation) │   (67-74% measured)      │ (GPT-4o, Claude, etc) │
└─────────────────┘                          └──────────────────────┘                          └───────────────────────┘
```

* **Zero GPU in the hot path:** Change detection runs on a 160×90 downscale using structural delta, ego-motion optical flow compensation, and edge metrics.
* **Hard Triggers:** Any frame whose novelty clears a sharp threshold is automatically preserved (never averaging away a 0.4s event).
* **Max-Gap Guarantees:** Ensures no extended timeframe is left un-sampled, even in static videos.
* **Compression example:** On a 23-minute lecture, **40,984 frames reduced to 46 keyframes**, retaining every slide transition. This is a compression figure; retrieval quality is measured separately in Benchmarks below.

---

## Installation

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

## Quickstart

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
    answer = your_expensive_vlm(frame)  # your model, untouched
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

## CLI Reference

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

## AI Agent Integration (MCP Server)

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

## Measured Benchmarks

### Ground-truth OCR evaluation

Scored against slide text that **neither index produced**, 20 grounded queries per lecture.
`uniform` and `frameroute` are compared at **matched call budgets** - the same number of
calls each, which is the only comparison a buyer should care about.

| Lecture | Calls | Every frame | Uniform (same calls) | frameroute |
| :--- | :---: | :---: | :---: | :---: |
| Cities & decarbonization | 104 | 0.450 | 0.600 | **0.750** |
| Deep learning | 131 | 0.450 | **0.750** | 0.650 |

*(hit@5. Reproduce: `python -m bench.grounded_eval --video <file> --budget 80 --n-queries 20`)*

**What this shows.** Against sending *every frame*, frameroute held or improved retrieval
while making **67-74% fewer calls**. The cost saving is real and reproducible.

**What it does not show.** Against *uniform sampling at the same budget*, frameroute won one
lecture (+0.15) and lost the other (-0.10). It did not replicate. The saving comes from
sending fewer frames - not from this picker being cleverer than uniform sampling, which is
about ten lines of code. Any claim to the contrary has been withdrawn.

### Limitations

- **Sample size:** two lectures, 20 queries each. This is not a published benchmark.
- **Domain:** slide/lecture footage. Continuous-motion video (traffic, crowds, fast sport)
  is where change-aware routing helps least, and no win has been measured there.
- **Hardware:** CPU timings come from a single Windows workstation, not a controlled rig.

---

## The Multimodal Search Stack (`adve`)

In addition to routing, `frameroute` includes the `adve` multimodal search engine:

| Modality | Module | Processing Engine |
| :--- | :--- | :--- |
| **Visual Search** | `adve.search` | CLIP (ViT-B/32) + FAISS Vector Index |
| **On-Screen Text** | `adve.vision` | EasyOCR + Full-Text Search Index |
| **Audio & Speech** | `adve.audio` | OpenAI Whisper Speech-to-Text |
| **Cross-Modal Fusion** | `adve.vision.unified_search` | Time-synchronized cross-modal retrieval |

---

## Repository Structure

```
.
├── frameroute/              # Core router - numpy + opencv only, no GPU
│   ├── signals.py           # CPU change detection + ego-motion compensation
│   ├── router.py            # Equal-area novelty budget router + StreamingRouter
│   ├── cascade.py           # Two-tier multi-model cascade router
│   ├── policies.py          # Baseline samplers (Uniform, SceneCut)
│   ├── semantics.py         # Query-aware scoring
│   ├── adapters.py          # Pipeline / framework adapters
│   ├── mcp_server.py        # Model Context Protocol (stdio) server
│   └── cli.py               # Command-line interface
├── adve/                    # Multimodal search stack (optional [search] extra)
│   ├── core/  search/  vision/  audio/  api/
├── bench/                   # Evaluation harness
│   ├── grounded_eval.py     # OCR ground-truth retrieval scoring
│   ├── cost_parity.py       # Wall-clock and dollar savings
│   ├── audit.py             # Customer-facing one-page audit
│   └── selftest.py          # Zero-dependency verification
├── tests/                   # pytest suites
├── examples/                # demo.py, run_frameroute_on_videoframe.py
├── results/                 # Benchmark outputs cited above
├── website/                 # WebGL marketing and evidence site
├── archive/                 # v1 prototype and superseded experiments
├── pyproject.toml
└── LICENSE
```

---

## Honest Engineering Notes

1. **Guarantees are about signals, not semantics:** The router guarantees it never misses large measured visual changes. Whether a change is meaningful is an empirical question measured by `bench/`.
2. **Savings come from sending fewer frames:** Changing from uniform clock sampling to change-aware sampling avoids spending in idle periods while protecting high-velocity events.
3. **Downstream independence:** `frameroute` does not touch your prompts, model choices, or API keys. You stay with your existing model provider.

---

## Author & Contact

* **Author:** Asmitha M
* **Email:** [asmitha8825@gmail.com](mailto:asmitha8825@gmail.com)
* **LinkedIn:** [linkedin.com/in/asmitha-m-ashh](https://www.linkedin.com/in/asmitha-m-ashh/)
* **Live Website:** [https://asmitha2025.github.io/ADVE-v1/](https://asmitha2025.github.io/ADVE-v1/)
* **Repository:** [https://github.com/asmitha2025/ADVE-v1](https://github.com/asmitha2025/ADVE-v1)

---

## License

This project is licensed under the **MIT License** — free for personal, commercial, and research use.
