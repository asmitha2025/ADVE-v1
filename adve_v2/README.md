# ADVE / frameroute

**Budget-constrained frame selection and multimodal search for video AI pipelines.**

> This README was rewritten in September 2026 after an audit found that the
> project's headline claim did not survive its own benchmark. The previous
> version is preserved verbatim at [`README_v1_ARCHIVE.md`](README_v1_ARCHIVE.md).
> Nothing below is asserted without a file you can open or a command you can run.

---

## What happened

The original claim was *"96.67% fewer CLIP encoder calls"*, achieved by
embedding only anchor frames and reconstructing the rest from spatial-graph
deltas of YOLO object tracks.

The call reduction was real. The **cost** reduction was not.

From [`adve_v2/results/traffic_benchmark_report.json`](adve_v2/results/traffic_benchmark_report.json),
same video, same machine, same 300 frames:

| | full CLIP | ADVE | delta |
|---|---:|---:|---:|
| time | 100.77 s | 95.47 s | **−5.3%** |
| throughput | 3.0 fps | 3.1 fps | +0.1 fps |
| memory | 1416 MB | 1668 MB | **+17.8%** |
| *calculated* GFLOPs | 1320 | 529 | −59.9% |

The 59.9% is arithmetic (frames × 4.4 GFLOPs). The 5.3% is a stopwatch.

The reason is in [`adve_v2/docs/GPU_PERFORMANCE_CERTIFICATE.md`](adve_v2/docs/GPU_PERFORMANCE_CERTIFICATE.md):

```
YOLOv8 detection & tracking      120.330 ms
DeltaReconstructorV3               1.431 ms
Spatial motion graph               0.000 ms
─────────────────────────────────────────
Total per-frame latency          121.760 ms   →  8.2 fps
```

To skip one CLIP pass, the pipeline ran a detector on the same frame. A
batched ViT-B/32 pass amortises to single-digit milliseconds. **The shortcut
cost roughly twenty times the road.** That is a design property, not a
tuning problem, and no amount of quantisation fixes it.

Three further findings from the same audit:

- **The 0.88 "hard floor" is circular.** In production `SafetyGate` has no
  ground truth, so it compares the prediction to the *anchor* rather than to
  the truth. A reconstructor that echoes the anchor scores 1.0 forever.
- **Retrieval was never measured.** Every accuracy figure in `docs/` is
  cosine similarity to a true CLIP vector — a metric that cannot fail,
  because unrelated frames from one fixed camera sit at 0.85–0.95 in CLIP
  space. Whether *search still works* was never tested.
- **The patent claims have prior art.** Keyframe feature propagation is
  Deep Feature Flow (CVPR 2017); temporal-redundancy gating in transformers
  is Eventful Transformers (ICCV 2023); anchor/delta structure is video
  codecs. Claims 1–6 are not novel.

None of the "certified", "verified" or "independent" documents in `docs/`
were produced by a third party. Treat them as drafts, not evidence.

---

## What the project is now

Two things, cleanly separated.

### 1. `frameroute` — sample in change, not in time

Every video pipeline has to answer *which frames do I send to the expensive
model?* Today the answer is 1 fps or scene-cut detection. Both are bad:
uniform sampling spends the same on an empty corridor as on a collision, and
scene-cut is blind to everything happening inside a shot.

`frameroute` cuts the **cumulative-novelty curve** into equal-area segments
and spends one call per segment. Static stretches collapse to one call; busy
stretches get as many as they earn.

The change signal costs **~1.2 ms per frame** on CPU, measured, with no
detector in the hot path — the correction to the 120 ms mistake above.

```python
from frameroute import FrameRouter, analyze_video

track = analyze_video("footage.mp4", weights="surveillance")
sel   = FrameRouter().select(track, budget_per_hour=240)

for pick in sel.picks:
    embedding = your_expensive_model(frame_at(pick.idx))   # VLM, CLIP, whatever
```

Two guarantees, both about the signal and neither about semantics:

- **hard trigger** — any frame whose novelty exceeds a threshold is selected
  regardless of budget, so a 0.4-second event is not averaged away
- **max gap** — no stretch longer than `max_gap_sec` goes uncovered

Whether a covered change was *meaningful* is an empirical question. `bench/`
answers it; the router does not assert it.

### 2. The multimodal search stack — the part that was always good

`adve_v2/adve/` indexes a video across three orthogonal signals and merges
them onto one timeline with source attribution:

| signal | module | what it catches |
|---|---|---|
| visual | `search/index.py` (CLIP + FAISS) | scenes, objects, composition |
| on-screen text | `vision/ocr_extractor.py` (EasyOCR + FTS) | slides, signage, scoreboards, UI |
| speech | `audio/indexer.py` (Whisper) | anything said |
| fusion | `vision/unified_search.py` | 3 s merge window, 8 s min gap |

On-screen text as a first-class, time-fused search signal is uncommon, and
it is decisive for lectures, screen recordings and meetings — the content
where CLIP alone reads worst.

---

## Results (measured)

Gate 1 (retrieval parity) and Gate 2 (router vs uniform) have now been run —
the experiments the earlier `docs/` skipped. The trained anchor-delta
reconstruction did not survive them and has been **removed**: the pipeline is
now CLIP-only (`adve/core/pipeline.py`, `adve/core/lean_indexer.py`), so every
stored embedding is a real model call and nothing is synthesised.

**Cost + quality** — 6 real lectures, routed frames vs full compute, judged by
a real VLM (Gemini) answering the same question of each frame:

| metric | value |
|---|---|
| answer parity | median **100%**, mean ~93% (range 75–100%) |
| model-call reduction at parity | ~4–7× (**median 5.4×**) |
| cost cut | ~**80%** |

**Latency** — end-to-end indexing on a laptop GPU (RTX 4050):

| metric | value |
|---|---|
| change signal | **1–2 ms/frame** |
| throughput | 4.5–12× realtime (**1 hr of video in ~5–13 min**) |
| vs encoding every frame | **~3× faster** |

**Where the router beats uniform:** unevenly-paced footage — lectures won 4/5
matched-budget tests (up to +0.275 recall). On continuous motion (traffic,
busy CCTV) plain uniform sampling ties it. Sell on the former, not the latter.

Reproduce: `python -m bench.cost_parity --video <lecture> --auto-span` and
`python -m bench.latency --video <lecture>`. These are single-clip-per-domain
samples — a strong directional signal, not yet a published benchmark.

---

## Quick start

All commands run from `adve_v2/`, where the packages live.

```bash
cd adve_v2
pip install -e .

# 1. verify the install with no GPU, no model, no API key
python -m bench.selftest

# 2. see what a policy would cost on your own footage — still no model
python -m frameroute cost your_video.mp4 --usd-per-call 0.002

# 3. look at the change signal
python -m frameroute signals your_video.mp4

# 4. select frames under a budget
python -m frameroute route your_video.mp4 --budget-per-hour 240
```

`cost` is the one to run in front of a customer: it counts the calls each
policy would make and multiplies by a price you supply. No inference, no
credits, and it sizes the saving on their own video in about a minute.

---

## The gates

Run in order. Each can kill the next. **No pricing, packaging or outreach
until all four clear** — the entire `docs/` folder was written before gate 1.

| | question | command | pass |
|---|---|---|---|
| **1** | Does skipping frames preserve *retrieval*? | `python -m frameroute gate1 VIDEO` | temporal recall ≥ 0.95 |
| **2** | Does change-aware routing beat uniform sampling at the *same* budget? | `python -m frameroute gate2 VIDEO` | beats best baseline by ≥ 0.02 on ≥3 of 5 corpora |
| **3** | Is it cheaper in *seconds and calls*? | same run | ≥ 5× cost cut at ≤ 5% recall loss |
| **4** | Will anyone pay? | 10 conversations | 3 LOIs |

### What Gate 1 found

It has now been run (see **Results** above). Zero-parameter fills beat the
trained reconstructor on retrieval, so the trained GRU / UALW stack was
**removed** — the pipeline is CLIP-only and nothing is synthesised. The
honest quality metric is *answer parity* (does a VLM give the same answer
from the routed frame as from full compute), not CLIP timestamp-recall, which
understates static-lecture footage because the routed index returns the
right *slide* at a slightly different *timestamp*.

```bash
python -m bench.cost_parity --video <lecture>.mp4 --auto-span   # cost + retrieval parity
python -m bench.answer_grade --video <lecture>.mp4              # free-form VLM answer parity (needs a vision API key)
python -m bench.latency --video <lecture>.mp4                   # wall-clock / throughput
```

---

## Layout

```
adve_v2/
  frameroute/        the router — no detector, no encoder in the hot path
    signals.py       cheap change signals (~1.2 ms/frame, measured)
    router.py        equal-accumulated-novelty allocation + guarantees
    policies.py      the baselines it has to beat
    adapters.py      model-agnostic downstream + honest call accounting
    cli.py           route / signals / cost / gate1 / gate2

  bench/             the experiments
    parity.py        retrieval parity + decode-once frame cache
    cost_parity.py   reduction-at-parity + $ per hour (CLIP proxy or live VLM)
    answer_grade.py  free-form VLM answer parity with an LLM judge
    latency.py       end-to-end wall-clock / throughput
    routing_bench.py matched-budget comparison, cost curve
    queries.py       query sets, labelled ground truth, OCR/transcript mining
    selftest.py      verify the install with no GPU and no model

  adve/              the multimodal search stack (visual + OCR + speech)
    core/lean_indexer.py   route -> CLIP-encode -> index (no reconstruction)
    core/pipeline.py       CLIP-only frame processing
  docs/              commercial drafts written before the gates. Not evidence.
  results/           benchmark JSON. Trust the .json, not the .md summaries.
```

---

## Honest limitations

- **Guarantees are about the signal, not about meaning.** The router can
  promise it never skipped a large measured change. It cannot promise the
  change mattered. Any product copy saying "zero missed events" without a
  labelled event list is doing what the v3.1 docs did.
- **The router's edge is domain-specific.** It beats uniform sampling on
  unevenly-paced footage (lectures, meetings, mostly-idle CCTV) and only
  ties it on continuous motion (traffic, busy streets, fast sports). Measured,
  not assumed — so don't sell it on the footage where it ties.
- **Sample size is small.** The results above are single-clip-per-domain with
  coarse query sets. A strong directional signal, not a published benchmark;
  the rigorous graded metric (`bench/answer_grade.py`) needs a paid vision key.
- **The architecture is not novel.** Change-aware sampling is prior art
  (Deep Feature Flow, CoViAR, AdaFocus, Skip-Convolutions, Eventful
  Transformers, SCSampler). The value is execution, honesty, and integration —
  not invention. Don't claim otherwise.
- **Cost figures depend entirely on the downstream model.** The savings are
  small against CLIP (it is cheap) and large against a VLM (it is not). Say
  which one you mean, every time.

---

## Licence

MIT. See `LICENSE`.
