# ADVE / frameroute

**Budget-constrained frame selection and multimodal search for video AI pipelines.**

> This README was rewritten in September 2026 after an audit found that the
> project's headline claim did not survive its own benchmark. The previous
> version is preserved verbatim at [`README_v1_ARCHIVE.md`](../README_v1_ARCHIVE.md).
> Nothing below is asserted without a file you can open or a command you can run.

---

## What happened

The original claim was *"96.67% fewer CLIP encoder calls"*, achieved by
embedding only anchor frames and reconstructing the rest from spatial-graph
deltas of YOLO object tracks.

The call reduction was real. The **cost** reduction was not.

From [`results/traffic_benchmark_report.json`](results/traffic_benchmark_report.json),
same video, same machine, same 300 frames:

| | full CLIP | ADVE | delta |
|---|---:|---:|---:|
| time | 100.77 s | 95.47 s | **−5.3%** |
| throughput | 3.0 fps | 3.1 fps | +0.1 fps |
| memory | 1416 MB | 1668 MB | **+17.8%** |
| *calculated* GFLOPs | 1320 | 529 | −59.9% |

The 59.9% is arithmetic (frames × 4.4 GFLOPs). The 5.3% is a stopwatch.

The reason is in [`docs/GPU_PERFORMANCE_CERTIFICATE.md`](docs/GPU_PERFORMANCE_CERTIFICATE.md):

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

**Retrieval** — two real lectures, scored by `bench/grounded_eval.py` against
ground truth *neither index produced*: the lecture's own slide text (OCR).
Queries are mined from recurring slide terms; the relevant frames are the ones
whose slides contain the term. No VLM judging its own output.

| lecture (20 queries, top-5) | calls | hit@5 | precision@5 |
|---|---:|---:|---:|
| cities_and_decarb — full | 400 | 0.450 | 0.400 |
| cities_and_decarb — **frameroute** | 104 | **0.750** | 0.320 |
| cities_and_decarb — uniform | 104 | 0.600 | 0.240 |
| deep_learning — full | 400 | 0.450 | 0.450 |
| deep_learning — **frameroute** | 131 | 0.650 | 0.430 |
| deep_learning — uniform | 131 | **0.750** | 0.460 |

**What holds:** at **67–74% fewer calls**, hit@5 was *higher* than full compute
on both clips (0.65–0.75 vs 0.45). Skipping frames does not hurt retrieval —
that is what the ~**80% cost cut** rests on, and it replicated.

**What does not hold:** change-aware routing showed **no consistent advantage
over plain uniform sampling** — +25% on one lecture, −13% on the other, at
identical call counts. On this evidence the saving comes from *sending fewer
frames*, not from our selection algorithm. Do not claim otherwise.

**Content-space novelty was tested too** (`frameroute/semantics.py`,
`bench/semantic_bench.py`). On-screen text is where a lecture actually
changes, so the signal measures token-level text change with a persistence
filter — a change must stick to count; that alone cut spurious events from
252 to 105 on one lecture while keeping every true slide transition. Fused
as a floor over pixel novelty, four lectures at matched call counts:

| lecture (20 OCR queries, top-5) | calls | full | uniform | router | router+text |
|---|---:|---:|---:|---:|---:|
| cities_and_decarbonization | 104 | 0.450 | 0.600 | 0.750 | **0.800** |
| computer_vision_2_2 | 86 | 0.450 | 0.500 | **0.550** | **0.550** |
| deep_learning | 131 | 0.450 | 0.650 | **0.650** | **0.650** |
| numerics | 188 | 0.650 | 0.600 | **0.650** | 0.600 |

**Mean over four: pixel router +0.062 vs uniform, text-fused +0.062 vs
uniform, text-vs-pixel 0.000 — a wash.** The content signal is not an edge;
it ships as an optional module, and the summary is reproducible with
`python -m bench.semantic_bench --summary "results/semantic_*.json"`.

**Withdrawn:** earlier versions of this file quoted "median 100% answer parity"
and "lectures won 4/5 matched-budget tests". The first came from a lenient
yes/no VLM check that counted two *different* slides as a match (a stricter
free-form judge scored the same clips 20–35%); the second came from CLIP
timestamp-recall, which penalises returning the right slide a few seconds off.
Both metrics were unreliable and have been replaced by the grounded evaluation
above.

**Latency** — end-to-end indexing on a laptop GPU (RTX 4050):

| metric | value |
|---|---|
| change signal | **1–2 ms/frame** |
| throughput | 4.5–12× realtime (**1 hr of video in ~5–13 min**) |
| vs encoding every frame | **~3× faster** |

Reproduce: `python -m bench.grounded_eval --video <lecture> --budget 80` and
`python -m bench.latency --video <lecture>`. Two clips, 20 queries each — a
real signal, not yet a published benchmark.

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

### The whole audit, one command

```bash
python -m frameroute audit your_video.mp4 --price-per-call 0.005 \
    --volume-hours 500 --out results/audit.json --md results/audit.md
```

Runs retrieval parity, latency and the money calculation, then writes the
one-page deliverable — with the caveats attached. If retrieval cannot be
measured (no OCR-readable text), it says the saving is not quality-safe
instead of quietly quoting a reduction.

### Use it from Claude / Cursor (MCP)

A dependency-free MCP server lets an assistant price, route or audit a video
in the conversation:

```bash
python -m frameroute.mcp_server        # or: frameroute-mcp
```

```json
{"mcpServers": {"frameroute": {"command": "python",
  "args": ["-m", "frameroute.mcp_server"]}}}
```

Tools: `cost_video` (no model, seconds), `route_video`, `signals_video`,
`audit_video`.

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
    semantics.py     content-space novelty from OCR text (optional, measured)
    router.py        equal-accumulated-novelty allocation + guarantees
    policies.py      the baselines it has to beat
    adapters.py      model-agnostic downstream + honest call accounting
    cli.py           route / signals / cost / audit / gate1 / gate2
    mcp_server.py    same tools over MCP (stdio), no SDK dependency

  bench/             the experiments
    parity.py        retrieval parity + decode-once frame cache
    cost_parity.py   reduction-at-parity + $ per hour (CLIP proxy or live VLM)
    answer_grade.py  free-form VLM answer parity with an LLM judge
    latency.py       end-to-end wall-clock / throughput
    routing_bench.py matched-budget comparison, cost curve
    queries.py       query sets, labelled ground truth, OCR/transcript mining
    audit.py         one command -> the customer-facing Frame Budget Audit page
    semantic_bench.py content-space vs pixel routing, OCR-grounded
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
- **The router has no demonstrated edge over uniform sampling.** Across runs
  it won one lecture (+15–25%) and tied or lost the other (0 to −13%); over
  four lectures, content-space novelty tied pixel novelty exactly (mean
  +0.000). In the newest four-clip benchmark the pixel router averaged +0.062
  over uniform — a signal, not a moat. *Skipping* frames is what saves the
  money; our choice of *which* frames is not yet shown to beat the obvious
  baseline a customer would write in ten lines. Sell the saving, not the
  algorithm.
- **Sample size is small and the metrics are genuinely hard.** Two clips, 20
  queries each. Three quality metrics were tried and two discarded: a lenient
  yes/no VLM check (it counted two *different* slides as a match) and CLIP
  timestamp-recall (it penalises returning the right slide a few seconds off).
  A strict free-form judge (`bench/answer_grade.py`) scored 20–35% but marks a
  frame 7 seconds from the reference a miss. Trust `bench/grounded_eval.py`,
  which scores against the slides' own text rather than a model judging itself.
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
