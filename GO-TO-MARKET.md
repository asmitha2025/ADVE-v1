# Go to market — the operating kit

Everything buildable is built. The only unknown left is whether anyone pays.
This file exists so that question costs you a week, not a quarter.

**The one metric:** video-hours routed through you per week.
**The one goal right now:** five Frame Budget Audits on real customer footage.

---

## 1. Who to contact

Ranked by fit. The router's benefit depends on **pacing** — it pays where change
is bunched in time. Lead with the top two groups; we measured that footage.

### Tier 1 — proven footage (lectures, slides, talking heads)
| Company | Why them |
|---|---|
| Panopto | Lecture capture; entire product is searchable recordings |
| Kaltura | Video platform for education + enterprise |
| Coursera / Udemy / edX (2U) | Huge lecture libraries, active AI features |
| Docebo, 360Learning, Skillsoft | Corporate LMS, AI summarisation |
| Thinkific, Teachable | Course platforms, smaller and faster to reach |

### Tier 1 — meeting & call intelligence (screen shares + talking heads)
| Company | Why them |
|---|---|
| Gong, Chorus (ZoomInfo) | Sales-call intelligence at scale; heavy inference spend |
| Fireflies.ai, Otter.ai, Avoma | Meeting assistants, cost-sensitive, fast-moving |
| Grain, Fathom, tl;dv, Read.ai | Smaller, founder-reachable, will reply |

### Tier 2 — video-RAG / AI apps (biggest $ saving per hour)
Reduct.video · Descript · Guru · Dashworks · Glean · and any startup posting
about "chat with video" or GPT-4o video costs. **These feel the pain most
acutely** — their margin is literally frames × model price.

### Tier 2 — screen recording / docs
Loom · Scribe · Tango · Guidde · Supademo

### Tier 3 — predicted, untested (idle CCTV)
Verkada · Rhombus · Spot AI · Coram AI · Ambient.ai
Long-static-then-event is the router's ideal case, but **we have not measured
it**. Say so if you approach them.

### Do not contact
Traffic analytics, crowd monitoring, live sports AI. We measured it: on
continuous motion, uniform sampling is as good. Selling there is dishonest and
you will be found out.

---

## 2. The cold email

Short, specific, offers free work, and discloses the weakness. That last part is
the differentiator — everyone claims a magic algorithm; almost nobody publishes
what failed.

> **Subject:** How few frames does your video pipeline actually need?
>
> Hi {Name} — I built a small open-source tool that measures how many frames you
> can skip before your video model's answers get worse.
>
> On two lecture datasets, sending **3–4× fewer frames** left retrieval *better*
> than sending every frame — scored against the slides' own text, not a model
> grading itself. That's roughly **80% off the vision-model bill**.
>
> I'll run it free on one hour of your video and send you the number: how few
> frames you actually need, and what it saves at your current pricing. No
> integration, no signup — just the file.
>
> Worth an hour of your footage?
>
> — {You}
>
> *(Straight answer on the weak part: my frame-selection algorithm did **not**
> beat plain uniform sampling in testing. The saving comes from sending fewer
> frames, not from my picker being clever. Full results, including what failed:
> {evidence link})*

### Follow-up (once, after 4 working days)
> Hi {Name} — bumping this once in case it got buried.
>
> Simplest version: send one hour of video, I send back a number for what you
> could stop paying for. Free, no integration. If it's not interesting, a "no"
> is a completely fine reply and I won't chase.

### If they say "we already sample at 1 fps"
> That's exactly the baseline I measure against. The question is whether 1 fps
> is already too much for your content — on the lectures I tested, 1 frame every
> 4 seconds held retrieval just as well. The audit tells you where your line is,
> on your footage.

---

## 3. The audit runbook

When someone sends a video, this is the whole job. Budget ~30 minutes of
attention and one GPU hour.

```bash
cd adve_v2

# The whole audit, one command. It runs all three measurements below and
# writes the one-page deliverable from §4, caveats attached.
python -m bench.audit --video "CUSTOMER.mp4" --budget 80 --max-frames 400 \
    --price-per-call 0.005 --volume-hours 500 \
    --out results/CUSTOMER_audit.json --md results/CUSTOMER_audit.md

# Equivalent via the product CLI:
#   frameroute audit "CUSTOMER.mp4" --price-per-call 0.005 --volume-hours 500
```

The three stages it wraps, if you need to run one on its own:

```bash
# 1. Retrieval vs independent ground truth (OCR-mined queries, no API key).
python -m bench.grounded_eval --video "CUSTOMER.mp4" --budget 80 --max-frames 400

# 2. Throughput / latency - ops teams block on this.
python -m bench.latency --video "CUSTOMER.mp4" --max-frames 3000

# 3. Price it at THEIR model's rate (ask them; don't guess).
python -m bench.cost_parity --video "CUSTOMER.mp4" --auto-span \
    --price-per-call 0.005 --backend clip
```

If the footage has no OCR-readable text (no slides, no screen content), the
audit tells you retrieval could not be measured and prints the saving as
`--skip-retrieval` does: counted calls only, **not quality-safe**. Say that in
the email rather than quoting a quality claim.

**Read the output honestly before sending:**
- If `hit@5` for frameroute is at or above full compute → the saving is real, say so.
- If `parity_met` is false → say the bar was not reached and report the budget
  where it was. Never quote a reduction at an unmet quality bar.
- If uniform beats frameroute at matched calls → **tell them**. The saving still
  stands; our picker just isn't the reason. That honesty is why they'll trust
  the number.

---

## 4. What you send back

Keep it to one page. Numbers, not adjectives.

> **Frame Budget Audit — {Company}, {video name}**
>
> **Your safe budget:** 1 frame in {N} (you currently send ~1 fps)
> **Retrieval at that budget:** hit@5 {x} vs {y} for full compute
> **Calls:** {full} → {routed} per hour of video
> **At your {model} pricing:** ${a}/hr → ${b}/hr — **saving ~${c} per video-hour**
> **At your volume ({hrs}/month):** ~${d}/month
> **Indexing speed:** 1 hour of video in {m} minutes
>
> **Method:** queries mined from your own slide text; relevant frames are those
> whose slides contain the term. Scored against ground truth neither index
> produced.
>
> **Caveats:** {n} queries on {len} of footage. Our router showed no consistent
> advantage over uniform sampling, so this saving comes from sending fewer
> frames — you can get most of it with the open-source library, free.
>
> **To run it yourself:** `pip install frameroute`

Then one question, nothing more:
**"Would you want this running on your whole library, or is the library enough?"**

---

## 5. The pricing conversation

Only when they ask. Anchor on their saving, never on your cost.

- **"How much?"** → "$3 per video-hour for hosted. You'd be saving about $15
  per video-hour, so you keep the rest. The library is free forever if you'd
  rather run it yourself."
- **"Why not just use the open-source one?"** → "You should, if you have the
  engineering time. People pay us so they don't run it, and for the searchable
  index on top."
- **"Can we self-host?"** → Yes. Enterprise, annual licence, footage never
  leaves their network.
- **Do not** invent an SLA, a dashboard, or a delivery date. The hosted API is
  not built. Say "private beta, you'd be first" and mean it.

---

## 6. Reading the results

| Signal | What it means | Do |
|---|---|---|
| 1+ asks for a pilot on more footage | Real demand | Build the hosted path for them, specifically |
| Interested but "no budget/time" | Nice-to-have, not painful | Keep warm, move on |
| "We don't spend much on inference" | Wrong buyer | Go up-market to someone who does |
| 5/5 silence or no | **The market answer** | Stop. It cost a week |

**Stop condition:** if five teams are offered *free* work that saves them ~80%
and none engages, the thesis is wrong. That is a good outcome — you learned it
for the price of five emails instead of six months of engineering.

---

## 7. What not to do

- ❌ Don't build the hosted API before someone asks for it.
- ❌ Don't run more benchmarks on our own footage. Diminishing returns.
- ❌ Don't claim the router beats uniform. It didn't replicate.
- ❌ Don't deploy the HTTP server. The audit runs offline; you don't need it.
- ❌ Don't pitch video *search* — Azure Video Indexer and Twelve Labs own that.
  Pitch **cost**.
