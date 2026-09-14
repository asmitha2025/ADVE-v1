# What to do next

Last updated: 15 September 2026. Everything buildable is built and pushed. The
only unknown left is whether anyone pays.

**The one metric:** video-hours routed through you per week.
**This month's goal:** five Frame Budget Audits on real customer footage.

---

## ✅ Done — don't redo it

| | |
|---|---|
| Code | CLIP-only pipeline, `pip install frameroute`, MIT, **27 tests green** |
| Security | 7 defects closed and verified (arbitrary file read, upload traversal, dev key, CORS, rate limiter, sports paths, retired chat model) |
| Evidence | Grounded eval on 2 lectures: **3–4× fewer calls, retrieval better than full compute** |
| Website | `website/` — landing + evidence page, deploy-ready |
| Docs | Dossier (+ `documents/frameroute-dossier.pdf`, 12pp), `GO-TO-MARKET.md`, `DEPLOY.md` |
| Backup | **Pushed to GitHub** — `asmitha2025/ADVE` |

---

## 🔴 This week — blocking, ~2 hours total

- [ ] **Rotate the four exposed API keys** *(5 min)* — three Gemini, one
      MiniMax. They were pasted in plain text in a chat log. Revoke and
      regenerate them; nothing in the repo depends on the old values.
- [ ] **Deploy the site** *(10 min)* — drag `website/` onto
      <https://app.netlify.com/drop>. `netlify.toml` is picked up automatically.
- [ ] **Swap the contact address** *(10 min)* — 8 `mailto:` links currently go
      to a personal Gmail. Use a business address or a Calendly.
      `grep -n "mailto:" website/*.html`
- [ ] **Replace the domain placeholders** *(2 min)* — `YOUR-DOMAIN.com` in
      `website/robots.txt` and `website/sitemap.xml`.
- [ ] **Send five emails** *(1 hr)* — targets, template and follow-up are in
      `GO-TO-MARKET.md`. Pick from Tier 1: lecture platforms (Panopto, Kaltura,
      Coursera) and meeting intelligence (Gong, Fireflies, Otter, Grain).

**Nothing else matters until these are done.** No further engineering changes
the answer to "will anyone pay".

---

## 🟡 When someone replies

- [ ] Ask for **one hour of video**, their **vision model**, and their
      **monthly video-hours** (so you price at their real rate, not a guess).
- [ ] Run the audit — exact commands in `GO-TO-MARKET.md` §3:
      ```bash
      cd adve_v2
      python -m bench.grounded_eval --video "CUSTOMER.mp4" --budget 80 --n-queries 20
      python -m bench.latency      --video "CUSTOMER.mp4" --max-frames 3000
      python -m bench.cost_parity  --video "CUSTOMER.mp4" --auto-span --price-per-call 0.005 --backend clip
      ```
- [ ] Send back the **one-page deliverable** (template in `GO-TO-MARKET.md` §4).
- [ ] **Report it honestly.** If uniform sampling ties us on their footage, say
      so — the saving still stands, our picker just isn't the reason. If routing
      won't help their content at all (continuous motion), tell them that
      instead of selling.

---

## 🟢 Only if a customer asks for it

Do **not** build these speculatively. Each is gated on a real request.

- [ ] **Hosted API** — the site calls it "private beta, not launched", which is
      true. Build it when someone wants keys, not before.
- [ ] **Server hardening** — the API server's 7 security defects are fixed, but
      it has never been load-tested, has no tenancy isolation
      (`/v1/stats` exposes every task to every caller) and stores keys in
      plaintext SQLite. Needed only for the hosted path. **The audit runs
      offline, so this blocks nothing today.**
- [ ] **Wire the server to `LeanIndexer`** — the live server still CLIP-encodes
      every processed frame rather than routing. Internal; no prospect sees it.
- [ ] **Budget-adaptive region gain** — `TODO` in `frameroute/signals.py`. Worth
      doing only when tuning to a specific customer's budget.
- [ ] **More benchmarks on our own footage** — diminishing returns. The next
      useful measurement is on *customer* data.

---

## ❌ Don't

- ❌ Claim the router beats uniform sampling. It won one lecture by +25% and
      lost the other by −13%. It did not replicate.
- ❌ Quote "median 100% answer parity" or "recall@10 0.912". Both were withdrawn
      after better measurement contradicted them.
- ❌ Deploy the HTTP server.
- ❌ Pitch video *search* — Azure Video Indexer and Twelve Labs own that. Pitch
      **cost**.
- ❌ Sell into traffic, crowds or fast sport. We measured it; uniform ties us.

---

## Stop condition

If five teams are offered **free** work that saves them ~80% and none engages,
the thesis is wrong. That is a good outcome: you learned it for the price of
five emails rather than six months of engineering.

If one asks for a pilot on more footage — that's the signal. Build the hosted
path for *them*, specifically.
