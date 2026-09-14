# Deploying the frameroute site

`index.html` is a single self-contained file — no build step, no bundler, no
framework. Everything is inline except Google Fonts. That means you can host it
anywhere that serves a static file.

---

## Fastest path (pick one) — all free tiers

### 1. Netlify Drop — 30 seconds, no account needed to try
1. Open <https://app.netlify.com/drop>
2. Drag the **`website/` folder** onto the page.
3. You get a live URL immediately (e.g. `random-name.netlify.app`).
4. Sign in to keep it and attach a custom domain.

### 2. Vercel
```bash
npm i -g vercel
cd website
vercel            # follow prompts; accept defaults
vercel --prod     # promote to production
```

### 3. GitHub Pages (free, versioned with the repo)
```bash
# from the repo root, on a branch you are happy to publish
git subtree push --prefix website origin gh-pages
```
Then: **GitHub → Settings → Pages → Source: `gh-pages` branch → `/ (root)`**.
Site appears at `https://<user>.github.io/<repo>/`.

### 4. Cloudflare Pages
Connect the repo, set **Build command: _(none)_** and
**Build output directory: `website`**.

---

## Deployment to-do list

### Before you publish
- [ ] **Read the page end to end.** It deliberately publishes our failed
      claims. That is the point — do not quietly delete the "what we could not
      prove" section to make it look better.
- [ ] **Replace the audit call-to-action** with a real destination — a
      `mailto:` link, a Calendly, or a form. Right now the buttons are anchors
      to page sections only.
- [ ] **Add a contact address** in the footer. A landing page with no way to
      reach you converts nothing.
- [ ] Decide whether to link the internal calculator/report artifacts. They are
      **private to your Claude account** and will 404 for anyone else — either
      make them public or leave them out.

### Domain and delivery
- [ ] Buy a domain and point it at the host (all four options above support
      custom domains + automatic HTTPS).
- [ ] Confirm HTTPS is on (automatic on Netlify/Vercel/Pages/Cloudflare).
- [ ] Check the page on a phone — it is responsive, but look at it yourself.
- [ ] Check dark mode; the page follows the visitor's system theme.

### Analytics (optional but do it before outreach)
- [ ] Add a privacy-friendly analytics snippet (Plausible, Fathom, or
      Cloudflare Web Analytics) so you learn which section people read before
      they bounce. Skip Google Analytics unless you need it.

### Do **not** deploy these yet
- [ ] ❌ **The API server** (`adve_v2/adve/api/server.py`). Seven security
      defects were fixed, but it has never been load-tested, has no HTTPS
      termination of its own, stores API keys in plaintext SQLite, and has no
      tenancy isolation — `/v1/stats` exposes every task to every caller.
      The audit offer runs **offline** on footage a customer sends you, so you
      do not need the server to sell the first customers.
- [ ] ❌ Any claim that the router beats uniform sampling. It did not replicate.

---

## If you change the page
Edit `index.html` directly. Keep these intact, because they are the reason the
page is credible:

- the **"What we could not prove"** block,
- the **"Not production ready"** status block,
- the honest sample-size caveat (two lectures, 20 queries each).

A measurement company that hides a measurement has nothing left to sell.
