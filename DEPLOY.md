# Deploying the frameroute site

Static files, no build step, no framework. CSS and page scripts are inline; the
only external requests are Google Fonts and Three.js from jsDelivr, so any
static host will serve it as-is.

```
website/
  index.html      landing page — scroll-driven WebGL scene, pricing, integrations
  scene.js        the 3D scene (ES module, imports Three.js from jsDelivr)
  evidence.html   the measurements, including what failed
  netlify.toml    publish config + security headers
  robots.txt      ← replace YOUR-DOMAIN.com
  sitemap.xml     ← replace YOUR-DOMAIN.com
  DEPLOY.md       this file
```

**The 3D scene degrades safely.** If WebGL is unavailable, the CDN is blocked,
or JavaScript is off, the canvas is hidden, a CSS gradient takes its place and
every word of the page still reads. The loader dismisses itself after 1.6s no
matter what, so a slow CDN can never leave a visitor on a black screen. With
`prefers-reduced-motion` the scroll journey flattens into ordinary stacked
sections.

**One deliberate inconsistency:** `index.html` is committed to dark (a lit
corridor of frames only reads against black); `evidence.html` follows the
visitor's system theme. That is intentional, not a bug.

---

## Deploy (pick one)

### Netlify — fastest
Drag the **`website/` folder** onto <https://app.netlify.com/drop>. Live in
seconds; `netlify.toml` is picked up automatically. Sign in to keep the URL and
attach a domain.

### Vercel
```bash
cd website && npx vercel --prod
```

### GitHub Pages
```bash
git subtree push --prefix website origin gh-pages
```
Then **Settings → Pages → Source: `gh-pages` / root**.

### Cloudflare Pages
Connect the repo. **Build command:** _(none)_ · **Output directory:** `website`

---

## Already done

- [x] `index.html` is the landing page; `evidence.html` holds the full results
- [x] Contact route wired — every CTA opens a pre-filled audit request email
- [x] Open Graph + Twitter card meta so shared links preview properly
- [x] Inline SVG favicon (no extra file to serve)
- [x] Security headers via `netlify.toml`
- [x] `robots.txt` + `sitemap.xml`
- [x] Verified no private Claude artifact links leak onto the public pages
- [x] The unbuilt hosted API is labelled **private beta / not launched**, and
      the API tier is a waitlist rather than a signup

## Before you announce it

- [x] **Contact address swapped** — all 8 CTAs now mail
      `hariharanm1802@gmail.com` (7 in `index.html`, 1 in `evidence.html`).
      If you later buy a business address or a Calendly, swap it again with
      `grep -n "mailto:" *.html`.
- [ ] **Replace `YOUR-DOMAIN.com`** in `robots.txt` and `sitemap.xml`.
- [ ] **Add an OG image** (1200×630 PNG) and reference it with
      `<meta property="og:image">` — link previews are much stronger with one.
- [ ] Buy a domain, point it at the host, confirm HTTPS (automatic on all four).
- [ ] Open it on a phone and in dark mode. It is responsive and theme-aware,
      but look at it yourself.
- [ ] Add privacy-friendly analytics (Plausible, Fathom or Cloudflare Web
      Analytics) so you learn where people stop reading before you do outreach.

## Do **not** deploy

- [ ] ❌ **The API server** (`adve_v2/adve/api/server.py`). Seven security
      defects were fixed and verified, but it has never been load-tested, has
      no tenancy isolation (`/v1/stats` exposes every task to every caller),
      and stores API keys in plaintext SQLite. The audit runs **offline** on
      footage a customer sends you, so nothing about this blocks selling.
- [ ] ❌ Any claim that the router beats uniform sampling. It did not replicate.

---

## If you edit the pages

Keep these three things. They are the reason the site is credible:

1. **"What we don't claim"** on `index.html` — the router did not beat uniform
   sampling, two quality claims were withdrawn, the sample is two lectures.
2. **"Where the product actually is today"** — the honest statement that the
   hosted API is not built.
3. **The status block** on `evidence.html` marking the server not production
   ready.

A company whose entire pitch is honest measurement cannot hide a measurement.
