# Deploying the frameroute site

Static files, no build step, no framework. Everything is inline except Google
Fonts, so any static host will serve it as-is.

```
website/
  index.html      landing page (3D hero, pricing, integrations)
  evidence.html   the measurements, including what failed
  netlify.toml    publish config + security headers
  robots.txt      ← replace YOUR-DOMAIN.com
  sitemap.xml     ← replace YOUR-DOMAIN.com
  DEPLOY.md       this file
```

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

- [ ] **Swap the contact address.** Every CTA currently mails
      `baranitharan2020@gmail.com`. Replace it with a business address or a
      Calendly link — **8 places**: 7 in `index.html` (hero, pricing tier,
      the four cards in "How to reach us", final CTA, footer) and 1 in
      `evidence.html`. Find them with `grep -n "mailto:" *.html`.
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
