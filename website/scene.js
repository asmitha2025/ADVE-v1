/* frameroute — scroll-driven WebGL scene.
 *
 * The scene is not decoration: it runs the real routing algorithm in the
 * browser. A synthetic novelty curve is cut into equal-area segments and one
 * frame is picked per segment — the same rule `FrameRouter.select()` applies to
 * a real video. Move the budget slider and the picks are recomputed live.
 *
 * Everything here is progressive enhancement. If the module fails to load, the
 * CDN is blocked, or WebGL is unavailable, the page keeps its CSS background
 * and every word of content stays readable.
 */
import * as THREE from "three";

const canvas = document.getElementById("gl");
const root = document.documentElement;
if (canvas) boot();

function boot() {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mobile = matchMedia("(max-width: 820px)").matches;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas, antialias: !mobile, alpha: true, powerPreference: "high-performance",
    });
  } catch (e) {
    root.classList.add("no-gl");
    return;
  }
  if (!renderer.getContext()) { root.classList.add("no-gl"); return; }

  renderer.setPixelRatio(Math.min(devicePixelRatio, mobile ? 1.5 : 2));
  renderer.setSize(innerWidth, innerHeight);

  const scene = new THREE.Scene();
  const BG = new THREE.Color(0x07080a);
  scene.fog = new THREE.Fog(BG, 40, 300);

  const camera = new THREE.PerspectiveCamera(56, innerWidth / innerHeight, 0.5, 400);
  camera.position.set(0, 0, 18);

  /* ── the frame field ────────────────────────────────────────────────────
   * N frames strung down -Z on a slow helix. They all face +Z, so they read as
   * billboards toward a camera that only travels in Z — that keeps the matrix
   * update to a position/scale compose per instance instead of a lookAt.
   */
  const N = mobile ? 200 : 460;
  const SPACING = 2.45;
  const DEPTH = N * SPACING;

  const frames = [];
  for (let i = 0; i < N; i++) {
    const a = i * 0.216 + Math.sin(i * 0.031) * 0.7;
    frames.push({
      angle: a,
      baseR: 7.2 + Math.sin(i * 0.047) * 2.2 + (i % 7) * 0.42,
      z: -i * SPACING,
      tilt: (Math.sin(i * 1.7) * 0.06),
      novelty: 0,
      kept: 0,     // 1 when the router picks this frame
      lit: 0,      // eased toward `kept`
      r: 0,        // eased radius
      scale: 0.86 + ((i * 37) % 11) / 34,
    });
  }

  /* A novelty curve with bunched change: long flat stretches (a slide held on
   * screen) punctuated by sharp events (a slide change). This is the shape the
   * router exists to exploit, and the shape real lecture footage has. */
  const events = [];
  for (let k = 0; k < 15; k++) {
    events.push({ at: (k + 0.35 + Math.sin(k * 2.1) * 0.28) * (N / 15), w: 5 + (k % 4) * 3.5, h: 0.55 + ((k * 13) % 9) / 11 });
  }
  let peak = 0;
  for (let i = 0; i < N; i++) {
    let v = 0.035 + Math.abs(Math.sin(i * 0.09)) * 0.03;      // baseline drift
    for (const e of events) { const d = (i - e.at) / e.w; v += e.h * Math.exp(-d * d); }
    frames[i].novelty = v;
    if (v > peak) peak = v;
  }
  for (const f of frames) f.novelty /= peak;

  /* The actual algorithm: cumulative novelty cut into equal-area segments,
   * one call spent per segment. Flat stretches collapse; busy ones earn more. */
  const cum = new Float64Array(N);
  let run = 0;
  for (let i = 0; i < N; i++) { run += frames[i].novelty; cum[i] = run; }
  const total = run;

  function route(budget) {
    for (const f of frames) f.kept = 0;
    let lo = 0;
    for (let k = 0; k < budget; k++) {
      const target = (total * (k + 0.5)) / budget;
      while (lo < N - 1 && cum[lo] < target) lo++;
      frames[lo].kept = 1;
    }
    let n = 0;
    for (const f of frames) n += f.kept;
    return n;
  }

  const plane = new THREE.PlaneGeometry(2.05, 1.18);
  const mesh = new THREE.InstancedMesh(
    plane, new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true, opacity: 0.97 }), N);
  const glow = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(3.0, 2.1),
    new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true, opacity: 0.38, blending: THREE.AdditiveBlending, depthWrite: false }), N);
  mesh.frustumCulled = false; glow.frustumCulled = false;
  scene.add(mesh, glow);

  // Thin edge cards sitting just behind each frame give the plates an outline
  // without a second material or an outline pass.
  const edge = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(2.2, 1.3),
    new THREE.MeshBasicMaterial({ toneMapped: false, transparent: true, opacity: 0.55, depthWrite: false }), N);
  edge.frustumCulled = false;
  scene.add(edge);

  const AMBER = new THREE.Color(0xf0a03c);
  const AMBER_HOT = new THREE.Color(0xffc477);
  const NEUTRAL = new THREE.Color(0x2e3a49);
  const SIGNAL = new THREE.Color(0x8fb0d0);
  const DROPPED = new THREE.Color(0x141a21);
  const EDGE_OFF = new THREE.Color(0x44546a);

  // Scratch objects, reused every frame — nothing is allocated inside the loop.
  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const v3 = new THREE.Vector3();
  const s3 = new THREE.Vector3();
  const AXIS = new THREE.Vector3(0, 0, 1);
  const cA = new THREE.Color(), cB = new THREE.Color();

  // dust, for depth cues between the plates
  let dust = null;
  if (!mobile) {
    const D = 1400, pos = new Float32Array(D * 3);
    for (let i = 0; i < D; i++) {
      const a = Math.random() * Math.PI * 2, r = 4 + Math.random() * 26;
      pos[i * 3] = Math.cos(a) * r;
      pos[i * 3 + 1] = Math.sin(a) * r * 0.75;
      pos[i * 3 + 2] = -Math.random() * DEPTH;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    dust = new THREE.Points(g, new THREE.PointsMaterial({
      size: 0.075, color: 0x5d6b7d, transparent: true, opacity: 0.55, depthWrite: false,
    }));
    scene.add(dust);
  }

  /* ── scroll + input ─────────────────────────────────────────────────── */
  const stageEl = document.getElementById("experience");
  const slider = document.getElementById("budget");
  const ratioEl = document.getElementById("ratio");
  const cutEl = document.getElementById("cut");
  const callsEl = document.getElementById("liveCalls");
  const savedEl = document.getElementById("liveSaved");
  const acts = [...document.querySelectorAll("[data-act]")];

  let keepEvery = slider ? +slider.value : 4;
  let budget = Math.max(2, Math.round(N / keepEvery));
  let picked = route(budget);

  function updateReadout() {
    const perHour = 3600;
    const routed = Math.round(perHour / keepEvery);
    if (ratioEl) ratioEl.textContent = "1 in " + keepEvery;
    if (cutEl) cutEl.textContent = Math.round((1 - 1 / keepEvery) * 100) + "% fewer calls";
    if (callsEl) callsEl.textContent = routed.toLocaleString();
    if (savedEl) savedEl.textContent = "$" + ((perHour - routed) * 0.005).toFixed(2);
  }
  if (slider) {
    slider.addEventListener("input", () => {
      keepEvery = +slider.value;
      budget = Math.max(2, Math.round(N / keepEvery));
      picked = route(budget);
      updateReadout();
    });
  }
  updateReadout();

  /* Cinematic mode drives the acts from scroll. With reduced motion we leave
     the page in its default stacked layout and park the scene on the routed
     state, so the visitor still sees what routing looks like — just still. */
  const cinematic = !reduced;
  let progress = cinematic ? 0 : 0.86;
  let targetProgress = progress;
  function readScroll() {
    if (!cinematic || !stageEl) return;
    const span = stageEl.offsetHeight - innerHeight;
    targetProgress = span > 0 ? Math.min(1, Math.max(0, scrollY / span)) : 0;
  }
  if (cinematic) addEventListener("scroll", readScroll, { passive: true });
  readScroll();

  /* QA hook: `?fr-p=0.88` pins the journey at one point so a screenshot can
     capture a specific act without scripting a scroll and waiting on easing. */
  const pinned = parseFloat(new URLSearchParams(location.search).get("fr-p"));
  const isPinned = !Number.isNaN(pinned);
  if (isPinned) progress = targetProgress = Math.min(1, Math.max(0, pinned));

  let mx = 0, my = 0, tmx = 0, tmy = 0;
  if (!reduced && !mobile) {
    addEventListener("pointermove", (e) => {
      tmx = (e.clientX / innerWidth - 0.5);
      tmy = (e.clientY / innerHeight - 0.5);
    }, { passive: true });
  }

  addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
    readScroll();
  });

  const smoothstep = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };

  /* ── loop ───────────────────────────────────────────────────────────── */
  let running = true;
  document.addEventListener("visibilitychange", () => {
    running = !document.hidden;
    if (running) renderer.setAnimationLoop(tick);
    else renderer.setAnimationLoop(null);
  });

  let t0 = performance.now();
  const tBoot = t0;
  function tick(now) {
    const dt = Math.min(0.05, (now - t0) / 1000); t0 = now;
    const ease = reduced ? 1 : 1 - Math.pow(0.0015, dt);

    if (!isPinned) progress += (targetProgress - progress) * ease;
    mx += (tmx - mx) * ease; my += (tmy - my) * ease;
    const p = progress;

    // The plates materialise on a clock rather than on scroll, so the hero is
    // fully present the moment the page settles instead of being at its
    // dimmest exactly when every visitor first sees it.
    const reveal = reduced ? 1 : smoothstep(0, 1, (now - tBoot) / 1600);

    // Act blending. Each stage of the story maps to a slice of the scroll.
    const signal = smoothstep(0.30, 0.55, p);   // the change signal lights up
    const routing = smoothstep(0.58, 0.90, p);  // kept vs dropped separate

    // Fly down the corridor, then hand the page back to the content.
    camera.position.z = 18 - p * (DEPTH * 0.82);
    camera.position.x = mx * 3.2;
    camera.position.y = -my * 2.1;
    camera.rotation.y = -mx * 0.10;
    camera.rotation.x = my * 0.07;

    const drift = reduced ? 0 : now * 0.00006;

    for (let i = 0; i < N; i++) {
      const f = frames[i];
      f.lit += (f.kept - f.lit) * (reduced ? 1 : 1 - Math.pow(0.004, dt));

      // radius: kept frames pull into the stream, dropped ones fall away.
      // `near` stays outside the tube the camera flies down — pull them any
      // closer and a kept plate swells across the whole viewport.
      const near = 7.6, far = 18.5;
      const targetR = f.baseR + routing * (f.kept ? near - f.baseR : (far - f.baseR) * 0.55);
      f.r += (targetR - f.r) * (reduced ? 1 : 1 - Math.pow(0.02, dt));

      const a = f.angle + drift;
      const px = Math.cos(a) * f.r, py = Math.sin(a) * f.r * 0.72;

      // Collapse plates the camera is about to pass through. Without this a
      // kept frame at close range covers the copy in a slab of amber.
      const nearFade = smoothstep(2, 15, camera.position.z - f.z);
      const litScale = 1 + f.lit * routing * 0.3;
      const s = f.scale * (0.55 + 0.45 * reveal) * litScale * nearFade;
      q.setFromAxisAngle(AXIS, f.tilt + f.lit * routing * 0.03);

      v3.set(px, py, f.z);
      s3.set(s, s, 1);
      m4.compose(v3, q, s3);
      mesh.setMatrixAt(i, m4);

      // Border card and glow sit just behind the plate; coplanar copies would
      // z-fight with it rather than reading as an outline.
      v3.set(px, py, f.z - 0.04);
      s3.set(s * 1.02, s * 1.02, 1);
      m4.compose(v3, q, s3);
      edge.setMatrixAt(i, m4);

      v3.set(px, py, f.z - 0.09);
      s3.set(s * 1.15, s * 1.15, 1);
      m4.compose(v3, q, s3);
      glow.setMatrixAt(i, m4);

      // colour: neutral → signal-lit → routed (amber) or dropped (near black)
      cA.copy(NEUTRAL).lerp(SIGNAL, f.novelty * signal * 0.85);
      cB.copy(f.kept ? AMBER : DROPPED);
      cA.lerp(cB, routing * (f.kept ? f.lit : 0.8));
      cA.multiplyScalar(0.25 + 0.75 * reveal);
      mesh.setColorAt(i, cA);

      cA.copy(EDGE_OFF).lerp(AMBER_HOT, f.lit * routing);
      cA.multiplyScalar(0.2 + 0.8 * reveal);
      edge.setColorAt(i, cA);

      const gI = f.lit * routing * nearFade * (0.35 + 0.65 * Math.abs(Math.sin(now * 0.0011 + i)));
      cA.copy(AMBER).multiplyScalar(gI * 0.44);
      glow.setColorAt(i, cA);
    }
    mesh.instanceMatrix.needsUpdate = true;
    edge.instanceMatrix.needsUpdate = true;
    glow.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (edge.instanceColor) edge.instanceColor.needsUpdate = true;
    if (glow.instanceColor) glow.instanceColor.needsUpdate = true;

    if (dust) { dust.position.z = camera.position.z * 0.12; dust.rotation.z = drift * 0.5; }

    // Act captions: one visible at a time, driven by the same progress value.
    // Only ever touched in cinematic mode — otherwise the acts keep the plain
    // stacked layout the stylesheet gives them and stay readable.
    if (cinematic) {
      for (const el of acts) {
        const [from, to] = el.dataset.act.split(",").map(Number);
        const vis = smoothstep(from - 0.06, from + 0.04, p) * (1 - smoothstep(to - 0.04, to + 0.06, p));
        el.style.opacity = vis.toFixed(3);
        el.style.transform = "translateY(" + ((1 - vis) * 26).toFixed(1) + "px)";
        el.style.pointerEvents = vis > 0.5 ? "auto" : "none";
      }
    }

    renderer.render(scene, camera);
  }

  // Live counters during the routing act read straight off the picked count.
  if (callsEl) {
    const io = () => {
      const routed = Math.round(3600 / keepEvery);
      callsEl.textContent = routed.toLocaleString();
    };
    io();
  }

  renderer.setAnimationLoop(tick);
  // Hand the cinematic layout over only now, once the scene is proven running.
  if (cinematic) root.classList.add("gl-ready");
  void picked;
}
