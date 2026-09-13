"""
bench.answer_grade — free-form answer parity (the metric that survives scrutiny).

The yes/no metric in bench.cost_parity is coarse. This is the rigorous version
a buyer's engineer would demand:

  for each of ~30 task queries:
     1. the FULL index and the ROUTED index each return their top frame
     2. a VLM DESCRIBES each frame in free text (not yes/no)
     3. a VLM JUDGE decides whether the two descriptions convey the same thing
  parity = fraction judged SAME.

This catches the failure the yes/no metric hides: a routed frame that is
"a slide" like the reference but a DIFFERENT slide. Uses Gemini via REST, with
exponential backoff on 429 so it survives a free-tier key slowly and a paid key
fast. Set GEMINI_API_KEY.

  python -m bench.answer_grade --video LECTURE.mp4 --reduction 6 --delay 1.5
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# 30 retrieval task-queries a real user of a lecture search tool would issue.
TASK_QUERIES: List[str] = [
    "a slide showing a graph or chart", "a slide with a mathematical equation",
    "the presenter speaking to the camera", "a slide that is mostly text",
    "a title or section-heading slide", "a diagram with labelled parts",
    "code shown on screen", "a table of numbers", "a photograph or real-world image",
    "a slide introducing a new topic", "a bulleted list of points",
    "a plot with two axes", "a hand-drawn sketch or annotation",
    "a slide with a single large figure", "a comparison between two things",
    "a flow chart or process diagram", "a map or geographic image",
    "a screenshot of software", "a slide summarising conclusions",
    "a formula being derived", "a bar chart", "a line graph over time",
    "a person pointing at the screen", "a dense technical slide",
    "a slide with a colored background", "a diagram of a network or graph",
    "an image of a physical experiment", "a slide asking a question",
    "a reference or citation list", "a closing or thank-you slide",
]


class Gemini:
    def __init__(self, model: Optional[str] = None, delay: float = 0.0,
                 max_retries: int = 5):
        self.key = os.environ["GEMINI_API_KEY"]
        # gemini-flash-latest is a stable alias that always resolves to a
        # current vision-capable flash model; pinned versions (e.g. 2.5-flash)
        # 404 for accounts created after they were retired. Override with
        # GEMINI_MODEL if needed.
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
        self.delay = delay
        self.max_retries = max_retries

    def _call(self, parts: list, max_tokens: int) -> str:
        # thinkingBudget=0 disables the model's internal reasoning tokens. Newer
        # flash models ("thinking" models) otherwise spend the whole output
        # budget reasoning and return an empty answer (finishReason=MAX_TOKENS),
        # which looks like a failure. Disabling it makes them answer directly.
        body = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_tokens, "temperature": 0.0,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        backoff = 2.0
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("x-goog-api-key", self.key)
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=45).read())
                if self.delay:
                    time.sleep(self.delay)
                cand = (r.get("candidates") or [{}])[0]
                cparts = (cand.get("content") or {}).get("parts") or [{}]
                txt = cparts[0].get("text", "")
                return txt.strip() if txt else f"__ERR__ empty({cand.get('finishReason')})"
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return f"__ERR__ {e.code}"
            except Exception as e:
                return f"__ERR__ {str(e)[:40]}"
        return "__ERR__ retries"

    @staticmethod
    def _img_part(frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        if w > 600:
            frame = cv2.resize(frame, (600, int(h * 600 / w)))
        b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
        return {"inline_data": {"mime_type": "image/jpeg", "data": b64}}

    def describe(self, frame: np.ndarray) -> str:
        return self._call([
            {"text": "Describe what this video frame shows in one concise sentence "
                     "(the main visual content: slide type, figures, text topic, people)."},
            self._img_part(frame),
        ], max_tokens=160)

    def judge(self, a: str, b: str) -> bool:
        out = self._call([
            {"text": "Two descriptions of two video frames follow. Reply with exactly "
                     "one word: SAME if they describe the same content/moment, or "
                     "DIFFERENT if not.\n"
                     f"A: {a}\nB: {b}"},
        ], max_tokens=12)
        return out.strip().upper().startswith("SAME")


def grade_clip(video: str, queries: List[str], reduction: float,
               max_frames: int, gem: Gemini, verbose: bool = True) -> dict:
    from frameroute.signals import analyze_video
    from frameroute.router import FrameRouter
    from frameroute.adapters import ClipEmbedder
    from bench.parity import build_reference_index, build_routed_index, read_frames

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    stride = max(1, total // max_frames) if total else 1

    emb = ClipEmbedder()
    track = analyze_video(video, stride=stride, max_frames=max_frames, weights="default")
    frames = read_frames(video, [s.idx for s in track.signals])
    ref = build_reference_index(video, track, emb, frames=frames)
    budget = max(2, int(round(track.n_analyzed / reduction)))
    picks = sorted(set(p.idx for p in FrameRouter().select(track, budget=budget).picks))
    cand = build_routed_index(video, track, emb, pick_indices=picks, fill="slerp", frames=frames)

    same = 0
    checked = 0
    errors = 0
    for q in queries:
        qv = emb.embed_text([q])[0]
        rt = ref.frame_idx[int(ref.index.search(qv, k=1)[0][0])]
        ct = cand.frame_idx[int(cand.index.search(qv, k=1)[0][0])]
        da = gem.describe(frames.get(rt))
        db = gem.describe(frames.get(ct))
        if da.startswith("__ERR__") or db.startswith("__ERR__"):
            errors += 1
            continue
        verdict = gem.judge(da, db)
        checked += 1
        same += int(verdict)
        if verbose:
            print(f"    [{'SAME' if verdict else 'DIFF'}] {q[:34]:36} | full={rt} routed={ct}", flush=True)

    parity = same / checked if checked else 0.0
    return {
        "video": os.path.basename(video), "reduction": round(ref.n_model_calls / max(len(picks), 1), 2),
        "full_calls": ref.n_model_calls, "routed_calls": len(picks),
        "queries_graded": checked, "errors": errors,
        "answer_parity": round(parity, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Free-form answer parity (Gemini judge).")
    ap.add_argument("--video", required=True)
    ap.add_argument("--reduction", type=float, default=6.0)
    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--n-queries", type=int, default=30)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between API calls (raise for free tier)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    gem = Gemini(delay=a.delay)
    res = grade_clip(a.video, TASK_QUERIES[:a.n_queries], a.reduction, a.max_frames, gem)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[answer_grade] wrote {a.out}")


if __name__ == "__main__":
    main()
