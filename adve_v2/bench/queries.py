"""
bench.queries — query sets and labelled ground truth.

A retrieval benchmark is only as good as its queries. The existing
quality_report.json searched for "person" and "object", got top scores of
0.254 and 0.236, and recorded `"works": true`. Those are ordinary raw CLIP
text-image scores; they show the code returned something, not that it
returned the right thing. Queries have to be specific enough that a wrong
answer is visibly wrong.

Three things live here:
  DEFAULT_QUERIES     domain-appropriate starter queries, usable today
  QuerySet            schema for labelled ground truth (query -> time spans)
  propose_queries     mine candidate queries from the OCR / transcript
                      tables this repo already populates
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# starter queries
# --------------------------------------------------------------------------

DOMAIN_QUERIES: Dict[str, List[str]] = {
    "traffic": [
        "a motorcycle turning at the intersection",
        "a white car waiting at the signal",
        "a bus crossing the junction",
        "a person walking across the road",
        "an auto rickshaw in traffic",
        "an empty road with no vehicles",
        "vehicles queued at a red light",
        "a truck passing through the frame",
    ],
    "surveillance": [
        "a person entering through a doorway",
        "someone carrying a bag",
        "two people standing close together",
        "an empty corridor",
        "a person running",
        "someone bending down to the floor",
        "a person leaving the frame",
        "a crowd of people walking",
    ],
    "lecture": [
        # slide-type queries (structure)
        "a slide showing a mathematical equation",
        "a diagram with labelled axes",
        "the presenter writing on a whiteboard",
        "a slide with bullet points",
        "a plot or graph on screen",
        "code displayed on the slide",
        "a title slide",
        "a slide showing a table of numbers",
        "a slide with a single large heading",
        "a dense slide full of text",
        "a slide that is mostly a picture",
        "a hand-drawn sketch on screen",
        "a presenter speaking to the camera",
        "a screen recording of software",
        "a slide with a colored background",
        "a close-up of a chart legend",
        # content queries (phase-change / simulation lecture)
        "a phase diagram of a substance",
        "a temperature versus time graph",
        "molecules arranged in a regular lattice",
        "a simulation of particles moving",
        "a grid of cells changing state",
        "a plot of energy over time",
        "an equation describing heat transfer",
        "a colored heat map of a field",
    ],
    "action": [
        "a person walking down a street",
        "someone falling to the ground",
        "two people in physical contact",
        "a person standing still",
        "someone reaching toward an object",
        "a person moving quickly",
    ],
    "generic": [
        "a person in the centre of the frame",
        "an empty scene with no people",
        "a vehicle in motion",
        "a close-up of an object",
        "text visible on screen",
        "an outdoor daylight scene",
        "an indoor scene",
        "several people together",
    ],
}

DEFAULT_QUERIES: List[str] = DOMAIN_QUERIES["generic"]


def queries_for(domain: str) -> List[str]:
    return DOMAIN_QUERIES.get(domain, DEFAULT_QUERIES)


# --------------------------------------------------------------------------
# labelled sets
# --------------------------------------------------------------------------

@dataclass
class QuerySet:
    """
    A benchmark query set.

    labels maps query -> list of [start_sec, end_sec] spans that a correct
    system should return. Leave labels empty to run in oracle mode.

    events is separate: time spans that any routing policy must cover at
    least once, regardless of query. This is what turns "zero missed events"
    from a slogan into a measurement.
    """
    name: str
    video_path: str
    domain: str = "generic"
    queries: List[str] = field(default_factory=list)
    labels: Dict[str, List[List[float]]] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)   # {"t0","t1","label"}
    notes: str = ""

    def spans(self) -> Dict[str, List[Tuple[float, float]]]:
        return {q: [(float(a), float(b)) for a, b in v] for q, v in self.labels.items()}

    def event_spans(self) -> List[Tuple[float, float]]:
        return [(float(e["t0"]), float(e["t1"])) for e in self.events]

    @property
    def is_labelled(self) -> bool:
        return bool(self.labels)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str) -> "QuerySet":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return QuerySet(**d)


def load_query_set(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def new_query_set(video_path: str, domain: str = "generic", name: str = "") -> QuerySet:
    """Scaffold a query set ready for hand-labelling."""
    return QuerySet(
        name=name or Path(video_path).stem,
        video_path=video_path,
        domain=domain,
        queries=queries_for(domain),
        labels={},
        events=[],
        notes=(
            "Fill `labels` with query -> [[start_sec, end_sec], ...] to move from "
            "oracle mode to labelled mode. Fill `events` with the moments a router "
            "must not miss. Twenty labelled queries across five videos is enough "
            "to make Gate 1 meaningful."
        ),
    )


# --------------------------------------------------------------------------
# mining candidate queries from what the repo already indexed
# --------------------------------------------------------------------------

def propose_queries_from_ocr(
    ocr_db_path: str, video_id: str, limit: int = 20, min_len: int = 8
) -> List[str]:
    """
    Pull distinctive on-screen text out of the OCR database that
    adve/vision/ocr_extractor.py populates, and turn it into queries.

    These make unusually good benchmark queries because the ground truth is
    unambiguous: the text was either on screen at that timestamp or it was
    not. No human judgement required.
    """
    out: List[str] = []
    try:
        db = sqlite3.connect(ocr_db_path)
        cur = db.execute(
            "SELECT text FROM ocr_text WHERE video_id = ? ", (video_id,)
        )
        seen = set()
        for (txt,) in cur.fetchall():
            t = (txt or "").strip()
            if len(t) >= min_len and t.lower() not in seen and any(c.isalpha() for c in t):
                seen.add(t.lower())
                out.append(t)
            if len(out) >= limit:
                break
        db.close()
    except sqlite3.Error:
        return []
    return out


def propose_queries_from_transcripts(
    metadata_db_path: str, video_path: str, limit: int = 20, min_words: int = 4
) -> List[str]:
    """Pull distinctive spoken phrases from the transcripts table."""
    out: List[str] = []
    try:
        db = sqlite3.connect(metadata_db_path)
        cur = db.execute(
            "SELECT text FROM transcripts WHERE video_path = ?", (video_path,)
        )
        for (txt,) in cur.fetchall():
            t = (txt or "").strip()
            if len(t.split()) >= min_words:
                out.append(t)
            if len(out) >= limit:
                break
        db.close()
    except sqlite3.Error:
        return []
    return out


# --------------------------------------------------------------------------
# starter sets for the videos already in this repo
# --------------------------------------------------------------------------

REPO_TEST_VIDEOS: List[Dict[str, str]] = [
    {
        "path": "Testing videos/Traffic/Vodra/North.mp4",
        "domain": "traffic",
        "why": "high motion, fixed camera; ADVE measured 69.3% savings, 0.795 min sim",
    },
    {
        "path": "Testing videos/education videos/",
        "domain": "lecture",
        "why": "ADVE's worst domain (0.418 min sim) and OCR fusion's best; the "
               "most informative single test in the repo",
    },
    {
        "path": "Testing videos/Videos/walk/UCFCRIME_Abuse007_walk_1.mp4",
        "domain": "action",
        "why": "ADVE's best domain (96% savings, 0.944 min sim)",
    },
    {
        "path": "Input video/MOT17-02-SDP-raw.webm",
        "domain": "surveillance",
        "why": "the sequence every existing benchmark in this repo cites",
    },
]


def scaffold_repo_query_sets(out_dir: str = "bench/query_sets") -> List[str]:
    """
    Write one query-set stub per test video already in this repo, so Gate 1
    can be run today and hand-labelled incrementally.
    """
    written = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for v in REPO_TEST_VIDEOS:
        qs = new_query_set(v["path"], domain=v["domain"])
        qs.notes = v["why"] + "\n\n" + qs.notes
        p = str(Path(out_dir) / f"{Path(v['path']).stem or v['domain']}.json")
        qs.save(p)
        written.append(p)
    return written


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Scaffold one query-set stub per test video in this repo."
    )
    ap.add_argument("--out", default="bench/query_sets",
                    help="directory to write the query-set JSON files into")
    a = ap.parse_args()
    for p in scaffold_repo_query_sets(a.out):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
