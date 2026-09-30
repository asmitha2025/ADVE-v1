#!/usr/bin/env python
"""
Run cascade routing audit on the 4 benchmark lectures.
Evaluates different budget splits between cheap (CLIP) and expensive (VLM) tiers.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).parent.parent.parent / "dataset_lectures" / "video"

VIDEOS = [
    ("cities_and_decarbonization", VIDEO_DIR / "cities_and_decarbonization_standard_res.mp4"),
    ("computer_vision_2_2", VIDEO_DIR / "computer_vision_2_2_high_res.mp4"),
    ("deep_learning", VIDEO_DIR / "deep_learning_high_res.mp4"),
    ("numerics", VIDEO_DIR / "numerics_high_res.mp4"),
]

BUDGETS = [
    (120, 0),      # baseline: 120 VLM/hr, no cheap
    (80, 400),     # 80 VLM/hr + 400 CLIP/hr
    (60, 800),     # 60 VLM/hr + 800 CLIP/hr
    (40, 1200),    # 40 VLM/hr + 1200 CLIP/hr
    (0, 2000),     # all CLIP
]

EXPENSIVE_COST = 0.005
CHEAP_COST = 0.0002


def run_cascade(video_path: str, exp_bph: int, cheap_bph: int) -> dict:
    cmd = [
        sys.executable, "-m", "frameroute", "cascade",
        str(video_path),
        "--expensive-budget-per-hour", str(exp_bph),
        "--cheap-budget-per-hour", str(cheap_bph),
        "--expensive-cost", str(0.005),
        "--cheap-cost", str(0.0002),
        "--max-frames", "400",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        return {"error": result.stderr}
    
    # Parse output with regex
    out = result.stdout
    data = {"expensive_bph": exp_bph, "cheap_bph": cheap_bph}
    
    # Pattern: "expensive calls  1  ($0.0050/call = $0.01)"
    m = re.search(r"expensive calls\s+(\d+)\s+\([^$]*\$([\d.]+)/call = \$([\d.]+)\)", result.stdout)
    if m:
        data["expensive_calls"] = int(m.group(1))
        data["expensive_cost_per_call"] = float(m.group(2))
        data["expensive_total"] = float(m.group(3))
    
    m = re.search(r"cheap calls\s+(\d+)\s+\([^$]*\$([\d.]+)/call = \$([\d.]+)\)", result.stdout)
    if m:
        data["cheap_calls"] = int(m.group(1))
        data["cheap_cost_per_call"] = float(m.group(2))
        data["cheap_total"] = float(m.group(3))
    
    m = re.search(r"total calls\s+(\d+)", result.stdout)
    if m:
        data["total_calls"] = int(m.group(1))
    
    m = re.search(r"total cost\s+\$([\d.]+)", result.stdout)
    if m:
        data["total_cost"] = float(m.group(1))
    
    m = re.search(r"cost/hour\s+\$([\d.]+)", result.stdout)
    if m:
        data["cost_per_hour"] = float(m.group(1))
    
    return data


def main():
    results = {}
    for name, video in VIDEOS:
        print(f"\n=== {name} ===")
        video_results = []
        for exp_bph, cheap_bph in [(120, 0), (80, 400), (60, 800), (40, 1200), (0, 2000)]:
            print(f"  Expensive {exp_bph}/hr + Cheap {cheap_bph}/hr...", end=" ", flush=True)
            res = run_cascade(str(video), exp_bph, cheap_bph)
            if "error" in res:
                print(f" ERROR: {res['error'][:100]}")
                video_results.append({"expensive_bph": exp_bph, "cheap_bph": cheap_bph, "error": res["error"]})
            else:
                cost = res.get('cost_per_hour', 'ERR')
                print(f" ${cost:.2f}/hr (exp:{res.get('expensive_calls',0)}, cheap:{res.get('cheap_calls',0)})")
                video_results.append(res)
        results[name] = video_results
    
    # Print summary table
    print("\n\n=== COST SUMMARY ($/hour) ===")
    budgets = [(120, 0), (80, 400), (60, 800), (40, 1200), (0, 2000)]
    header = f"{'Video':<25}" + "".join([f"{exp:>4}/{cheap:<4}" for exp, cheap in [(120,0),(80,400),(60,800),(40,1200),(0,2000)]])
    print(header)
    for name, results_list in results.items():
        row = f"{name:<25}"
        for res in results_list:
            cost = res.get('cost_per_hour', 'ERR')
            if isinstance(cost, (int, float)):
                row += f"{cost:>8.2f}"
            else:
                row += f"{'ERR':>8}"
        print(row)
    
    # Save JSON
    out_path = Path(__file__).parent / "results" / "cascade_audit.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    import json
    main()