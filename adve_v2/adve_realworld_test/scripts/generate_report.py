import os
import json
import argparse


def generate_executive_report(results_path: str, output_md: str):
    print(f"=== Generating Executive Validation Report from '{results_path}' ===")
    
    if not os.path.exists(results_path):
        print(f"Error: Results file '{results_path}' not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    badge = data.get("badge", "NEEDS IMPROVEMENT")
    overall = data.get("overall_metrics", {})
    domains = data.get("domain_breakdown", {})
    videos = data.get("video_results", [])

    md_content = f"""# ADVE Real-World Enterprise Validation Report

**Status**: `{badge}`  
**Generated At**: {data.get('timestamp', 'N/A')}

---

## 1. Executive Summary Metrics

| Metric | Measured Value | Enterprise Target | Status |
| :--- | :--- | :--- | :--- |
| **Overall Mean Cosine Similarity** | `{overall.get('mean_cosine_sim', 0.0):.4f}` | $\\ge 0.9850$ (or $\\ge 0.9400$) | {'✅ PASS' if overall.get('mean_cosine_sim', 0.0) >= 0.9400 else '❌ NEEDS RETRAINING'} |
| **Overall Min Cosine Similarity** | `{overall.get('min_cosine_sim', 0.0):.4f}` | $\\ge 0.9200$ (or $\\ge 0.8500$) | {'✅ PASS' if overall.get('min_cosine_sim', 0.0) >= 0.8500 else '❌ NEEDS RETRAINING'} |
| **Encoder Budget Savings** | `{overall.get('mean_encoder_savings_pct', 0.0):.1f}%` | $\\ge 70.0\\%$ | {'✅ PASS' if overall.get('mean_encoder_savings_pct', 0.0) >= 70.0 else '❌ FAIL'} |
| **Effective Throughput** | `{overall.get('mean_effective_fps', 0.0):.1f} FPS` | $\\ge 30.0\\text{{ FPS}}$ | {'✅ PASS' if overall.get('mean_effective_fps', 0.0) >= 30.0 else '⚠️ CPU / LIMITED'} |

---

## 2. Per-Domain Performance Breakdown

| Target Domain | Mean Cosine Similarity | Domain Status |
| :--- | :--- | :--- |
"""

    for dom, val in domains.items():
        status_str = "✅ EXCELLENT" if val >= 0.9700 else ("✅ PASS" if val >= 0.9400 else "❌ NEEDS FINE-TUNING")
        md_content += f"| `{dom}` | `{val:.4f}` | {status_str} |\n"

    md_content += """
---

## 3. Video-by-Video Validation Table

| Video Name | Domain | Total Frames | Encoder Calls | Savings % | Mean CosSim | Min CosSim | Effective FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for v in videos:
        md_content += f"| `{v.get('name')}` | `{v.get('domain')}` | {v.get('total_frames')} | {v.get('encoder_calls')} | {v.get('encoder_savings_pct'):.1f}% | `{v.get('mean_cosine_sim'):.4f}` | `{v.get('min_cosine_sim'):.4f}` | {v.get('effective_fps'):.1f} |\n"

    md_content += """
---

## 4. Recommendations & Commercial Next Steps

1. **Retraining & Fine-Tuning**: If domain mean CosSim drops below `0.9400`, incorporate additional domain clips into `generate_training_data.py` and run 15 epochs of `train_reconstructor.py`.
2. **Anchor Refresh Tuning**: For low-light or fast camera motion scenes, trigger local cluster refreshes or lower spatial thresholds to guarantee worst-case frame accuracy stay above `0.8500`.
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[OK] Generated Executive Validation Report -> '{output_md}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/realworld_report.json")
    parser.add_argument("--output", default="results/ADVE_RealWorld_Validation_Report.md")
    args = parser.parse_args()

    generate_executive_report(args.results, args.output)
