"""
Smoke tests for the benchmark modules (cost_parity, latency, answer_grade).

These do not spend API credits or run CLIP — they guard the parts that broke
before: import health, dataclass/table rendering (a bad format spec crashed
latency.table once), and the query-set contract. Anything needing a GPU or a
key is exercised by the CLI, not here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_modules_import():
    import bench.cost_parity as cp
    import bench.latency as lat
    import bench.answer_grade as ag
    import bench.semantic_bench as sb
    import bench.audit as au
    assert hasattr(cp, "run") and hasattr(cp, "ClipParityBackend")
    assert hasattr(lat, "run") and hasattr(lat, "LatencyReport")
    assert hasattr(ag, "grade_clip") and hasattr(ag, "Gemini")
    assert hasattr(sb, "run") and hasattr(sb, "SemanticReport")
    assert hasattr(au, "run_audit") and hasattr(au, "AuditReport")


def test_latency_report_table_renders():
    from bench.latency import LatencyReport
    r = LatencyReport(
        video="clip.mp4", device="cuda:0", seconds_of_video=120.0,
        frames_analyzed=3000, frames_encoded_routed=473,
        signal_ms_per_frame=2.02, encode_ms_per_call=18.2, decode_fps=236.4,
        routed_wall_sec=26.8, routed_realtime_factor=4.48, routed_hour_minutes=13.4,
        full_wall_sec=79.2, full_realtime_factor=1.52, full_hour_minutes=39.6,
        speedup_vs_full=2.96,
    )
    t = r.table()
    assert "LATENCY" in t and "1hr video" in t
    assert "4.5x" in t and "3.0x faster" in t          # format spec is valid


def test_cost_result_table_and_parity_flag():
    from bench.cost_parity import CostResult
    ok = CostResult(
        video="clip.mp4", domain="lecture", duration_sec=2462.0, frames_analyzed=800,
        price_per_call_usd=0.005, parity_threshold=0.9, quality_metric="temporal recall@10",
        full_calls=800, routed_calls=149, routed_quality=0.912, parity_met=True,
        uniform_calls_at_parity=400, reduction_x=5.4, usd_full_per_hour=5.85,
        usd_routed_per_hour=1.09, usd_saved_per_hour=4.76, saved_pct=81.0,
        router_beats_uniform=True, notes="",
    )
    assert "COST PARITY" in ok.table()
    assert "PARITY NOT MET" not in ok.table()          # met => no warning

    miss = CostResult(**{**ok.__dict__, "parity_met": False, "routed_quality": 0.45})
    assert "PARITY NOT MET" in miss.table()             # not met => warns honestly


def test_answer_grade_query_set():
    from bench.answer_grade import TASK_QUERIES
    assert len(TASK_QUERIES) == 30
    assert all(isinstance(q, str) and q for q in TASK_QUERIES)


def test_semantic_summary_renders(tmp_path):
    import json
    from bench.semantic_bench import summarize

    d = {
        "video": "clip.mp4",
        "full": {"hit_at_k": 0.450},
        "uniform": {"hit_at_k": 0.500},
        "router": {"hit_at_k": 0.600, "model_calls": 100},
        "router_text": {"hit_at_k": 0.550},
        "router_vs_uniform": 0.1, "text_vs_uniform": 0.05, "text_vs_router": -0.05,
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    table = summarize([str(p)])
    assert "| clip.mp4 | 100 | 0.450 | 0.500 | 0.600 | 0.550 |" in table
    assert "Mean over 1 lectures" in table


if __name__ == "__main__":
    test_modules_import()
    test_latency_report_table_renders()
    test_cost_result_table_and_parity_flag()
    test_answer_grade_query_set()
    print("[PASS] bench module smoke tests")
