"""
Offline tests for bench.audit — the Frame Budget Audit one-pager.

No CLIP, no OCR, no API keys: the stage functions are injected or fabricated
so the money math, the verdict logic and the honesty guardrails are what gets
tested. The heavy end-to-end path is exercised by running the audit itself.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _grounded(hit_full=0.45, hit_routed=0.75, hit_uniform=0.60, n_queries=20,
              full_calls=400, routed_calls=104, uniform_calls=104,
              duration=2462.0, frames=400):
    from bench.grounded_eval import GroundedReport
    def arm(name, calls, hit):
        return {"name": name, "model_calls": calls, "hit_at_k": hit,
                "precision_at_k": round(hit * 0.8, 3), "n_queries": n_queries}
    return GroundedReport(
        video="lecture.mp4", frames_analyzed=frames, duration_sec=duration,
        budget=routed_calls, k=5, n_queries=n_queries,
        full=arm("full", full_calls, hit_full),
        routed=arm("frameroute", routed_calls, hit_routed),
        uniform=arm("uniform", uniform_calls, hit_uniform),
        routed_vs_uniform_hit_delta=round(hit_routed - hit_uniform, 4),
        routed_vs_uniform_rel_gain_pct=25.0,
        routed_calls_saved_vs_full_pct=74.0,
    )


def _latency():
    from bench.latency import LatencyReport
    return LatencyReport(
        video="lecture.mp4", device="cpu", seconds_of_video=2462.0,
        frames_analyzed=400, frames_encoded_routed=104,
        signal_ms_per_frame=1.8, encode_ms_per_call=18.2, decode_fps=236.4,
        routed_wall_sec=26.8, routed_realtime_factor=4.48, routed_hour_minutes=13.4,
        full_wall_sec=79.2, full_realtime_factor=1.52, full_hour_minutes=39.6,
        speedup_vs_full=2.96,
    )


def test_assemble_money_and_verdict():
    from bench.audit import assemble
    rep = assemble(
        "lecture.mp4", grounded=_grounded(), latency=_latency(),
        price_per_call=0.005, volume_hours=500, domain="lecture",
    )
    assert rep.retrieval_available and rep.latency_available
    assert rep.full_calls == 400 and rep.routed_calls == 104
    # 400 calls over 2462s = 585/hr -> $2.92; 104 calls -> $0.76; saving $2.16
    assert rep.full_per_hour == pytest.approx(585.0, rel=0.01)
    assert rep.usd_full_per_hour == pytest.approx(2.92, abs=0.01)
    assert rep.usd_routed_per_hour == pytest.approx(0.76, abs=0.01)
    assert rep.usd_saved_per_hour == pytest.approx(2.16, abs=0.02)
    assert rep.saved_pct == pytest.approx(74.0, abs=0.1)
    assert rep.usd_saved_per_month == pytest.approx(rep.usd_saved_per_hour * 500, abs=1.0)
    assert "beat uniform" in rep.verdict
    assert rep.sec_per_call == pytest.approx(2462 / 104, rel=0.01)

    page = rep.one_pager()
    for must in ("Frame Budget Audit", "Your safe budget", "Retrieval at that budget",
                 "$ / video-hour", "Caveats", "What we do not claim",
                 "1 model call every"):
        assert must in page


def test_uniform_winning_is_reported_honestly():
    from bench.audit import assemble
    rep = assemble("lecture.mp4", grounded=_grounded(hit_routed=0.60, hit_uniform=0.75))
    assert "Uniform sampling beat" in rep.verdict
    page = rep.one_pager()
    assert "Uniform sampling beat" in page
    assert any("Router vs uniform" in c for c in rep.caveats)


def test_tie_is_not_dressed_up_as_a_win():
    from bench.audit import assemble
    rep = assemble("lecture.mp4", grounded=_grounded(hit_routed=0.70, hit_uniform=0.69))
    assert "Router tied uniform" in rep.verdict
    assert "beat uniform" not in rep.verdict


def test_retrieval_failure_marks_saving_unsafe(tmp_path):
    """An audit with no OCR ground truth must not present a quality-safe result."""
    import cv2
    import numpy as np
    from frameroute.signals import analyze_video
    from bench.audit import assemble

    rng = np.random.default_rng(0)
    path = str(tmp_path / "no_ocr.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
    for i in range(150):
        frame = np.full((90, 160, 3), 128, np.uint8) if i < 100 else \
            rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    track = analyze_video(path, stride=1, max_frames=150)
    rep = assemble(path, track=track, grounded=None, retrieval_note="no OCR labels",
                   price_per_call=0.005)
    assert not rep.retrieval_available
    assert rep.routed_calls > 0 and rep.full_calls == 150
    assert "Not measured" in rep.one_pager()
    assert "not** a quality-safe" in rep.one_pager()


def test_run_audit_with_injected_stages(tmp_path):
    from bench.selftest import make_test_video
    from bench.audit import run_audit

    video = str(tmp_path / "clip.mp4")
    make_test_video(video, w=160, h=90, fps=30)

    calls = {"retrieval": 0, "latency": 0}

    def fake_retrieval(video, budget, max_frames, k, n_queries, weights, device):
        calls["retrieval"] += 1
        return _grounded(duration=15.0, frames=450, full_calls=450, routed_calls=120)

    def fake_latency(video, max_frames, stride, bph, weights, device):
        calls["latency"] += 1
        return _latency()

    rep = run_audit(video, skip_retrieval=False, skip_latency=False,
                    retrieval_fn=fake_retrieval, latency_fn=fake_latency,
                    price_per_call=0.01, volume_hours=100)
    assert calls == {"retrieval": 1, "latency": 1}
    assert rep.retrieval_available and rep.latency_available
    assert rep.full_calls == 450 and rep.routed_calls == 120
    assert rep.usd_saved_per_hour > 0
    json_path = str(tmp_path / "audit.json")
    rep.to_json(json_path)
    assert os.path.exists(json_path)
