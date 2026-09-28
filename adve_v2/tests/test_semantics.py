"""
Offline tests for frameroute.semantics — content-space novelty.

No OCR, no CLIP: token maths, upsampling, fusion and the sparse-extraction
loop with an injected stub reader.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


def test_text_change_scores_slide_flip_and_text_loss():
    from frameroute.semantics import TextObservation, text_change_scores

    obs = [
        TextObservation(0.0, "gradient descent optimization neural networks"),
        TextObservation(2.0, "gradient descent optimization neural networks"),
        TextObservation(4.0, "convolutional filters pooling layers architecture"),
        TextObservation(6.0, "convolutional filters pooling layers architecture"),
        TextObservation(8.0, ""),
        TextObservation(10.0, ""),
    ]
    s = text_change_scores(obs)
    assert s[0] == 0.0
    assert s[1] == 0.0                      # same slide, OCR stable
    assert s[2] > 0.6                       # genuinely different slide, persisted
    assert s[3] == 0.0                      # same new slide
    assert s[4] == pytest.approx(0.6)       # text disappeared and stayed gone
    assert s[5] == 0.0                      # still no text


def test_ocr_jitter_does_not_fire():
    """A one-frame misread that reverts must not become a routing event."""
    from frameroute.semantics import TextObservation, text_change_scores

    a = "alpha beta gamma delta"
    b = "alpha beta gamma epsilon"
    obs = [
        TextObservation(0.0, a),
        TextObservation(1.0, a),
        TextObservation(2.0, b),   # OCR misread, reverts immediately
        TextObservation(3.0, a),
        TextObservation(4.0, a),
    ]
    s = text_change_scores(obs)
    assert s[2] == 0.0


def test_ubiquitous_footer_is_ignored():
    from frameroute.semantics import TextObservation, text_change_scores

    base = "university course slide"
    obs = [
        TextObservation(0.0, f"{base} alpha beta gamma delta"),
        TextObservation(1.0, f"{base} alpha beta gamma delta"),
        TextObservation(2.0, f"{base} alpha beta gamma epsilon"),
        TextObservation(3.0, f"{base} alpha beta gamma epsilon"),
        TextObservation(4.0, f"{base} zeta eta theta iota"),
    ]
    s = text_change_scores(obs)
    # The footer must not dilute the change between real content.
    assert s[2] > 0.3
    assert s[3] == 0.0
    assert s[4] > 0.5


def test_novelty_series_upsamples_onto_frame_times():
    from frameroute.semantics import TextObservation, text_novelty_series

    obs = [
        TextObservation(0.0, "alpha beta"),
        TextObservation(10.0, "gamma delta"),
    ]
    times = [0.0, 2.0, 5.0, 10.0, 12.0]
    s = text_novelty_series(obs, times)
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] == 0.0
    assert s[3] > 0.5
    assert s[4] == s[3]                     # held until the next observation


def test_fuse_track_is_a_floor_and_does_not_mutate():
    from frameroute.signals import FrameSignals, SignalTrack
    from frameroute.semantics import fuse_track

    sigs = [FrameSignals(idx=i, t=float(i), novelty=v)
            for i, v in enumerate([0.1, 0.2, 0.9, 0.05])]
    track = SignalTrack(video_path="x.mp4", fps=30.0, n_frames_total=4,
                        stride=1, signals=sigs)
    fused = fuse_track(track, np.array([0.0, 0.8, 0.0, 0.9]), weight=1.0)
    assert np.allclose(fused.novelty_array(), [0.1, 0.8, 0.9, 0.9])
    assert np.allclose(track.novelty_array(), [0.1, 0.2, 0.9, 0.05])
    # frame metadata is preserved
    assert [s.idx for s in fused.signals] == [0, 1, 2, 3]
    assert [s.t for s in fused.signals] == [0.0, 1.0, 2.0, 3.0]


def test_fuse_track_rejects_wrong_length():
    from frameroute.signals import FrameSignals, SignalTrack
    from frameroute.semantics import fuse_track

    track = SignalTrack(video_path="x", fps=30.0, n_frames_total=2, stride=1,
                        signals=[FrameSignals(idx=0, t=0.0),
                                 FrameSignals(idx=1, t=1.0)])
    with pytest.raises(ValueError):
        fuse_track(track, np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        track.with_novelty([0.1])


def test_extract_text_observations_with_stub_ocr(tmp_path):
    import cv2
    from frameroute.semantics import extract_text_observations

    path = str(tmp_path / "clip.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(50):
        out.write(np.full((64, 64, 3), i * 5 % 255, np.uint8))
    out.release()

    def stub_ocr(frames):
        return {i: f"slide {i // 10}" for i in frames}

    obs = extract_text_observations(path, every_sec=1.0, ocr_fn=stub_ocr)
    assert len(obs) == 5
    assert [o.t for o in obs] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert obs[0].text == "slide 0" and obs[4].text == "slide 4"
