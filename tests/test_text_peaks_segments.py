"""Stage tests: summarisation helpers, story-to-film alignment, peak fusion and clip planning."""

import numpy as np
import pytest

from recap.peaks import build_windows, find_peaks, monotonic_alignment
from recap.segments import plan_clips
from recap.summarize import chunk_segments, extractive_summary, split_sentences, textrank_scores

EMB = "sentence-transformers/all-MiniLM-L6-v2"


def _cfg(tmp_path):
    return {
        "output_dir": str(tmp_path),
        "peaks": {"embedding_model": EMB, "window_seconds": 10, "window_hop_seconds": 5, "search_radius": 8,
                  "weights": {"semantic": 0.4, "centrality": 0.15, "text_emotion": 0.15, "audio": 0.15, "visual": 0.15}},
        "recap": {"max_duration_seconds": 40, "clip_min_seconds": 3, "clip_max_seconds": 8, "clip_pad_seconds": 0.3},
    }


def test_split_sentences():
    assert split_sentences("Al leaves. Vera arrives! Is it over? Yes.") == ["Al leaves.", "Vera arrives!", "Is it over?", "Yes."]
    assert split_sentences("Mr. Haskell drives. Charles Haskell Jr. sleeps.") == ["Mr. Haskell drives.", "Charles Haskell Jr. sleeps."]


def test_chunking_covers_all_segments(fake_segments):
    chunks = chunk_segments(fake_segments, chunk_seconds=20)
    ids = [i for c in chunks for i in c["segment_ids"]]
    assert ids == list(range(len(fake_segments)))
    assert all(c["end"] - c["start"] <= 24 for c in chunks)


def test_textrank_and_extractive(fake_segments):
    cent = textrank_scores(fake_segments, EMB)
    assert cent.shape == (len(fake_segments),) and cent.min() == 0 and cent.max() == pytest.approx(1)
    ext = extractive_summary(fake_segments, cent, target_words=30)
    assert ext["segment_ids"] == sorted(ext["segment_ids"])
    assert len(ext["text"].split()) >= 30


def test_monotonic_alignment_respects_order():
    # Sentence 1 matches column 4 best, but order forces it after sentence 0 (best at column 6).
    sim = np.array([[0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.1],
                    [0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5]])
    path = monotonic_alignment(sim, min_gap=1)
    assert path == [6, 7]
    assert monotonic_alignment(np.eye(4), min_gap=1) == [0, 1, 2, 3]


def test_windows(fake_segments):
    w = build_windows(fake_segments, 10, 5)
    assert w[0]["segment_ids"] == [0, 1]
    assert w[-1]["start"] < fake_segments[-1]["end"]


def _feats(n_seg, n_sec=60, loud_second=None):
    arousal = [0.1] * n_sec
    if loud_second is not None:
        arousal[loud_second] = 1.0
    return {"audio": {"arousal": arousal}, "visual": {"activity": [0.2] * n_sec, "shot_boundaries": [10.0, 20.0, 30.0]},
            "emotion": [{"label": "neutral", "intensity": 0.1}] * n_seg}


def test_peaks_follow_story(fake_segments, tmp_path):
    summary = {"story_sentences": ["Haskell falls out of the car and dies.", "Vera blackmails him over the car.",
                                   "The police arrest him on the highway."],
               "centrality": [0.5] * len(fake_segments)}
    res = find_peaks(_cfg(tmp_path), fake_segments, summary, _feats(len(fake_segments)), cache=False)
    ids = [p["peak_segment_id"] for p in res["peaks"]]
    assert ids == sorted(ids), "peaks must stay in film order"
    assert ids[0] == 5 and ids[1] == 8 and ids[2] == 11


def test_audio_signal_can_move_a_peak(fake_segments, tmp_path):
    """With equal text evidence, the louder neighbouring segment should win under audio-heavy weights."""
    summary = {"story_sentences": ["Haskell is in the car in the rain."], "centrality": [0.5] * len(fake_segments)}
    cfg = _cfg(tmp_path)
    w = {"semantic": 0.2, "audio": 0.8}
    res = find_peaks(cfg, fake_segments, summary, _feats(len(fake_segments), loud_second=26), weights=w, cache=False)
    assert res["peaks"][0]["peak_segment_id"] == 5  # segment 5 spans 25-29 s, which contains the loud second


def test_plan_clips_budget_and_order(fake_segments, tmp_path):
    peaks = [{"index": i + 1, "peak_segment_id": sid, "story_sentence": fake_segments[sid]["text"]}
             for i, sid in enumerate([1, 3, 5, 8, 10])]
    clips = plan_clips(_cfg(tmp_path), fake_segments, peaks, [10.0, 20.0, 30.0, 40.0], film_duration=60, overhead_seconds=5)
    assert len(clips) == 5
    assert sum(c["duration"] for c in clips) <= 35 + 1e-6
    for a, b in zip(clips, clips[1:]):
        assert a["end"] <= b["start"] + 1e-6, "clips must not overlap"
    for c, p in zip(clips, peaks):
        seg = fake_segments[p["peak_segment_id"]]
        assert c["start"] <= seg["start"] + 0.5 and c["end"] >= seg["end"] - 0.5, "clip must contain its peak"
