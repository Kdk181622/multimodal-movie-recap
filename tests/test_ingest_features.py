"""Stage tests: A/V extraction (ffmpeg / OpenCV) and per-modality features."""

import numpy as np
import soundfile as sf

from conftest import BURST, CUT_AT, FPS, SECONDS
from recap.features import audio_features, segment_signal, visual_features
from recap.ingest import extract_audio, probe_video
from recap.utils import fmt_ts, minmax


def test_probe_video(synthetic_movie):
    info = probe_video(synthetic_movie)
    assert info["fps"] == FPS
    assert abs(info["duration_seconds"] - SECONDS) < 0.2
    assert (info["width"], info["height"]) == (320, 240)


def test_extract_audio(synthetic_movie, tmp_path):
    out = extract_audio(synthetic_movie, str(tmp_path))
    y, sr = sf.read(out["wav_16k"])
    assert sr == 16000 and y.ndim == 1
    assert abs(len(y) / sr - SECONDS) < 0.2


def test_visual_shot_boundary_and_motion(synthetic_movie):
    v = visual_features(synthetic_movie, sample_fps=5, shot_threshold=0.35)
    # Exactly one hard cut, detected within one sample of 4 s.
    assert len(v["shot_boundaries"]) == 1
    assert abs(v["shot_boundaries"][0] - CUT_AT) <= 0.25
    # The moving square makes seconds 0-3 more active than the static seconds 5-7.
    assert np.mean(v["motion"][1:4]) > np.mean(v["motion"][5:8])


def test_audio_arousal_peaks_on_burst(synthetic_movie, tmp_path):
    wav = extract_audio(synthetic_movie, str(tmp_path))["wav_16k"]
    a = audio_features(wav)
    assert int(np.argmax(a["arousal"])) == int(BURST[0])


def test_helpers():
    assert fmt_ts(3725.5) == "01:02:05"
    assert fmt_ts(1.25, srt=True) == "00:00:01,250"
    assert minmax(np.array([2.0, 4.0, 6.0])).tolist() == [0.0, 0.5, 1.0]
    assert segment_signal([0.1, 0.9, 0.2], 0.5, 2.0) == 0.9
