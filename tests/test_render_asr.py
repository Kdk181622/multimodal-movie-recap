"""Stage tests: OpenCV recap rendering (A/V sync, duration, frame accuracy) and speech-to-text."""

import cv2
import numpy as np
import pytest
import soundfile as sf

from recap.ingest import probe_video
from recap.render import render_recap, synthesize_narration
from recap.utils import ffmpeg_exe, read_audio_slice


def _cfg(movie, tmp_path):
    return {"output_dir": str(tmp_path), "input": {"local_path": movie},
            "recap": {"title_cards": True, "card_min_seconds": 1.0, "output_name": "recap.mp4"}}


def test_render_recap_duration_and_sync(synthetic_movie, tmp_path):
    clips = [{"peak": 1, "start": 1.0, "end": 3.0, "duration": 2.0, "start_hms": "00:00:01", "end_hms": "00:00:03",
              "transcript": "first clip"},
             {"peak": 2, "start": 4.8, "end": 6.4, "duration": 1.6, "start_hms": "00:00:04", "end_hms": "00:00:06",
              "transcript": "second clip"}]
    peaks = [{"index": 1, "timestamp": 1.5, "timestamp_hms": "00:00:01", "story_sentence": "A square moves across a dark room."},
             {"index": 2, "timestamp": 5.0, "timestamp_hms": "00:00:05", "story_sentence": "A loud noise fills the bright room."}]
    info = render_recap(_cfg(synthetic_movie, tmp_path), clips, peaks, [None, None], None, None, 8.0, "Synthetic")
    out = probe_video(info["path"])
    assert abs(out["duration_seconds"] - info["duration_seconds"]) < 0.1
    # Audio track length equals video length (sample-locked assembly).
    audio = read_audio_slice(info["path"], 0, info["duration_seconds"] + 1, 16000, 1)
    nonzero = np.nonzero(np.abs(audio[:, 0]) > 1e-4)[0]
    assert nonzero[-1] / 16000 <= info["duration_seconds"] + 0.1
    # The loud burst (source 5-6 s) must appear inside clip 2's slot in the recap.
    clip2 = [e for e in info["edl"] if e["type"] == "clip"][1]
    seg = audio[int(clip2["recap_start"] * 16000): int(clip2["recap_end"] * 16000), 0]
    assert np.abs(seg).max() > 0.4
    assert {e["type"] for e in info["edl"]} == {"intro", "chapter", "clip", "outro"}


def test_frame_accurate_seek(synthetic_movie):
    """Seeking by frame index returns that exact frame (the frame counter drawn in red differs per frame)."""
    cap = cv2.VideoCapture(synthetic_movie)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    for target in (13, 77, 150):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, f = cap.read()
        assert ok and np.abs(f.astype(int) - frames[target].astype(int)).mean() < 2.0
    cap.release()


def test_tts_and_whisper_round_trip(tmp_path):
    """Speak a sentence with the OS voice, then transcribe it with faster-whisper tiny."""
    voices = synthesize_narration(["The pianist hitchhikes to California."], tmp_path)
    if voices[0] is None:
        pytest.skip("No offline TTS engine on this machine")
    from recap.transcribe import transcribe

    import subprocess

    wav = tmp_path / "speech_44k.wav"
    sf.write(str(wav), voices[0], 44100)
    wav16 = tmp_path / "speech_16k_mono.wav"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(wav), "-ar", "16000", "-ac", "1", str(wav16)], check=True)
    res = transcribe(str(wav16), {"model": "tiny", "device": "cpu", "compute_type": "int8", "beam_size": 1,
                                  "language": "en", "vad_filter": False, "word_timestamps": True})
    text = " ".join(s["text"] for s in res["segments"]).lower()
    assert "california" in text
    words = [w for s in res["segments"] for w in s["words"]]
    assert all(a["start"] <= b["start"] for a, b in zip(words, words[1:])), "word timestamps must be monotonic"
