"""Stage 3: speech-to-text with faster-whisper, keeping segment- and word-level timestamps.

The timestamps produced here are the "time index" that later links summary sentences
back to positions in the original film.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from .utils import fmt_ts, load_json, log, save_json


@lru_cache(maxsize=2)
def load_model(name: str, device: str, compute_type: str):
    """Load a faster-whisper model once per process."""
    from faster_whisper import WhisperModel

    log.info("Loading faster-whisper '%s' (%s, %s)", name, device, compute_type)
    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe(audio_path: str, asr_cfg: dict, out_json: str | None = None) -> dict:
    """Transcribe an audio file and return {"segments": [...], "info": {...}}.

    Each segment: id, start, end, text, avg_logprob, no_speech_prob, words[{start,end,word,prob}].
    Results are cached to out_json because transcription is the slowest stage on CPU.
    """
    if out_json and Path(out_json).exists():
        log.info("Using cached transcript: %s", Path(out_json).name)
        return load_json(out_json)

    model = load_model(asr_cfg["model"], asr_cfg["device"], asr_cfg["compute_type"])
    t0 = time.time()
    seg_iter, info = model.transcribe(
        audio_path,
        language=asr_cfg.get("language"),
        beam_size=asr_cfg.get("beam_size", 5),
        vad_filter=asr_cfg.get("vad_filter", True),
        word_timestamps=asr_cfg.get("word_timestamps", True),
        condition_on_previous_text=False,  # limits repetition loops on long films
    )
    segments = []
    for s in seg_iter:
        text = s.text.strip()
        if not text:
            continue
        segments.append({
            "id": len(segments),
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": text,
            "avg_logprob": round(s.avg_logprob, 3),
            "no_speech_prob": round(s.no_speech_prob, 3),
            "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word, "prob": round(w.probability, 3)}
                      for w in (s.words or [])],
        })
        if len(segments) % 100 == 0:
            log.info("  ... %d segments, reached %s", len(segments), fmt_ts(s.end))
    result = {
        "segments": segments,
        "info": {
            "model": asr_cfg["model"],
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "audio_duration": round(info.duration, 2),
            "processing_seconds": round(time.time() - t0, 1),
        },
    }
    log.info("Transcribed %d segments in %.1f s", len(segments), result["info"]["processing_seconds"])
    if out_json:
        save_json(result, out_json)
    return result


def write_srt(segments: list[dict], path: str) -> None:
    """Write segments as a SubRip subtitle file."""
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{fmt_ts(s['start'], srt=True)} --> {fmt_ts(s['end'], srt=True)}", s["text"], ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def transcript_text(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)
