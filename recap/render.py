"""Stage 9-10: segment extraction with OpenCV and assembly of the final movie recap.

Video: every frame of the recap is produced with OpenCV - clips are read from the source with
frame-accurate seeking, annotated (original timecode, peak number, film-position bar) and written
with cv2.VideoWriter. Chapter cards carrying the summary-story sentences are drawn between clips.

Audio: assembled in parallel as a float32 sample buffer whose length is tied to the number of video
frames written so far, so audio and picture cannot drift apart. ffmpeg is used only at the end to
mux the two streams and encode H.264/AAC.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from .utils import fmt_ts, log, read_audio_slice, run_ffmpeg

SR = 44100
ACCENT = (64, 186, 255)  # BGR amber, used for badges and the position bar


# ----------------------------------------------------------------------------- text drawing
def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(text: str, font, max_width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def draw_text_block(img: np.ndarray, items: list[tuple]) -> np.ndarray:
    """Draw (text, (x, y), font, rgb) items with Pillow for clean anti-aliased TrueType text."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, xy, font, fill in items:
        draw.text(xy, text, font=font, fill=fill)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def position_bar(img: np.ndarray, start: float, end: float, film_duration: float, now: float | None = None) -> None:
    """Thin bar along the bottom edge showing where in the original film the clip comes from."""
    h, w = img.shape[:2]
    y = h - 6
    cv2.rectangle(img, (0, y), (w, h), (40, 40, 40), -1)
    x0, x1 = int(w * start / film_duration), max(int(w * end / film_duration), int(w * start / film_duration) + 2)
    cv2.rectangle(img, (x0, y), (x1, h), ACCENT, -1)
    if now is not None:
        xn = int(w * now / film_duration)
        cv2.rectangle(img, (xn - 1, y - 4), (xn + 1, h), (255, 255, 255), -1)


# ----------------------------------------------------------------------------- narration
def synthesize_narration(texts: list[str], out_dir: Path, enabled: bool = True) -> list[np.ndarray | None]:
    """Text-to-speech for chapter cards using the offline OS voice (pyttsx3 / Windows SAPI).
    Returns mono float32 arrays at 44.1 kHz, or None for each card when TTS is unavailable."""
    if not enabled:
        return [None] * len(texts)
    out_dir.mkdir(parents=True, exist_ok=True)
    # File names include a hash of the text, so an edited story never reuses stale audio.
    paths = [out_dir / f"narration_{i:02d}_{hashlib.md5(t.encode()).hexdigest()[:8]}.wav" for i, t in enumerate(texts)]
    todo = [(t, p) for t, p in zip(texts, paths) if not p.exists()]
    if todo:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            for text, path in todo:
                engine.save_to_file(text, str(path))
            engine.runAndWait()
        except Exception as exc:  # no TTS engine on this OS: cards stay silent
            log.warning("TTS unavailable (%s); chapter cards will be silent", exc)
            return [None] * len(texts)
    voices = []
    for p in paths:
        try:
            y, sr = sf.read(str(p), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            if sr != SR:
                import librosa

                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            voices.append(0.9 * y / (np.abs(y).max() + 1e-9))
        except Exception as exc:
            log.warning("Could not read %s (%s)", p.name, exc)
            voices.append(None)
    return voices


def card_duration(voice: np.ndarray | None, text: str, min_seconds: float) -> float:
    spoken = len(voice) / SR if voice is not None else len(text.split()) / 2.6
    return max(min_seconds, spoken + 0.9)


# ----------------------------------------------------------------------------- recap writer
class RecapWriter:
    """Keeps the OpenCV video stream and the audio buffer in lock-step."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.fps, self.size = fps, size
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise IOError(f"cv2.VideoWriter could not open {path}")
        self.frames = 0
        self.audio: list[np.ndarray] = []
        self.samples = 0

    @property
    def time(self) -> float:
        return self.frames / self.fps

    def add(self, frames: list[np.ndarray] | np.ndarray, audio: np.ndarray) -> None:
        for f in frames:
            self.writer.write(f)
            self.frames += 1
        # Pad or trim the audio so total samples == total video time * SR.
        need = int(round(self.frames / self.fps * SR)) - self.samples
        audio = audio[:need] if len(audio) >= need else np.vstack([audio, np.zeros((need - len(audio), 2), np.float32)])
        self.audio.append(audio.astype(np.float32))
        self.samples += len(audio)

    def close(self) -> np.ndarray:
        self.writer.release()
        return np.vstack(self.audio) if self.audio else np.zeros((0, 2), np.float32)


def _fade(audio: np.ndarray, ms: float = 25) -> np.ndarray:
    n = min(len(audio) // 2, int(SR * ms / 1000))
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def _speech_rms(y: np.ndarray, frame: int = 2205) -> float:
    """RMS over the louder half of 50 ms frames - ignores pauses so speech levels are comparable."""
    n = len(y) // frame
    if n == 0:
        return float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1))
    return float(np.mean(np.sort(rms)[n // 2:]))


def _card_frames(background: np.ndarray, items: list[tuple], seconds: float, fps: float) -> list[np.ndarray]:
    card = draw_text_block(background, items)
    n = max(1, int(math.ceil(seconds * fps)))
    fade = min(8, n // 4)
    frames = []
    for k in range(n):
        a = min(1.0, (k + 1) / fade, (n - k) / fade) if fade else 1.0
        frames.append(card if a >= 1.0 else cv2.convertScaleAbs(card, alpha=a))
    return frames


def _card_background(frame: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if frame is None:
        return np.full((h, w, 3), 18, np.uint8)
    blur = cv2.GaussianBlur(frame, (0, 0), 9)
    return cv2.convertScaleAbs(blur, alpha=0.32)


def _read_frame_at(cap: cv2.VideoCapture, t: float, fps: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    return frame if ok else None


def render_recap(cfg: dict, clips: list[dict], peaks: list[dict], voices: list, intro_voice, outro_voice,
                 film_duration: float, title: str) -> dict:
    """Write the recap MP4, its edit decision list (EDL) and subtitles. Returns render metadata."""
    out_dir = Path(cfg["output_dir"])
    rc = cfg["recap"]
    src = cfg["input"]["local_path"]
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w, h = size
    margin = int(w * 0.07)
    f_small, f_body, f_big = _font(max(12, w // 40), True), _font(max(16, w // 26)), _font(max(22, w // 17), True)

    tmp_video = out_dir / "_recap_video_only.mp4"
    rw = RecapWriter(tmp_video, fps, size)
    edl, subs = [], []
    peak_by_index = {p["index"]: p for p in peaks}

    # Read every clip's soundtrack first so the narration can be loudness-matched to the film.
    # (A 1945 soundtrack is far quieter than clean TTS; without matching, the dialogue gets buried.)
    clip_audio = []
    for clip in clips:
        f0, n_frames = int(round(clip["start"] * fps)), max(1, int(round(clip["duration"] * fps)))
        clip_audio.append(_fade(read_audio_slice(src, f0 / fps, n_frames / fps, SR, 2)))
    film_rms = float(np.median([_speech_rms(a.mean(axis=1)) for a in clip_audio])) if clip_audio else 0.1

    def add_card(kind: str, lines: list[tuple], seconds: float, voice, still, narration: str, peak=None):
        start = rw.time
        frames = _card_frames(_card_background(still, size), lines, seconds, fps)
        n_samples = int(round(len(frames) / fps * SR))
        audio = np.zeros((n_samples, 2), np.float32)
        if voice is not None:
            off = int(0.45 * SR)
            seg = voice[: max(0, n_samples - off)] * (film_rms / (_speech_rms(voice) + 1e-9))
            audio[off:off + len(seg)] = seg[:, None]
        rw.add(frames, audio)
        edl.append({"type": kind, "peak": peak, "recap_start": round(start, 2), "recap_end": round(rw.time, 2),
                    "source_start": "", "source_end": "", "text": narration})
        subs.append((start + 0.45, rw.time - 0.2, narration))

    # --- intro card
    still = _read_frame_at(cap, clips[0]["start"] if clips else 0, fps)
    intro_text = f"An AI-generated recap of {title}."
    y0 = int(h * 0.30)
    add_card("intro", [
        ("MULTIMODAL AI MOVIE RECAP", (margin, y0), f_small, (255, 186, 64)),
        (title, (margin, y0 + 30), f_big, (255, 255, 255)),
        (f"Original running time {fmt_ts(film_duration)}  |  {len(clips)} peak points", (margin, y0 + 90), f_body, (220, 220, 220)),
        ("Transcript > story > peak points > time index > clips", (margin, y0 + 122), f_small, (170, 170, 170)),
    ], max(4.0, card_duration(intro_voice, intro_text, 4.0)), intro_voice, still, intro_text)

    for n, clip in enumerate(clips, 1):
        p = peak_by_index[clip["peak"]]
        still = _read_frame_at(cap, clip["start"] + 0.5, fps)
        # --- chapter card: the summary-story sentence that this clip illustrates
        if rc["title_cards"]:
            body = _wrap(p["story_sentence"], f_body, w - 2 * margin)
            line_h = int(f_body.size * 1.35)
            top = max(int(h * 0.12), (h - (len(body) * line_h + 110)) // 2)
            items = [(f"CHAPTER {n} OF {len(clips)}", (margin, top), f_small, (255, 186, 64))]
            items += [(ln, (margin, top + 34 + i * line_h), f_body, (255, 255, 255)) for i, ln in enumerate(body)]
            items.append((f"Peak point at {p['timestamp_hms']}  ->  clip {clip['start_hms']} - {clip['end_hms']}",
                          (margin, top + 50 + len(body) * line_h), f_small, (190, 190, 190)))
            voice = voices[n - 1] if voices else None
            add_card("chapter", items, card_duration(voice, p["story_sentence"], rc["card_min_seconds"]),
                     voice, still, p["story_sentence"], peak=p["index"])

        # --- the extracted video segment
        start_t = rw.time
        f0 = int(round(clip["start"] * fps))
        n_frames = max(1, int(round(clip["duration"] * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames, last = [], None
        for k in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                frame = last if last is not None else np.zeros((h, w, 3), np.uint8)
            last = frame
            frame = frame.copy()
            src_t = (f0 + k) / fps
            # Original-timecode badge (top-left) and peak badge (top-right) make the time index visible.
            scale, box_h = 0.5 * w / 640, 8 + 24 * h // 480
            badge = f"ORIGINAL {fmt_ts(src_t)}"
            (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.rectangle(frame, (8, 8), (bw + 20, box_h), (0, 0, 0), -1)
            cv2.putText(frame, badge, (14, 8 + 18 * h // 480), cv2.FONT_HERSHEY_SIMPLEX, scale, ACCENT, 1, cv2.LINE_AA)
            tag = f"PEAK {p['index']}"
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.rectangle(frame, (w - tw - 22, 8), (w - 8, box_h), ACCENT, -1)
            cv2.putText(frame, tag, (w - tw - 15, 8 + 18 * h // 480), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
            # Mark the exact peak moment with a short white flash on the border.
            if abs(src_t - p["timestamp"]) < 0.5:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 3)
            position_bar(frame, clip["start"], clip["end"], film_duration, now=src_t)
            frames.append(frame)
        rw.add(frames, clip_audio[n - 1])
        edl.append({"type": "clip", "peak": p["index"], "recap_start": round(start_t, 2), "recap_end": round(rw.time, 2),
                    "source_start": clip["start"], "source_end": clip["end"], "text": clip["transcript"]})
        log.info("Clip %2d/%d  %s-%s  -> recap %s", n, len(clips), clip["start_hms"], clip["end_hms"], fmt_ts(start_t))

    # --- outro card
    outro_text = "That was the AI recap of the film."
    add_card("outro", [
        ("END OF RECAP", (margin, int(h * 0.36)), f_small, (255, 186, 64)),
        (title, (margin, int(h * 0.36) + 30), f_big, (255, 255, 255)),
        ("Public-domain film. Recap built from its audio, video and text.",
         (margin, int(h * 0.36) + 90), f_small, (190, 190, 190)),
    ], max(3.5, card_duration(outro_voice, outro_text, 3.5)), outro_voice, _card_background(None, size), outro_text)
    cap.release()

    audio = rw.close()
    tmp_audio = out_dir / "_recap_audio.wav"
    # Peak-normalise the whole soundtrack to -0.5 dBFS (lifts the quiet film audio, never clips).
    peak_level = float(np.abs(audio).max()) or 1.0
    sf.write(str(tmp_audio), audio * (0.94 / peak_level), SR, subtype="PCM_16")
    final = out_dir / rc["output_name"]
    run_ffmpeg(["-i", str(tmp_video), "-i", str(tmp_audio), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(final)])
    tmp_video.unlink(missing_ok=True)
    tmp_audio.unlink(missing_ok=True)

    duration = rw.frames / fps
    _write_edl(edl, out_dir / "recap_edl.csv")
    _write_recap_srt(edl, subs, out_dir / "movie_recap.srt")
    log.info("Recap written: %s (%.1f s = %s)", final.name, duration, fmt_ts(duration))
    return {"path": str(final), "duration_seconds": round(duration, 2), "frames": rw.frames, "fps": fps,
            "size": list(size), "edl": edl}


def _write_edl(edl: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(edl[0].keys()))
        wr.writeheader()
        wr.writerows(edl)


def _write_recap_srt(edl: list[dict], subs: list[tuple], path: Path) -> None:
    """Subtitles for the recap: narration on cards, a timecode note on clips."""
    entries = list(subs)
    for e in edl:
        if e["type"] == "clip":
            entries.append((e["recap_start"], min(e["recap_start"] + 3, e["recap_end"]),
                            f"[Original {fmt_ts(e['source_start'])} - {fmt_ts(e['source_end'])}]"))
    entries.sort()
    lines = []
    for i, (a, b, text) in enumerate(entries, 1):
        lines += [str(i), f"{fmt_ts(a, True)} --> {fmt_ts(b, True)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
