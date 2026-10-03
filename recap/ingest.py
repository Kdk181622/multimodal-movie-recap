"""Stage 1-2: YouTube download and audio/video separation.

The video stream stays in the MP4 and is read frame by frame with OpenCV later on.
The audio stream is demuxed to a 16 kHz mono WAV (speech-to-text input) and an MP3 copy.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .utils import ffmpeg_exe, log, run_ffmpeg


def download_youtube(url: str, out_path: str) -> str:
    """Download a YouTube video as MP4 (<=480p video + original-language audio) using yt-dlp."""
    out = Path(out_path)
    if out.exists():
        log.info("Input video already present: %s", out.name)
        return str(out)
    import yt_dlp

    out.parent.mkdir(parents=True, exist_ok=True)
    opts = {
        # Prefer the original English audio track: some uploads carry auto-dubbed tracks too.
        "format": "bv*[height<=480][vcodec^=avc1]+ba[language^=en]/bv*[height<=480]+ba/b[height<=480]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out.with_suffix("")) + ".%(ext)s",
        "ffmpeg_location": ffmpeg_exe(),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    if not out.exists():
        raise FileNotFoundError(f"Download finished but {out} was not created")
    return str(out)


def probe_video(path: str) -> dict:
    """Read basic stream properties with OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "path": path,
        "fps": fps,
        "frame_count": frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_seconds": frames / fps if fps else 0.0,
    }
    cap.release()
    return info


def extract_audio(video_path: str, out_dir: str) -> dict:
    """Demux the soundtrack: 16 kHz mono WAV for ASR and a 192 kbps MP3 for listening."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav, mp3 = out / "audio_16k_mono.wav", out / "audio.mp3"
    if not wav.exists():
        run_ffmpeg(["-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    if not mp3.exists():
        run_ffmpeg(["-i", video_path, "-vn", "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)])
    log.info("Audio extracted -> %s", wav.name)
    return {"wav_16k": str(wav), "mp3": str(mp3)}
