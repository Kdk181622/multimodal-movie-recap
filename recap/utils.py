"""Shared helpers: configuration, logging, ffmpeg access, JSON I/O and time formatting."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("recap")


def load_config(path: str | Path = PROJECT_ROOT / "config.yaml") -> dict:
    """Read the YAML config and resolve every project-relative path to an absolute one."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["input"]["local_path"] = str(PROJECT_ROOT / cfg["input"]["local_path"])
    cfg["input"]["reference_summary"] = str(PROJECT_ROOT / cfg["input"]["reference_summary"])
    out = PROJECT_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out)
    return cfg


def ffmpeg_exe() -> str:
    """Return a usable ffmpeg binary; imageio-ffmpeg ships one, so no system install is needed."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg quietly and raise with its stderr if it fails."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def read_audio_slice(path: str, start: float, duration: float, sr: int = 44100, channels: int = 2) -> np.ndarray:
    """Decode [start, start+duration) of any media file to float32 samples, shape (n, channels)."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
           "-i", path, "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).copy()  # writable copy
    # Pad or trim so the slice length is exact; this keeps audio locked to the video frames.
    target = int(round(duration * sr))
    if len(audio) < target:
        audio = np.vstack([audio, np.zeros((target - len(audio), channels), np.float32)])
    return audio[:target]


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def fmt_ts(seconds: float, srt: bool = False) -> str:
    """Format seconds as HH:MM:SS (or HH:MM:SS,mmm for SRT files)."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)) % 1000:03d}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1]; a constant array maps to zeros."""
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min() if x.size else 0.0
    return (x - x.min()) / span if span > 1e-12 else np.zeros_like(x)
