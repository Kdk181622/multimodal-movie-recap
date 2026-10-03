"""Shared fixtures: a synthetic 8-second movie with known structure.

* 0-4 s: dark frame with a moving white square; 4-8 s: static bright frame -> one hard cut at 4 s.
* Audio: quiet 220 Hz tone, with a loud noise burst between 5 and 6 s.
Known ground truth lets each component be tested in isolation before integration.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recap.utils import run_ffmpeg  # noqa: E402

FPS, W, H, SECONDS, SR = 25, 320, 240, 8, 16000
CUT_AT, BURST = 4.0, (5.0, 6.0)


@pytest.fixture(scope="session")
def synthetic_movie(tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("media")
    silent = d / "video_only.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        t = i / FPS
        if t < CUT_AT:
            frame = np.full((H, W, 3), 30, np.uint8)
            x = int(20 + 60 * t)
            cv2.rectangle(frame, (x, 90), (x + 50, 140), (255, 255, 255), -1)
        else:
            frame = np.full((H, W, 3), 210, np.uint8)
        cv2.putText(frame, f"{i:03d}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        writer.write(frame)
    writer.release()

    t = np.arange(SR * SECONDS) / SR
    audio = 0.05 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(0)
    burst = (t >= BURST[0]) & (t < BURST[1])
    audio[burst] = 0.8 * rng.uniform(-1, 1, burst.sum())
    wav = d / "audio.wav"
    sf.write(str(wav), audio.astype(np.float32), SR)

    movie = d / "synthetic_movie.mp4"
    run_ffmpeg(["-i", str(silent), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(movie)])
    return str(movie)


@pytest.fixture()
def fake_segments():
    """Twelve transcript segments of 4 s each with a 1 s gap."""
    texts = [
        "A pianist plays in a small nightclub in New York every night.",
        "His girlfriend Sue decides to leave for Hollywood to become a star.",
        "He starts hitchhiking across the country to reach California.",
        "A bookie named Haskell offers him a ride in his convertible.",
        "It starts raining and Haskell will not wake up in the car.",
        "Haskell falls out of the car and hits his head on a rock.",
        "He hides the body and takes the money and the clothes.",
        "He picks up a hitchhiker called Vera at a gas station.",
        "Vera knows the car belongs to Haskell and blackmails him.",
        "They rent an apartment in Hollywood to sell the car.",
        "Vera locks herself in the bedroom with the telephone and dies.",
        "He walks down the highway and the police pick him up.",
    ]
    return [{"id": i, "start": i * 5.0, "end": i * 5.0 + 4.0, "text": tx} for i, tx in enumerate(texts)]
