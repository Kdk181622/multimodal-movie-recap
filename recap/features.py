"""Stage 4: per-modality signals used to locate peak moments.

* Audio  (librosa): RMS loudness, onset strength (spectral flux) and zero-crossing rate,
  pooled to one value per second -> an "arousal" curve.
* Video  (OpenCV): HSV-histogram shot-boundary detection and frame-difference motion,
  pooled to one value per second -> a "visual activity" curve.
* Text   (transformers): emotional intensity of every transcript segment,
  1 - P(neutral) from a DistilRoBERTa emotion classifier.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .utils import load_json, log, minmax, save_json


# ----------------------------------------------------------------------------- audio
def audio_features(wav_path: str, hop_seconds: float = 1.0) -> dict:
    """Return per-second audio descriptors and a combined arousal score in [0, 1]."""
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    hop = int(sr * 0.05)  # 50 ms frames, pooled to 1 s below
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=hop * 2, hop_length=hop)[0]

    per_bin = int(round(hop_seconds / 0.05))
    n_bins = int(np.ceil(len(y) / sr / hop_seconds))

    def pool(x: np.ndarray, fn=np.mean) -> np.ndarray:
        x = np.pad(x, (0, max(0, n_bins * per_bin - len(x))))[: n_bins * per_bin]
        return fn(x.reshape(n_bins, per_bin), axis=1)

    rms_s, onset_s, zcr_s = pool(rms), pool(onset), pool(zcr)
    # Peak-amplitude fluctuation inside each second: the "calm vs agitated" heuristic from class.
    rms_var = pool(rms, np.std)
    loud_db = 20 * np.log10(rms_s + 1e-6)
    arousal = 0.4 * minmax(loud_db) + 0.3 * minmax(onset_s) + 0.3 * minmax(rms_var)
    return {
        "hop_seconds": hop_seconds,
        "rms_db": loud_db.round(3).tolist(),
        "onset_strength": onset_s.round(4).tolist(),
        "zcr": zcr_s.round(4).tolist(),
        "rms_fluctuation": rms_var.round(5).tolist(),
        "arousal": arousal.round(4).tolist(),
    }


# ----------------------------------------------------------------------------- video
def _hsv_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def visual_features(video_path: str, sample_fps: float = 2.0, shot_threshold: float = 0.35) -> dict:
    """Scan the whole film once, sampling `sample_fps` frames per second.

    Shot boundaries: Bhattacharyya distance between consecutive HSV histograms > threshold.
    (Black-and-white films have flat hue/saturation, so the value channel does most of the work.)
    Motion: mean absolute grey-level difference between consecutive samples at 160 px width.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / sample_fps)))

    times, motion, hist_dist, brightness = [], [], [], []
    prev_hist, prev_grey = None, None
    for idx in range(total):
        # grab() decodes without converting the frame; retrieve() only on sampled frames keeps the scan fast.
        if not cap.grab():
            break
        if idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        small = cv2.resize(frame, (160, int(160 * frame.shape[0] / frame.shape[1])))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hist = _hsv_hist(small)
        times.append(idx / fps)
        brightness.append(float(grey.mean()))
        motion.append(0.0 if prev_grey is None else float(np.abs(grey - prev_grey).mean()))
        hist_dist.append(0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
        prev_hist, prev_grey = hist, grey
        if len(times) % 2000 == 0:
            log.info("  ... visual scan at %.0f s", times[-1])
    cap.release()

    times, motion, hist_dist = np.array(times), np.array(motion), np.array(hist_dist)
    cut_idx = np.where(hist_dist > shot_threshold)[0]
    # Enforce a minimum 1 s shot length so a flash or fade does not create a burst of cuts.
    shot_times: list[float] = []
    for i in cut_idx:
        if not shot_times or times[i] - shot_times[-1] >= 1.0:
            shot_times.append(round(float(times[i]), 2))

    # Pool to one value per second.
    n_sec = int(np.ceil(times[-1])) + 1 if len(times) else 0
    motion_s = np.zeros(n_sec)
    cuts_s = np.zeros(n_sec)
    sec_of = np.floor(times).astype(int)
    np.maximum.at(motion_s, sec_of, motion)
    for t in shot_times:
        cuts_s[int(t)] += 1
    # Cut density over a 10 s window captures fast editing (chases, fights, montages).
    # ("full" + centred slice keeps the length right even for clips shorter than the kernel.)
    cut_density = np.convolve(cuts_s, np.ones(10) / 10, mode="full")[5:5 + n_sec]
    activity = 0.6 * minmax(np.log1p(motion_s)) + 0.4 * minmax(cut_density)
    return {
        "fps": fps,
        "sample_fps": sample_fps,
        "shot_boundaries": shot_times,
        "motion": motion_s.round(4).tolist(),
        "cut_density": cut_density.round(4).tolist(),
        "brightness": np.array(brightness).round(2).tolist(),
        "activity": activity.round(4).tolist(),
    }


# ----------------------------------------------------------------------------- text
def text_emotion(segments: list[dict], model_name: str, batch_size: int = 32) -> list[dict]:
    """Classify each transcript segment; returns label, confidence and intensity = 1 - P(neutral)."""
    from transformers import pipeline

    clf = pipeline("text-classification", model=model_name, top_k=None, truncation=True, device=-1)
    texts = [s["text"] for s in segments]
    results = []
    for i in range(0, len(texts), batch_size):
        for scores in clf(texts[i:i + batch_size]):
            probs = {d["label"]: d["score"] for d in scores}
            top = max(probs, key=probs.get)
            results.append({"label": top, "confidence": round(probs[top], 4),
                            "intensity": round(1.0 - probs.get("neutral", 0.0), 4)})
    return results


# ----------------------------------------------------------------------------- stage runner
def compute_all(cfg: dict, segments: list[dict], wav_path: str) -> dict:
    """Run (or load cached) audio, visual and text-emotion features."""
    out = Path(cfg["output_dir"])
    cache = {"audio": out / "features_audio.json", "visual": out / "features_visual.json",
             "emotion": out / "features_text_emotion.json"}
    feats = {}
    if cache["audio"].exists():
        feats["audio"] = load_json(cache["audio"])
    else:
        log.info("Audio features (librosa)...")
        feats["audio"] = audio_features(wav_path)
        save_json(feats["audio"], cache["audio"])
    if cache["visual"].exists():
        feats["visual"] = load_json(cache["visual"])
    else:
        log.info("Visual features (OpenCV)... this reads the whole film once")
        fc = cfg["features"]
        feats["visual"] = visual_features(cfg["input"]["local_path"], fc["visual_sample_fps"], fc["shot_threshold"])
        save_json(feats["visual"], cache["visual"])
    if cache["emotion"].exists():
        feats["emotion"] = load_json(cache["emotion"])
    else:
        log.info("Text emotion over %d segments...", len(segments))
        feats["emotion"] = text_emotion(segments, cfg["features"]["emotion_model"])
        save_json(feats["emotion"], cache["emotion"])
    log.info("Features ready: %d shot boundaries detected", len(feats["visual"]["shot_boundaries"]))
    return feats


def segment_signal(curve: list[float], start: float, end: float, reducer=np.max) -> float:
    """Summarise a per-second curve over [start, end]."""
    if not curve:
        return 0.0
    a = min(int(start), len(curve) - 1)
    b = min(max(a + 1, int(np.ceil(end))), len(curve))
    return float(reducer(np.asarray(curve[a:b])))
