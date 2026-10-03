"""Stage 6-7: peak-point identification and time-index mapping.

For every sentence of the summary story:
1. Alignment - the sentence is matched to a 30 s transcript window. A monotonic dynamic-programming
   alignment forces story order = film order, so a sentence about the ending cannot be matched to an
   opening scene just because they share words.
2. Peak picking - inside +/- search_radius seconds of the aligned window, every transcript segment is
   scored by post-fusion of five normalised signals:
       semantic (story sentence vs. segment text), TextRank centrality, text emotion intensity,
       audio arousal and visual activity.
   The highest-scoring segment is the peak point; its Whisper timestamp is the time index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import segment_signal
from .summarize import embed
from .utils import fmt_ts, load_json, log, minmax, save_json

MODALITIES = ("semantic", "centrality", "text_emotion", "audio", "visual")


def build_windows(segments: list[dict], window: float, hop: float) -> list[dict]:
    """Sliding transcript windows: all segments overlapping [t, t + window), stepped by `hop`."""
    end_time = segments[-1]["end"]
    windows, t = [], 0.0
    while t < end_time:
        ids = [s["id"] for s in segments if s["end"] > t and s["start"] < t + window]
        if ids:
            windows.append({"start": t, "end": t + window, "segment_ids": ids,
                            "text": " ".join(segments[i]["text"] for i in ids)})
        t += hop
    return windows


def monotonic_alignment(sim: np.ndarray, min_gap: int = 1) -> list[int]:
    """Assign row i (story sentence) to column j_i (window) maximising sum(sim[i, j_i])
    subject to j_i >= j_{i-1} + min_gap. Runs in O(rows * cols) using a running prefix maximum."""
    k, w = sim.shape
    if k > 1 and (k - 1) * min_gap > w - 1:  # too many sentences for the gap: shrink it
        min_gap = max(0, (w - 1) // (k - 1))
    dp = np.full((k, w), -np.inf)
    back = np.zeros((k, w), dtype=int)
    dp[0] = sim[0]
    for i in range(1, k):
        best_val, best_idx = -np.inf, -1
        for j in range(w):
            # Best predecessor must end at least min_gap columns earlier.
            p = j - min_gap
            if p >= 0 and dp[i - 1, p] > best_val:
                best_val, best_idx = dp[i - 1, p], p
            if best_idx >= 0:
                dp[i, j] = sim[i, j] + best_val
                back[i, j] = best_idx
    path = [int(np.argmax(dp[-1]))]
    for i in range(k - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def segment_modalities(segments: list[dict], centrality: list[float], feats: dict) -> dict[str, np.ndarray]:
    """Story-independent per-segment signals, each scaled to [0, 1] over the whole film."""
    audio = feats["audio"]["arousal"]
    visual = feats["visual"]["activity"]
    return {
        "centrality": minmax(np.array(centrality)),
        "text_emotion": np.array([e["intensity"] for e in feats["emotion"]]),
        "audio": minmax(np.array([segment_signal(audio, s["start"], s["end"], np.mean) for s in segments])),
        "visual": minmax(np.array([segment_signal(visual, s["start"], s["end"], np.max) for s in segments])),
    }


def find_peaks(cfg: dict, segments: list[dict], summary: dict, feats: dict,
               weights: dict | None = None, cache: bool = True) -> dict:
    """Return the peak point for each story sentence plus the alignment diagnostics."""
    out = Path(cfg["output_dir"]) / "peaks.json"
    if cache and weights is None and out.exists():
        log.info("Using cached peaks: %s", out.name)
        return load_json(out)

    pcfg = cfg["peaks"]
    weights = dict(weights or pcfg["weights"])
    total_w = sum(weights.values())
    weights = {m: weights.get(m, 0.0) / total_w for m in MODALITIES}

    sentences = summary["story_sentences"]
    windows = build_windows(segments, pcfg["window_seconds"], pcfg["window_hop_seconds"])
    model = pcfg["embedding_model"]
    sent_emb = embed(sentences, model)
    win_emb = embed([w["text"] for w in windows], model)
    # Segment text with one neighbour on each side gives short lines enough context to embed well.
    ctx_text = [" ".join(segments[j]["text"] for j in range(max(0, i - 1), min(len(segments), i + 2)))
                for i in range(len(segments))]
    ctx_emb = embed(ctx_text, model)
    own_emb = embed([s["text"] for s in segments], model)

    win_sim = sent_emb @ win_emb.T
    # Coarse time index: a sentence written from chunk k may only align inside chunk k's time span.
    spans = None
    if summary.get("sentence_chunks") and summary.get("chunks"):
        margin = pcfg["window_seconds"]
        spans = [(summary["chunks"][k]["start"] - margin, summary["chunks"][k]["end"] + margin)
                 for k in summary["sentence_chunks"]]
        centres = np.array([(w["start"] + w["end"]) / 2 for w in windows])
        for i, (a, b) in enumerate(spans):
            win_sim[i, (centres < a) | (centres > b)] -= 10.0
    path = monotonic_alignment(win_sim, min_gap=max(1, int(pcfg["window_seconds"] / pcfg["window_hop_seconds"]) // 2))
    # The segment's own wording dominates; neighbour context helps very short lines.
    seg_sim = 0.6 * (sent_emb @ own_emb.T) + 0.4 * (sent_emb @ ctx_emb.T)
    modal = segment_modalities(segments, summary["centrality"], feats)

    mids = np.array([(s["start"] + s["end"]) / 2 for s in segments])
    peaks, last_end = [], -1.0
    for i, (sent, j) in enumerate(zip(sentences, path)):
        w = windows[j]
        centre = (w["start"] + w["end"]) / 2
        ok = (np.abs(mids - centre) <= pcfg["search_radius"]) & (mids > last_end)
        if spans:
            ok &= (mids >= spans[i][0]) & (mids <= spans[i][1])
        cand = np.where(ok)[0]
        if cand.size == 0:  # radius exhausted by the previous peak: look further ahead
            cand = np.where(mids > last_end)[0][:20]
        if cand.size == 0:
            cand = np.arange(len(segments))
        semantic = minmax(seg_sim[i])  # normalised over the whole film for this sentence
        parts = {"semantic": semantic[cand], **{m: modal[m][cand] for m in MODALITIES if m != "semantic"}}
        fused = sum(weights[m] * parts[m] for m in MODALITIES)
        best = int(np.argmax(fused))
        sid = int(cand[best])
        seg = segments[sid]
        last_end = seg["end"]
        peaks.append({
            "index": i + 1,
            "story_sentence": sent,
            "aligned_window": [round(w["start"], 1), round(w["end"], 1)],
            "window_similarity": round(float(win_sim[i, j]), 4),
            "peak_segment_id": sid,
            "timestamp": seg["start"],
            "timestamp_hms": fmt_ts(seg["start"]),
            "segment_end": seg["end"],
            "transcript": seg["text"],
            "emotion_label": feats["emotion"][sid]["label"],
            "scores": {**{m: round(float(parts[m][best]), 4) for m in MODALITIES}, "fused": round(float(fused[best]), 4)},
            "candidates": int(cand.size),
        })
        log.info("Peak %2d @ %s  fused=%.2f  | %s", i + 1, fmt_ts(seg["start"]), fused[best], sent[:70])

    # Film-wide salience curve (per second) without the story term, for the timeline figure.
    n_sec = int(np.ceil(segments[-1]["end"])) + 1
    salience = np.zeros(n_sec)
    for s_idx, s in enumerate(segments):
        val = sum(weights[m] * modal[m][s_idx] for m in MODALITIES if m != "semantic") / max(1e-9, 1 - weights["semantic"])
        a, b = int(s["start"]), min(n_sec, int(np.ceil(s["end"])) + 1)
        salience[a:b] = np.maximum(salience[a:b], val)

    result = {"weights": weights, "peaks": peaks, "alignment_path": path,
              "windows": [{k: w[k] for k in ("start", "end")} for w in windows],
              "salience_curve": salience.round(4).tolist()}
    if cache and out.parent.exists():
        save_json(result, out)
    return result


def write_peaks_csv(peaks: list[dict], path: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["peak", "timestamp", "seconds", "story_sentence", "peak_transcript", "emotion",
                     *MODALITIES, "fused"])
        for p in peaks:
            wr.writerow([p["index"], p["timestamp_hms"], p["timestamp"], p["story_sentence"], p["transcript"],
                         p["emotion_label"], *[p["scores"][m] for m in MODALITIES], p["scores"]["fused"]])
