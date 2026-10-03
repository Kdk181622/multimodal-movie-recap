"""Stage 8: turn peak timestamps into video segments (clip windows) that fit the recap budget.

Each peak segment is grown with neighbouring dialogue that best matches its story sentence,
padded, snapped to nearby shot boundaries so clips start and end on cuts, and kept
chronological and non-overlapping. The total is held under the configured duration
(< 600 s, as required by the assignment).
"""

from __future__ import annotations

import numpy as np

from .summarize import embed
from .utils import fmt_ts, log


def _snap(t: float, cuts: np.ndarray, before: float, after: float) -> float:
    """Move t to the nearest shot boundary in [t - before, t + after], if there is one."""
    if cuts.size == 0:
        return t
    near = cuts[(cuts >= t - before) & (cuts <= t + after)]
    return float(near[np.argmin(np.abs(near - t))]) if near.size else t


def plan_clips(cfg: dict, segments: list[dict], peaks: list[dict], shot_boundaries: list[float],
               film_duration: float, overhead_seconds: float) -> list[dict]:
    """Build one clip per peak.

    overhead_seconds: time already spent on title / chapter cards, subtracted from the budget.
    """
    rc = cfg["recap"]
    budget = rc["max_duration_seconds"] - overhead_seconds
    per_clip = float(np.clip(budget / max(1, len(peaks)), rc["clip_min_seconds"], rc["clip_max_seconds"]))
    cuts = np.array(sorted(shot_boundaries))
    sent_emb = embed([p["story_sentence"] for p in peaks], cfg["peaks"]["embedding_model"])
    seg_emb = embed([s["text"] for s in segments], cfg["peaks"]["embedding_model"])

    clips = []
    for k, p in enumerate(peaks):
        lo = hi = p["peak_segment_id"]
        sim = seg_emb @ sent_emb[k]
        # Grow the clip one neighbouring segment at a time, preferring the side that is more on-topic.
        while True:
            dur = segments[hi]["end"] - segments[lo]["start"]
            options = []
            if lo > 0 and segments[lo]["start"] - segments[lo - 1]["end"] < 4.0:
                options.append(("lo", sim[lo - 1], segments[hi]["end"] - segments[lo - 1]["start"]))
            if hi < len(segments) - 1 and segments[hi + 1]["start"] - segments[hi]["end"] < 4.0:
                # Small bonus for moving forward: a scene usually resolves after its key line.
                options.append(("hi", sim[hi + 1] + 0.05, segments[hi + 1]["end"] - segments[lo]["start"]))
            options = [o for o in options if o[2] <= per_clip]
            if not options or dur >= per_clip:
                break
            side = max(options, key=lambda o: o[1])[0]
            lo, hi = (lo - 1, hi) if side == "lo" else (lo, hi + 1)
        start = segments[lo]["start"] - rc["clip_pad_seconds"]
        end = segments[hi]["end"] + rc["clip_pad_seconds"]
        # Too short (isolated line)? Extend evenly with surrounding picture.
        if end - start < rc["clip_min_seconds"]:
            extra = (rc["clip_min_seconds"] - (end - start)) / 2
            start, end = start - extra, end + extra
        start = _snap(start, cuts, before=1.5, after=0.4)
        end = _snap(end, cuts, before=0.4, after=1.5)
        clips.append({"peak": p["index"], "start": max(0.0, start), "end": min(film_duration, end),
                      "segment_ids": list(range(lo, hi + 1))})

    # Resolve overlaps between consecutive clips (they stay chronological because peaks are).
    for a, b in zip(clips, clips[1:]):
        if b["start"] < a["end"]:
            mid = (a["end"] + b["start"]) / 2
            a["end"], b["start"] = mid, mid

    # Final guard: shrink every clip proportionally if the total is still over budget.
    total = sum(c["end"] - c["start"] for c in clips)
    if total > budget:
        scale = budget / total
        for c in clips:
            centre, half = (c["start"] + c["end"]) / 2, (c["end"] - c["start"]) * scale / 2
            c["start"], c["end"] = centre - half, centre + half
    for c in clips:
        c["start"], c["end"] = round(c["start"], 2), round(c["end"], 2)
        c["duration"] = round(c["end"] - c["start"], 2)
        c["start_hms"], c["end_hms"] = fmt_ts(c["start"]), fmt_ts(c["end"])
        c["transcript"] = " ".join(segments[i]["text"] for i in c["segment_ids"]
                                   if segments[i]["end"] > c["start"] and segments[i]["start"] < c["end"])
    log.info("Planned %d clips, %.1f s of footage (budget %.1f s, target %.1f s per clip)",
             len(clips), sum(c["duration"] for c in clips), budget, per_clip)
    return clips
