"""Stage 12: figures and a Markdown report tying summary, peaks, timestamps and clips together."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import fmt_ts  # noqa: E402

# Palette: validated categorical slots (light mode) + neutral ink, see the dataviz reference palette.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _smooth(x, k: int = 9) -> np.ndarray:
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="same") if len(x) > k else x


def timeline_figure(segments, feats, peaks, clips, path) -> None:
    """Small multiples over the film's running time: one panel per modality + fused salience + clips."""
    audio = _smooth(feats["audio"]["arousal"])
    visual = _smooth(feats["visual"]["activity"])
    # Salience is defined only where someone speaks; carry it across short pauses, then smooth.
    sal = np.asarray(peaks["salience_curve"], float)
    for i in range(1, len(sal)):
        if sal[i] == 0 and sal[i - 1] > 0:
            sal[i] = sal[i - 1] * 0.97
    sal = _smooth(sal, 15)
    t_a, t_v, t_s = np.arange(len(audio)) / 60, np.arange(len(visual)) / 60, np.arange(len(sal)) / 60
    seg_t = np.array([(s["start"] + s["end"]) / 2 for s in segments]) / 60
    emo = np.array([e["intensity"] for e in feats["emotion"]])

    fig, axes = plt.subplots(5, 1, figsize=(13, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.3, 0.55]})
    axes[0].plot(t_a, audio, color=BLUE, lw=1.2)
    axes[0].set_title("Audio arousal (librosa: loudness, onset strength, amplitude fluctuation)", loc="left")
    axes[1].plot(t_v, visual, color=ORANGE, lw=1.2)
    for c in feats["visual"]["shot_boundaries"]:
        axes[1].axvline(c / 60, color=MUTED, lw=0.3, alpha=0.5, zorder=0)
    axes[1].set_title("Visual activity (OpenCV: frame-difference motion + shot-cut density; grey ticks = detected cuts)", loc="left")
    axes[2].scatter(seg_t, emo, s=9, color=AQUA, alpha=0.7, linewidths=0)
    axes[2].set_title("Text emotion intensity per transcript segment (DistilRoBERTa, 1 - P(neutral))", loc="left")
    axes[3].plot(t_s, sal, color=VIOLET, lw=0.9, label="fused salience (story-independent signals)")
    last_x, level = -99.0, 0
    for p in peaks["peaks"]:
        x = p["timestamp"] / 60
        axes[3].axvline(x, color=INK, lw=0.8, ls="--")
        # Stagger labels vertically when neighbouring peaks are closer than ~1.2 minutes.
        level = (level + 1) % 3 if x - last_x < 1.2 else 0
        last_x = x
        axes[3].text(x, 1.03 + 0.09 * level, str(p["index"]), ha="center", va="bottom", fontsize=8, color=INK,
                     transform=axes[3].get_xaxis_transform())
    axes[3].set_title("Fused multimodal salience (15 s smoothed) with the selected peak points (dashed, numbered)",
                      loc="left", pad=30)
    axes[3].legend(loc="upper right")
    for c in clips:
        axes[4].barh(0, (c["end"] - c["start"]) / 60, left=c["start"] / 60, height=0.6, color=BLUE)
    axes[4].set_yticks([])
    axes[4].set_title("Extracted video segments used in the recap", loc="left")
    axes[4].set_xlabel("Film running time (minutes)")
    for ax in axes[:4]:
        ax.set_ylabel("score (0-1)")
        ax.set_ylim(0, 1.05)
    axes[4].set_xlim(0, max(t_a[-1], t_v[-1]))
    fig.suptitle("Where the recap comes from: multimodal signals across the full film", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics_figure(evaluation, path) -> None:
    """One small panel per metric; the proposed summary is blue, baselines are grey."""
    ev = evaluation["summary_vs_reference"]
    names = list(ev.keys())
    metrics = [("bleu4", "BLEU-4"), ("bleu1", "BLEU-1"), ("rougeL", "ROUGE-L F1"), ("semantic_similarity", "Semantic similarity")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (key, label) in zip(axes, metrics):
        vals = [ev[n][key] for n in names]
        colors = [BLUE if "ours" in n else MUTED for n in names]
        ax.barh(y, vals, color=colors, height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v, yy, f" {v:.3f}", va="center", fontsize=8, color=INK2)
        ax.set_title(label, loc="left")
        ax.set_xlim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    fig.suptitle("Summary vs. human reference plot (Wikipedia): proposed system vs. baselines", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fidelity_figure(evaluation, path) -> None:
    rows = evaluation.get("recap_fidelity", {}).get("clips", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["bleu"] for r in rows], width=0.38, color=BLUE, label="clip BLEU (recap ASR vs. original transcript)")
    ax.bar(x + 0.2, [r["peak_line_recall"] for r in rows], width=0.38, color=ORANGE, label="peak-line word recall")
    ax.axhline(0.7, color=INK2, lw=0.8, ls="--")
    ax.text(len(rows) - 0.5, 0.71, "0.7 reference threshold", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{r['peak']}" for r in rows])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (0-1)")
    ax.set_xlabel("Peak / clip")
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("Recap fidelity: does each clip contain the dialogue at its mapped timestamp?", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def architecture_figure(path) -> None:
    """Pipeline diagram used in the README and presentation."""
    stages = [
        ("YouTube URL", "yt-dlp", MUTED), ("Movie (MP4)", "OpenCV probe", MUTED),
        ("Audio + Video", "ffmpeg demux / OpenCV frames", BLUE), ("Transcript", "faster-whisper small.en\nword timestamps", BLUE),
        ("Summary Story", "Qwen2.5-3B rolling context\n(TextRank fallback)", BLUE), ("Peak Points", "5-signal post-fusion\nDP story alignment", ORANGE),
        ("Time Index", "Whisper timestamps\n+ shot snapping", ORANGE), ("Segment Extraction", "OpenCV seek / read / annotate", ORANGE),
        ("Movie Recap", "< 10 min MP4 + EDL + SRT", AQUA), ("Evaluation", "BLEU / ROUGE / fidelity", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    # Snake layout: row 1 runs left to right, row 2 runs right to left, so every arrow is short.
    pos = [(i + 0.5, 1.5) if i < 5 else (9 - i + 0.5, 0.42) for i in range(len(stages))]
    for (x, y), (name, tool, col) in zip(pos, stages):
        ax.add_patch(plt.Rectangle((x - 0.43, y - 0.3), 0.86, 0.6, facecolor="white", edgecolor=col, lw=2))
        ax.text(x, y + 0.1, name, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, y - 0.13, tool, ha="center", va="center", fontsize=7.5, color=INK2)
    arrow = dict(arrowstyle="-|>", color=INK2, lw=1.2)
    for i in range(len(stages) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if y0 == y1:
            sign = 1 if x1 > x0 else -1
            ax.annotate("", xy=(x1 - 0.44 * sign, y1), xytext=(x0 + 0.44 * sign, y0), arrowprops=arrow)
        else:
            ax.annotate("", xy=(x1, y1 + 0.31), xytext=(x0, y0 - 0.31), arrowprops=arrow)
    # Side signals feed the peak fusion stage.
    ax.text(2.4, 0.96, "side signals: librosa audio arousal  |  OpenCV motion + shot cuts  |  text emotion",
            ha="center", va="center", fontsize=8.5, color=ORANGE)
    ax.annotate("", xy=(4.06, 0.66), xytext=(3.32, 0.96), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
    ax.set_title("Multimodal movie-recap pipeline", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(cfg, video_info, transcript, summary, peaks, clips, render_info, evaluation, path) -> None:
    edl_clip = {e["peak"]: e for e in render_info["edl"] if e["type"] == "clip"}
    ev, cons = evaluation["summary_vs_reference"], evaluation["constraints"]
    fid = evaluation.get("recap_fidelity", {})
    abl = evaluation["ablation"]
    info = transcript["info"]
    lines = [
        f"# Movie Recap Report - {cfg['input']['title']}", "",
        f"Source: {cfg['input']['url']}  ",
        f"Film length: {fmt_ts(video_info['duration_seconds'])} ({video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} fps)  ",
        f"Recap length: **{fmt_ts(cons['recap_duration_seconds'])}** ({cons['recap_duration_seconds']:.1f} s, "
        f"under 10 minutes: **{cons['under_10_minutes']}**, compression {cons['compression_ratio']}x)", "",
        "![pipeline](../../docs/architecture.png)", "",
        "## 1. Transcript", "",
        f"faster-whisper `{info['model']}` produced {len(transcript['segments'])} timestamped segments "
        f"({sum(len(s['text'].split()) for s in transcript['segments'])} words) in {info['processing_seconds']} s on CPU. "
        "Full text: `transcript.srt` / `transcript.json`.", "",
        "## 2. Summary story", "",
        f"Generated by `{summary['backend']}` from {len(summary['chunks'])} transcript chunks "
        "(rolling context, 1-2 sentences per chunk).", "",
        f"> {summary['story']}", "",
        "## 3. Summary -> peak points -> time index -> video segments", "",
        _md_table(["#", "Story sentence", "Peak time", "Peak line (transcript)", "Clip in film", "Clip in recap", "Emotion", "Fused"],
                  [[p["index"], p["story_sentence"], p["timestamp_hms"], p["transcript"][:90],
                    f"{c['start_hms']}-{c['end_hms']}",
                    f"{fmt_ts(edl_clip[p['index']]['recap_start'])}-{fmt_ts(edl_clip[p['index']]['recap_end'])}",
                    p["emotion_label"], p["scores"]["fused"]] for p, c in zip(peaks["peaks"], clips)]), "",
        f"Fusion weights: " + ", ".join(f"{k} {v:.2f}" for k, v in peaks["weights"].items()), "",
        "![timeline](timeline.png)", "",
        "## 4. Evaluation", "",
        "### 4.1 Summary vs. human reference (Wikipedia plot)", "",
        _md_table(["System", "BLEU-4", "BLEU-4 (NLTK)", "sacreBLEU", "BLEU-1", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Semantic sim.", "Plot coverage", "Words"],
                  [[n, m["bleu4"], m["bleu4_nltk"], m["sacrebleu"], m["bleu1"], m["rouge1"], m["rouge2"], m["rougeL"],
                    m["semantic_similarity"], m["plot_coverage@0.5"], m["words"]] for n, m in ev.items()]), "",
        "![metrics](metrics.png)", "",
    ]
    if fid:
        lines += [
            "### 4.2 Recap fidelity (recap re-transcribed with Whisper)", "",
            f"- Mean clip BLEU (recap speech vs. original transcript at the mapped timestamps): **{fid['mean_clip_bleu']}** (median {fid['median_clip_bleu']})",
            f"- Peak-line hit rate (>= 60% of the peak line's words heard in its clip): **{fid['peak_hit_rate@0.6']}**",
            f"- Whole-recap BLEU (speech + narration vs. expected script): **{fid['overall_bleu']}**",
            f"- Narration intelligibility (BLEU of ASR on chapter cards vs. story sentences): {fid['mean_narration_bleu']}", "",
            "![fidelity](fidelity.png)", "",
        ]
    lines += [
        "### 4.3 Ablation: text-only vs. multimodal peak selection", "",
        _md_table(["Selection", *abl["multimodal_profile"].keys()],
                  [["Text only (semantic + centrality)", *abl["text_only_profile"].values()],
                   ["Multimodal fusion (proposed)", *abl["multimodal_profile"].values()]]), "",
        f"{abl['peaks_changed']} of {cons['n_peaks']} peaks moved when audio, visual and emotion signals were added "
        f"(mean shift {abl['mean_shift_seconds']} s).", "",
        "## 5. Output files", "",
        "- `movie_recap.mp4` - final recap; `movie_recap.srt` - its subtitles; `recap_edl.csv` - edit decision list",
        "- `transcript.json/.srt`, `summary.json`, `peaks.json/.csv`, `clips.json`, `evaluation.json`",
        "- `features_audio.json`, `features_visual.json`, `features_text_emotion.json`",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
