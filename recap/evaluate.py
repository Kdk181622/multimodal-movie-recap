"""Stage 11: evaluation.

A. Summary quality - the summary story is compared with a human-written reference plot
   (Wikipedia) using BLEU (own implementation, NLTK and sacreBLEU), ROUGE and embedding
   similarity, next to four baselines of equal length: extractive TextRank, lead-N,
   random segments and uniformly spaced segments.
B. Recap fidelity - the finished recap is transcribed again with Whisper. Each clip's
   speech is compared (BLEU) with the original transcript at the mapped timestamps, which
   verifies the summary -> peak -> time index -> segment chain end to end.
C. Constraints - recap duration < 600 s and compression ratio.
D. Ablation - text-only vs. multimodal peak selection.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from .bleu import bleu, rouge, tokenize
from .peaks import MODALITIES, find_peaks
from .summarize import embed, split_sentences
from .transcribe import transcribe
from .utils import load_json, log, save_json


def _take_words(segments: list[dict], ids: list[int], n_words: int) -> str:
    words: list[str] = []
    for i in ids:
        words += segments[i]["text"].split()
        if len(words) >= n_words:
            break
    return " ".join(words[:n_words])


def baseline_summaries(segments: list[dict], summary: dict, n_words: int, seed: int) -> dict[str, str]:
    long_ids = [s["id"] for s in segments if len(s["text"].split()) >= 6]
    rng = random.Random(seed)
    k = max(1, n_words // 12)
    uniform_ids = [long_ids[int(i)] for i in np.linspace(0, len(long_ids) - 1, k)]
    return {
        "Extractive TextRank": summary["extractive_baseline"]["text"],
        "Lead-N (first words)": _take_words(segments, [s["id"] for s in segments], n_words),
        "Uniform segments": _take_words(segments, uniform_ids, n_words),
        "Random segments": _take_words(segments, sorted(rng.sample(long_ids, min(len(long_ids), k * 2))), n_words),
    }


def text_metrics(candidate: str, reference: str, model_name: str) -> dict:
    """BLEU (three implementations), ROUGE, semantic similarity and plot coverage."""
    import sacrebleu
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    own = bleu(candidate, [reference])
    nltk_score = corpus_bleu([[tokenize(reference)]], [tokenize(candidate)],
                             smoothing_function=SmoothingFunction().method2)
    sacre = sacrebleu.corpus_bleu([candidate], [[reference]]).score / 100
    ref_sents, cand_sents = split_sentences(reference), split_sentences(candidate) or [candidate]
    r_emb, c_emb = embed(ref_sents, model_name), embed(cand_sents, model_name)
    doc_sim = float(np.dot(r_emb.mean(0), c_emb.mean(0)) / (np.linalg.norm(r_emb.mean(0)) * np.linalg.norm(c_emb.mean(0))))
    best = (r_emb @ c_emb.T).max(axis=1)
    return {
        "bleu4": round(own["bleu"], 4),
        "bleu1": round(own["precisions"][0] * own["bp"], 4),
        "bleu4_nltk": round(nltk_score, 4),
        "sacrebleu": round(sacre, 4),
        "precisions": [round(p, 4) for p in own["precisions"]],
        "brevity_penalty": round(own["bp"], 4),
        **{k: round(v, 4) for k, v in rouge(candidate, reference).items()},
        "semantic_similarity": round(doc_sim, 4),
        "plot_coverage@0.5": round(float((best >= 0.5).mean()), 4),
        "words": len(candidate.split()),
    }


def recap_fidelity(cfg: dict, render_info: dict, peaks: list[dict]) -> dict:
    """Re-transcribe the recap and check that each clip carries the dialogue of its mapped timestamps.

    Every EDL entry (chapter card or clip) is cut out of the finished recap's soundtrack and
    transcribed on its own, so one entry's audio cannot leak into another's text.
    """
    import soundfile as sf

    from .utils import run_ffmpeg

    out = Path(cfg["output_dir"])
    asr_cfg = dict(cfg["asr"], vad_filter=False)
    wav = out / "audio" / "recap_16k.wav"
    run_ffmpeg(["-i", render_info["path"], "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    recap_audio, sr = sf.read(str(wav), dtype="float32")
    piece = out / "audio" / "_piece.wav"
    clip_rows, narr_rows, recap_asr = [], [], []
    for e in render_info["edl"]:
        sf.write(str(piece), recap_audio[int(e["recap_start"] * sr):int(e["recap_end"] * sr)], sr)
        segs = transcribe(str(piece), asr_cfg)["segments"]
        recap_asr += [dict(s, start=s["start"] + e["recap_start"], end=s["end"] + e["recap_start"], words=[]) for s in segs]
        heard = " ".join(s["text"] for s in segs)
        if e["type"] == "clip" and e["text"].strip():
            b = bleu(heard, [e["text"]])["bleu"]
            peak = next(p for p in peaks if p["index"] == e["peak"])
            peak_words = {t for t in tokenize(peak["transcript"]) if t.isalnum()}
            heard_words = set(tokenize(heard))
            recall = len(peak_words & heard_words) / max(1, len(peak_words))
            clip_rows.append({"peak": e["peak"], "bleu": round(b, 4), "peak_line_recall": round(recall, 4),
                              "expected": e["text"], "heard": heard})
        elif e["type"] == "chapter":
            narr_rows.append(bleu(heard, [e["text"]])["bleu"])
    piece.unlink(missing_ok=True)
    save_json({"segments": recap_asr}, out / "recap_transcript.json")
    expected_all = " ".join(e["text"] for e in render_info["edl"] if e["text"])
    heard_all = " ".join(s["text"] for s in recap_asr)
    return {
        "overall_bleu": round(bleu(heard_all, [expected_all])["bleu"], 4),
        "mean_clip_bleu": round(float(np.mean([r["bleu"] for r in clip_rows])), 4) if clip_rows else None,
        "median_clip_bleu": round(float(np.median([r["bleu"] for r in clip_rows])), 4) if clip_rows else None,
        "peak_hit_rate@0.6": round(float(np.mean([r["peak_line_recall"] >= 0.6 for r in clip_rows])), 4) if clip_rows else None,
        "mean_narration_bleu": round(float(np.mean(narr_rows)), 4) if narr_rows else None,
        "clips": clip_rows,
    }


def ablation(cfg: dict, segments: list[dict], summary: dict, feats: dict, peaks_mm: list[dict]) -> dict:
    """Compare peak picks from text-only weights with the multimodal fusion weights."""
    text_only = find_peaks(cfg, segments, summary, feats,
                           weights={"semantic": 0.75, "centrality": 0.25}, cache=False)["peaks"]
    def profile(ps):
        return {m: round(float(np.mean([p["scores"][m] for p in ps])), 4) for m in MODALITIES}
    shifts = [abs(a["timestamp"] - b["timestamp"]) for a, b in zip(text_only, peaks_mm)]
    return {
        "text_only_profile": profile(text_only),
        "multimodal_profile": profile(peaks_mm),
        "peaks_changed": int(sum(s > 1.0 for s in shifts)),
        "mean_shift_seconds": round(float(np.mean(shifts)), 2),
        "text_only_timestamps": [p["timestamp_hms"] for p in text_only],
    }


def evaluate(cfg: dict, segments: list[dict], summary: dict, feats: dict, peaks: dict,
             render_info: dict, film_duration: float) -> dict:
    ref_path = cfg["input"]["reference_summary"]
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference summary missing: {ref_path}")
    reference = Path(ref_path).read_text(encoding="utf-8")
    model = cfg["peaks"]["embedding_model"]
    n_words = len(summary["story"].split())

    systems = {"Summary story (ours)": summary["story"],
               **baseline_summaries(segments, summary, n_words, cfg["evaluation"]["seed"])}
    summary_eval = {name: text_metrics(text, reference, model) for name, text in systems.items()}
    for name, m in summary_eval.items():
        log.info("%-24s BLEU-4 %.4f | BLEU-1 %.4f | ROUGE-L %.4f | sem %.3f", name, m["bleu4"], m["bleu1"], m["rougeL"], m["semantic_similarity"])

    result = {
        "reference": {"source": "Wikipedia - Detour (1945 film), Plot section", "words": len(reference.split())},
        "summary_vs_reference": summary_eval,
        "constraints": {
            "film_duration_seconds": round(film_duration, 2),
            "recap_duration_seconds": render_info["duration_seconds"],
            "under_10_minutes": render_info["duration_seconds"] < 600,
            "compression_ratio": round(film_duration / render_info["duration_seconds"], 2),
            "footage_seconds": round(sum(e["recap_end"] - e["recap_start"] for e in render_info["edl"] if e["type"] == "clip"), 2),
            "n_peaks": len(peaks["peaks"]),
        },
        "ablation": ablation(cfg, segments, summary, feats, peaks["peaks"]),
    }
    if cfg["evaluation"]["retranscribe_recap"]:
        log.info("Re-transcribing the recap for the fidelity check...")
        result["recap_fidelity"] = recap_fidelity(cfg, render_info, peaks["peaks"])
        log.info("Fidelity: mean clip BLEU %.3f, peak hit rate %.2f",
                 result["recap_fidelity"]["mean_clip_bleu"], result["recap_fidelity"]["peak_hit_rate@0.6"])
    save_json(result, Path(cfg["output_dir"]) / "evaluation.json")
    return result


def load_evaluation(cfg: dict) -> dict:
    return load_json(Path(cfg["output_dir"]) / "evaluation.json")
