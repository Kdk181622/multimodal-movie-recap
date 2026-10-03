"""End-to-end orchestration:

YouTube video -> movie -> audio + video -> transcript -> summary story -> peak points
-> time index -> video segment extraction -> movie recap -> BLEU evaluation

Expensive stages (transcription, features, LLM summary, peaks) cache their results as JSON in the
output folder, so later stages can be re-run and tuned without repeating the slow ones.
Delete a JSON file (or pass --fresh) to recompute that stage.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import features, ingest, peaks as peaks_mod, render, report, segments, summarize, transcribe
from .evaluate import evaluate
from .utils import PROJECT_ROOT, fmt_ts, load_config, log, save_json

CACHE_FILES = ["transcript.json", "features_audio.json", "features_visual.json", "features_text_emotion.json",
               "summary.json", "peaks.json"]


def run(config_path: str | None = None, url: str | None = None, fresh: list[str] | None = None,
        skip_eval: bool = False) -> dict:
    cfg = load_config(config_path or PROJECT_ROOT / "config.yaml")
    if url:
        cfg["input"]["url"] = url
    out = Path(cfg["output_dir"])
    for name in fresh or []:
        (out / name).unlink(missing_ok=True)
    timings: dict[str, float] = {}

    def stage(name):
        timings[name] = time.time()
        log.info("=== %s ===", name)

    def done(name):
        timings[name] = round(time.time() - timings[name], 1)

    # 1-2. YouTube video -> movie -> audio + video
    stage("1. Ingest & A/V extraction")
    video_path = ingest.download_youtube(cfg["input"]["url"], cfg["input"]["local_path"])
    video_info = ingest.probe_video(video_path)
    save_json(video_info, out / "video_info.json")
    audio = ingest.extract_audio(video_path, str(out / "audio"))
    done("1. Ingest & A/V extraction")

    # 3. Audio -> transcript (with time index)
    stage("2. Speech-to-text")
    tr = transcribe.transcribe(audio["wav_16k"], cfg["asr"], str(out / "transcript.json"))
    transcribe.write_srt(tr["segments"], str(out / "transcript.srt"))
    segs = tr["segments"]
    done("2. Speech-to-text")

    # 4. Multimodal signals
    stage("3. Multimodal features")
    feats = features.compute_all(cfg, segs, audio["wav_16k"])
    done("3. Multimodal features")

    # 5. Transcript -> summary story
    stage("4. Summarisation")
    summary = summarize.summarize(cfg, segs)
    (out / "summary_story.txt").write_text(summary["story"] + "\n", encoding="utf-8")
    done("4. Summarisation")

    # 6-7. Summary -> peak points -> time index
    stage("5. Peak points & time index")
    pk = peaks_mod.find_peaks(cfg, segs, summary, feats)
    peaks_mod.write_peaks_csv(pk["peaks"], str(out / "peaks.csv"))
    done("5. Peak points & time index")

    # 8. Time index -> video segments (budgeted to stay under 10 minutes)
    stage("6. Segment planning")
    rc = cfg["recap"]
    title = cfg["input"]["title"]
    narr_texts = [p["story_sentence"] for p in pk["peaks"]]
    intro_text, outro_text = f"An AI-generated recap of {title}.", "That was the AI recap of the film."
    voices = render.synthesize_narration(narr_texts + [intro_text, outro_text], out / "narration", rc["narration"])
    card_voices, intro_v, outro_v = voices[:-2], voices[-2], voices[-1]
    overhead = max(4.0, render.card_duration(intro_v, intro_text, 4.0)) + max(3.5, render.card_duration(outro_v, outro_text, 3.5))
    if rc["title_cards"]:
        overhead += sum(render.card_duration(v, t, rc["card_min_seconds"]) for v, t in zip(card_voices, narr_texts))
    clips = segments.plan_clips(cfg, segs, pk["peaks"], feats["visual"]["shot_boundaries"],
                                video_info["duration_seconds"], overhead + 1.0)
    save_json(clips, out / "clips.json")
    done("6. Segment planning")

    # 9-10. OpenCV extraction + assembly -> movie recap
    stage("7. Recap rendering (OpenCV)")
    render_info = render.render_recap(cfg, clips, pk["peaks"], card_voices, intro_v, outro_v,
                                      video_info["duration_seconds"], title)
    save_json({k: v for k, v in render_info.items() if k != "edl"}, out / "render_info.json")
    done("7. Recap rendering (OpenCV)")
    if render_info["duration_seconds"] >= 600:
        raise RuntimeError(f"Recap is {render_info['duration_seconds']} s - over the 10 minute limit")

    result = {"cfg": cfg, "video_info": video_info, "transcript": tr, "features": feats, "summary": summary,
              "peaks": pk, "clips": clips, "render": render_info}
    if skip_eval:
        return result

    # 11. Evaluation (BLEU etc.) and report
    stage("8. Evaluation")
    ev = evaluate(cfg, segs, summary, feats, pk, render_info, video_info["duration_seconds"])
    done("8. Evaluation")
    report.timeline_figure(segs, feats, pk, clips, out / "timeline.png")
    report.metrics_figure(ev, out / "metrics.png")
    report.fidelity_figure(ev, out / "fidelity.png")
    report.architecture_figure(PROJECT_ROOT / "docs" / "architecture.png")
    report.write_report(cfg, video_info, tr, summary, pk, clips, render_info, ev, out / "REPORT.md")
    save_json(timings, out / "stage_timings.json")
    log.info("Done. Recap: %s (%s). Report: %s", render_info["path"], fmt_ts(render_info["duration_seconds"]),
             out / "REPORT.md")
    result["evaluation"] = ev
    result["timings"] = timings
    return result
