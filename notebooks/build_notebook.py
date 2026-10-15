"""Builds Movie_Recap_Walkthrough.ipynb (run from the project root, then execute with nbconvert).

Every pipeline component follows the same sequence:
Objective -> Purpose -> Methodology -> [Prompt, only for an instructable AI model, placed directly before
the code cell that calls it] -> Implementation -> Coding -> Output -> Inference -> Insights -> Summary
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


def head(num: str, title: str, objective: str, purpose: str, methodology: str) -> None:
    md(f"## {num}. {title}\n\n### Objective\n{objective.strip()}\n\n### Purpose\n{purpose.strip()}\n\n### Methodology\n{methodology.strip()}")


def implementation(text: str) -> None:
    md("### Implementation\n" + text.strip())


def output(text: str = "") -> None:
    md("### Output\n" + (text.strip() or "The results of the code above are displayed below."))


def chart_note(observation: str, interpretation: str, key_insight: str) -> None:
    """Reading of the chart(s) just displayed, inside the Output step (Notebook Standards v2)."""
    md(f"**Reading the chart**\n\n* **Observation:** {observation.strip()}\n* **Interpretation:** {interpretation.strip()}\n"
       f"* **Key insight:** {key_insight.strip()}")


def close(inference: str, insights: str, summary: str) -> None:
    md(f"### Inference\n{inference.strip()}\n\n### Insights\n{insights.strip()}\n\n### Summary\n{summary.strip()}")


# ============================================================================ title
md("""
# Multimodal Movie Recap: YouTube Film → Summary Story → Peak Points → Recap Under 10 Minutes

**AI11 · Module 8 · Multimodal AI Systems: Speech, Text, and Conversational Agents · Assignment 1**
**Student:** Karthikeyan K  ·  **Faculty:** Dr. J. B. Simha

**Problem statement.** A full-length film takes more than an hour to watch. The task is to build a multimodal
AI system that takes a YouTube movie, processes its video, audio and text, writes a summary story, finds the
important (peak) moments with their timestamps, cuts those moments out of the film and joins them into a recap
shorter than ten minutes. The summary is evaluated with BLEU.

**Assignment objectives**
1. Implement a multimodal pipeline that processes video, audio and text.
2. Apply speech-to-text and summarisation to produce a concise summary story.
3. Identify peak points and map them to time indices in the original film.
4. Extract the relevant segments (OpenCV) and combine them into a recap of less than ten minutes.

**Workflow**

`YouTube Video → Movie → Audio + Video → Transcript → Summary Story → Peak Points → Time Index → Video Segment Extraction → Movie Recap → BLEU Evaluation`

**Input film.** *Detour* (1945, dir. Edgar G. Ulmer), a public-domain film noir of 67 minutes with heavy
first-person voice-over narration. Public-domain status allows the recap clips to be shared freely, and a
human-written Wikipedia plot summary serves as the BLEU reference.

**How to read this notebook.** Each component follows the same sequence: *Objective → Purpose → Methodology →
Prompt (only where an instruction-following AI model is called) → Implementation → Coding → Output → Inference →
Insights → Summary*. The full pipeline lives in the `recap/` package (`python run_pipeline.py`). The slowest stages
(22 min of transcription, about 33 min of LLM summarisation on a CPU) cache their results as JSON; this notebook loads those
results, re-runs the lighter components live, and repeats one live LLM call per prompt so each prompt can be seen in action.

**Where prompts appear.** Only the summarisation component (Section 6) uses a model that follows written
instructions (Qwen2.5-7B-Instruct). It has three prompts (notes, editor, repair), each shown directly before the call it controls. Whisper, the emotion classifier and the sentence-embedding model take raw audio
or text with no instructions, and the remaining components are algorithmic, so they have no Prompt step.
""")

# ============================================================================ 1 environment
head("1", "Environment and reproducibility",
     "Prepare a reproducible environment: fixed random seeds, recorded library versions and one shared configuration.",
     "Every later component reads the same paths and parameters; without a fixed environment the reported numbers could not be reproduced.",
     "The project root is added to the import path, `config.yaml` is loaded through the same helper the pipeline uses, "
     "seeds are fixed for Python and NumPy, and the versions of the key libraries are collected into a table.")
implementation("The cell below imports the shared libraries, sets the seed (42), loads the configuration and defines plotting colours. "
               "A second cell lists the library versions used for the reported results.")
code("""
# Imports, seeds, configuration and plotting style shared by every section
import json, random, sys, time, warnings
from pathlib import Path

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from recap.utils import load_config, load_json, fmt_ts

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

cfg = load_config(ROOT / "config.yaml")
OUT = Path(cfg["output_dir"])
VIDEO = cfg["input"]["local_path"]
pd.set_option("display.max_colwidth", 120)
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
""")
output("Library versions used for every result in this notebook:")
code("""
import faster_whisper, sentence_transformers, transformers, nltk, sacrebleu
versions = {"python": sys.version.split()[0], "opencv": cv2.__version__, "librosa": librosa.__version__,
            "faster-whisper": faster_whisper.__version__, "transformers": transformers.__version__,
            "sentence-transformers": sentence_transformers.__version__, "nltk": nltk.__version__,
            "sacrebleu": sacrebleu.__version__, "numpy": np.__version__}
pd.DataFrame([versions]).T.rename(columns={0: "version"})
""")
close("The environment loads without errors and the versions above are the ones behind the reported results.",
      "Every model in the pipeline is pre-trained and used zero-shot, so there is no training step. Reproducibility therefore "
      "depends on fixed model versions, temperature 0 for the LLM and a fixed seed for the random baseline.",
      "With a fixed environment in place, the pipeline can start from its input: the YouTube film.")

# ============================================================================ 2 input
head("2", "Input: YouTube video → movie file",
     "Obtain the film from YouTube and confirm that the video stream is complete and readable.",
     "Every later stage (audio, transcript, features, clip extraction) depends on a valid, correctly-tracked source file.",
     "`recap/ingest.py` downloads the film with **yt-dlp** at ≤480p and explicitly selects the **original English audio "
     "track** (this upload also carries auto-dubbed Russian, Korean and Portuguese tracks). **OpenCV** reads the stream "
     "properties (frame rate, frame count, resolution) and samples frames evenly across the running time.")
implementation("`probe_video()` opens the file with `cv2.VideoCapture` and returns its properties. The loop then seeks to six "
               "evenly spaced positions (10%–95% of the film) and reads one frame at each.")
code("""
# Probe the film with OpenCV and sample six evenly spaced frames
from recap.ingest import probe_video

info = probe_video(VIDEO)
cap = cv2.VideoCapture(VIDEO)
sample_times = np.linspace(0.1, 0.95, 6) * info["duration_seconds"]
sample_frames = []
for t in sample_times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * info["fps"]))  # seek by frame index
    ok, frame = cap.read()
    sample_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
""")
output("Stream properties and the six sampled frames:")
code("""
display(pd.DataFrame([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items() if k != "path"}]))
fig, axes = plt.subplots(1, 6, figsize=(16, 2.6))
for ax, t, frame in zip(axes, sample_times, sample_frames):
    ax.imshow(frame)
    ax.set_title(fmt_ts(t), fontsize=9)
    ax.axis("off")
fig.suptitle("Frames sampled across the film (OpenCV)", x=0.01, ha="left", fontweight="bold")
plt.tight_layout()
plt.show()
""")
chart_note("Six frames from 00:00 to the final minutes show the black-and-white picture, steady framing and mostly interior "
           "or night scenes.",
           "Colour carries no information in this film; brightness, motion and cuts are the only usable visual cues.",
           "The visual features in Section 5 are therefore built on grey-level motion and histogram cuts, not on colour.")
close("The film is 640×480 at 30 fps and runs for 67 min 38 s; the sampled frames are black and white and evenly "
      "cover the story from the opening to the ending.",
      "Choosing the audio track is a real multimodal pitfall: a blind default could have fed a machine-dubbed voice to speech "
      "recognition. Because the footage is black and white, hue and saturation carry almost no information, so the visual "
      "features later rely on brightness and motion.",
      "A valid video file is in place; the next component separates its soundtrack from the picture.")

# ============================================================================ 3 audio
head("3", "Audio extraction: movie → audio + video",
     "Separate the soundtrack from the picture in a format suitable for speech recognition and audio analysis.",
     "Speech-to-text and the audio features operate on the soundtrack alone, while OpenCV keeps working on the video stream.",
     "**ffmpeg** (bundled with `imageio-ffmpeg`, so no system install is needed) demuxes the soundtrack to a 16 kHz mono WAV, "
     "the format Whisper expects, plus an MP3 copy. A log-mel spectrogram, $S_{dB} = 10\\log_{10}(\\text{Mel}(|STFT(y)|^2))$, "
     "visualises the speech energy of a short excerpt.")
implementation("`librosa.load` reads the extracted WAV. The full waveform is downsampled only for plotting, and a 20-second dialogue "
               "excerpt at 08:03 is converted to a 64-band log-mel spectrogram.")
code("""
# Load the extracted soundtrack and compute a log-mel spectrogram of one dialogue excerpt
wav_path = OUT / "audio" / "audio_16k_mono.wav"
y, sr = librosa.load(wav_path, sr=16000)
excerpt_start = 8 * 60 + 3  # a dialogue scene at 08:03
excerpt = y[excerpt_start * sr:(excerpt_start + 20) * sr]
mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=excerpt, sr=sr, n_mels=64), ref=np.max)
""")
output()
code("""
print(f"Soundtrack: {len(y) / sr / 60:.1f} min at {sr} Hz, peak amplitude {np.abs(y).max():.2f}")
fig, axes = plt.subplots(1, 2, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
step = 400  # downsample for plotting only
axes[0].plot(np.arange(0, len(y), step) / sr / 60, y[::step], color=BLUE, lw=0.4)
axes[0].set(title="Waveform of the full soundtrack", xlabel="Film time (minutes)", ylabel="Amplitude")
img = axes[1].imshow(mel_db, origin="lower", aspect="auto", cmap="magma", extent=[0, 20, 0, 64])
axes[1].set(title=f"Log-mel spectrogram, {fmt_ts(excerpt_start)} + 20 s", xlabel="Seconds", ylabel="Mel band")
fig.colorbar(img, ax=axes[1], label="dB")
plt.tight_layout()
plt.show()
""")
chart_note("The waveform stays well below full scale for the whole film with short louder bursts; the spectrogram of the "
           "08:03 excerpt shows stacked harmonic bands separated by gaps.",
           "Harmonic bands with gaps are the signature of voiced speech and pauses between lines; the louder bursts are music cues "
           "and raised voices.",
           "The soundtrack is clean enough for speech recognition, and loudness bursts are a usable arousal signal for peaks.")
close("The waveform peaks below half of full scale, and the spectrogram shows stacked harmonic bands of voices separated "
      "by short pauses, so the soundtrack carries clean, recognisable speech.",
      "A 16 kHz mono signal keeps the band up to 8 kHz where most speech energy lies while cutting the data about six-fold "
      "compared with 48 kHz stereo. The 1945 soundtrack is quiet: clean text-to-speech narration was far louder and at first "
      "masked the film dialogue in the recap, so the renderer now matches narration loudness to the film's speech level.",
      "The speech track is ready for transcription, the first text-producing component.")

# ============================================================================ 4 ASR
head("4", "Speech-to-text: transcript with a time index",
     "Convert the soundtrack into text while recording when every line and word was spoken.",
     "These timestamps are the *time index* that links summary sentences, peak points and video segments throughout the pipeline.",
     "`faster-whisper` runs OpenAI's **Whisper `small.en`** (244 M parameters, encoder–decoder transformer) through CTranslate2 "
     "with int8 weights on the CPU, with voice-activity filtering and word-level timestamps. `condition_on_previous_text=False` "
     "prevents the repetition loops Whisper can fall into on long audio. Decoding confidence per segment is "
     "$\\exp(\\overline{\\log p})$, the exponent of the average token log-probability. Whisper takes audio only and is not "
     "given instructions, so this component has no prompt.")
implementation("The full-film transcription took 22 minutes and is cached in `transcript.json`; the cell loads it and derives "
               "word counts, speech coverage and per-segment confidence.")
code("""
# Load the cached Whisper transcript and compute summary statistics
tr = load_json(OUT / "transcript.json")
segs = tr["segments"]
words = sum(len(s["text"].split()) for s in segs)
conf = np.exp([s["avg_logprob"] for s in segs])
speech_time = sum(s["end"] - s["start"] for s in segs)
""")
output("Transcript statistics, eight sample lines with their time index, word-level timestamps and the confidence distribution:")
code("""
display(pd.DataFrame([{"segments": len(segs), "words": words, "speech minutes": round(speech_time / 60, 1),
                       "share of film with speech": f"{speech_time / tr['info']['audio_duration']:.0%}",
                       "ASR runtime (min)": round(tr["info"]["processing_seconds"] / 60, 1),
                       "median confidence": round(float(np.median(conf)), 3)}]))
display(pd.DataFrame([{"start": fmt_ts(s["start"]), "end": fmt_ts(s["end"]), "text": s["text"]} for s in segs[60:68]]))
print("Word-level timestamps for one line:", [(w["word"].strip(), w["start"]) for w in segs[64]["words"]])
plt.figure(figsize=(8, 3))
plt.hist(conf, bins=30, color=BLUE, edgecolor="white")
plt.title("Whisper decoding confidence per transcript segment")
plt.xlabel("exp(average log-probability)")
plt.ylabel("Number of segments")
plt.tight_layout()
plt.show()
""")
chart_note("Most segments sit at high decoding confidence, with a thin tail of low-confidence lines.",
           "The tail is mainly music, overlapping speech and one-word exclamations, not whole misheard scenes.",
           "Transcript errors are local; summarisation over 5-minute chunks is robust to the occasional wrong line.")
close("The transcript contains 946 segments covering a little under half of the running time, and the confidence "
      "histogram is concentrated at the high end with a thin tail of uncertain lines.",
      "Speech filling under half of a feature film is normal (music, silent action and pauses fill the rest). Low-confidence "
      "lines are mostly music, overlapping speech or short exclamations. At about 3× faster than real time on a CPU, "
      "transcription is the second most expensive stage, which is why it is cached.",
      "The timestamped transcript is the text modality used by summarisation and by peak detection.")

# ============================================================================ 5 features
head("5", "Multimodal signals: audio, video and text emotion",
     "Measure how intense each moment of the film is in each modality.",
     "Peak points should not be chosen from text alone: the dramatic moment in a scene is often marked by a raised voice, "
     "fast editing or emotional wording.",
     "* **Audio (librosa):** RMS loudness, onset strength (spectral flux) and amplitude fluctuation within each second "
     "(the session heuristic: calm speech has steady peak amplitude, agitated speech does not), combined as "
     "arousal $= 0.4\\,\\hat{L}_{dB} + 0.3\\,\\hat{O} + 0.3\\,\\hat{\\sigma}_{RMS}$ (hats = min-max scaled).\n"
     "* **Video (OpenCV):** one pass at 2 frames per second; a shot cut is declared when the Bhattacharyya distance between "
     "consecutive HSV histograms exceeds 0.35; motion is the mean absolute grey-level difference; "
     "activity $= 0.6\\,\\widehat{\\log(1+m)} + 0.4\\,\\widehat{\\text{cut density}}$.\n"
     "* **Text (transformers):** a DistilRoBERTa emotion classifier labels each transcript line; intensity $= 1 - P(\\text{neutral})$.\n\n"
     "The modalities are computed independently and fused later (post-fusion). None of these models takes instructions, so "
     "there is no prompt.")
implementation("The three feature files were produced by `recap/features.py` during the pipeline run. The cell loads them and "
               "derives shot statistics and the distribution of emotion labels.")
code("""
# Load the cached per-modality features and summarise each one
fa, fv, fe = (load_json(OUT / f) for f in ("features_audio.json", "features_visual.json", "features_text_emotion.json"))
shots = np.array(fv["shot_boundaries"])
emotion_counts = pd.Series([e["label"] for e in fe]).value_counts()
""")
output()
code("""
print(f"Shot cuts detected: {len(shots)}  |  median shot length: {np.median(np.diff(shots)):.1f} s")
print(f"Audio arousal: mean {np.mean(fa['arousal']):.2f}, 95th percentile {np.percentile(fa['arousal'], 95):.2f}")
print(f"Visual activity: mean {np.mean(fv['activity']):.2f}, 95th percentile {np.percentile(fv['activity'], 95):.2f}")
plt.figure(figsize=(8, 3.2))
plt.bar(emotion_counts.index, emotion_counts.values, color=[GREY if l == "neutral" else AQUA for l in emotion_counts.index])
for i, v in enumerate(emotion_counts.values):
    plt.text(i, v + 5, str(v), ha="center", fontsize=9)
plt.title("Dominant emotion of transcript segments (DistilRoBERTa)")
plt.xlabel("Emotion label")
plt.ylabel("Number of segments")
plt.tight_layout()
plt.show()
""")
chart_note("Neutral is the largest class; among the non-neutral labels surprise, disgust and anger lead and joy is rare.",
           "A tense film noir produces many suspicious and hostile exchanges, so non-neutral lines cluster around the conflicts.",
           "Emotion intensity (1 - P(neutral)) separates dramatic lines from routine dialogue and earns its 15 % fusion weight.")
close("About half of the lines are neutral; surprise, disgust and anger are the most common non-neutral emotions and joy "
      "is rare. 312 shot cuts were detected.",
      "The emotional profile fits a film noir: tense, suspicious and hostile exchanges dominate, so non-neutral lines are a "
      "useful marker of dramatic moments. The visual features are low-level (motion, cuts); scenes that are important but "
      "visually calm depend on the other modalities.",
      "The three per-moment intensity signals are ready to be fused when the peak points are chosen (Section 7).")

# ============================================================================ 6 summary (prompts)
PROMPT_SRC = Path(__file__).resolve().parents[1].joinpath("recap", "summarize.py").read_text(encoding="utf-8")


def prompt_text(name: str) -> str:
    """Return the exact prompt template `name` from recap/summarize.py, so the notebook always shows the live prompt."""
    return PROMPT_SRC.split(f'{name} = """')[1].split('"""')[0].replace('\\"', '"').strip()


head("6", "Text processing and summarisation: the summary story",
     "Turn about 9,000 words of transcript into a short summary story that a viewer can follow from start to end, with the "
     "right people doing the right things and the climax and ending included.",
     "The summary story decides what the recap tells: every story sentence later becomes one peak point, one clip and one "
     "narrated chapter card. An error here (a wrong character, a missing ending) is carried into the final video.",
     "The transcript is split into 13 chunks of about 5 minutes. A local LLM, **Qwen2.5-7B-Instruct** served by Ollama at "
     "temperature 0 (deterministic decoding, seed 42), processes them in order with the story written so far as context "
     "(*rolling context*). Version 2 of the summariser adds four grounding steps on top of the rolling context used in run E:\n\n"
     "1. **Grounded notes (LLM call per chunk).** The model first lists 2–4 events as *who – does what – to whom (quote)*, with "
     "a short quote from the excerpt as evidence, and only then writes exactly 2 story sentences from those notes (3 for the final chunk, which holds the climax and the ending); if it stops after the notes, a short fallback call writes them.\n"
     "2. **Editor pass (one LLM call).** The whole draft is checked against the notes of every chunk: role errors are corrected, "
     "repeated events removed, and missing key events (deaths, arrests, the ending) restored.\n"
     "3. **Key-event check (deterministic) with a targeted addition (LLM call only when needed).** Transcript lines with strong "
     "evidence of a death, killing or murder are located per chunk (at least two such lines, negations such as \"didn't kill\" "
     "ignored). If the story says nothing about them for that chunk, one call writes the missing sentence from the quoted lines.\n"
     "4. **Character-entry check (deterministic) with a targeted repair (LLM call only when needed).** For each cast member "
     "the time $t_c$ at which the name is first spoken is found in the transcript. A sentence from a chunk ending at $t_e$ may "
     "not name character $c$ if $t_c > t_e$; offending sentences are rewritten by one repair call and re-checked.\n\n"
     "A near-duplicate filter (cosine similarity ≥ 0.88) and a TextRank extractive summary of the same length (baseline) are "
     "kept from version 1. Each sentence keeps the time span of its chunk, the first and coarse level of the time index.")

md(f"""
### Prompt
**Prompt 1 of 4: grounded notes and story** (`NOTES_PROMPT` in `recap/summarize.py`), sent once per chunk. The values in
braces are filled in by the code cell below.

```text
{prompt_text("NOTES_PROMPT")}
```

**Filled-in values for this call:** `title` = "Detour (1945)"; `cast` = the cast list in `config.yaml`; `context` = the last
six draft sentences written before chunk 4; `start`–`end` = the time span of chunk 4; `n` = "2"; `text` = the transcript
of chunk 4. Decoding: temperature 0, seed 42, at most 520 new tokens. Chunk 4 is shown because it is the part of the film in
which the earlier run E wrongly placed Vera in the car.

**Fallback in the same call path** (`STORY_PROMPT`): if the model stops after the notes and writes no story, one short
call turns that chunk's notes into the sentences (at most 160 new tokens).

```text
{prompt_text("STORY_PROMPT")}
```
""")
implementation("The summary was produced by the pipeline (`summary.json`, all 13 chunks plus the editor and repair calls) and "
               "is loaded from cache. The three cells below then repeat **one live call for each prompt** with exactly the "
               "prompts shown, so each prompt can be seen controlling the model: (1) the notes call for chunk 4, (2) the "
               "editor pass over the full draft, (3) the key-event check followed by an addition call for the climax that "
               "run E left out, and (4) the character-entry check followed by a repair call on the run-E sentence that broke "
               "the rule. Every live call needs Ollama with `qwen2.5:7b`; if it is unreachable, the cell "
               "falls back to the cached pipeline output and says so.")
code("""
# Coding, call 1 of 4: grounded NOTES -> STORY for chunk 4 with the prompt shown above
import requests
from recap.summarize import (KEY_EVENT_PROMPT, KEY_STORY, NOTES_PROMPT, STORY_PROMPT, REVISE_PROMPT, REPAIR_PROMPT, _clean, cast_characters,
                             entry_violations, first_mentions, key_lines, ollama_available, ollama_generate, parse_notes_story, parse_revision,
                             split_sentences)


def llm(prompt, max_tokens, ctx=4096):
    # One live call; returns (text, source) and never raises, so the notebook runs end to end without Ollama
    if not LIVE:
        return None, "Ollama not reachable: cached pipeline output shown"
    t = time.time()
    try:
        return ollama_generate(prompt, cfg["summary"], max_tokens, num_ctx=ctx), f"live call ({time.time() - t:.0f} s)"
    except requests.RequestException as exc:
        return None, f"live call failed ({exc.__class__.__name__}): cached pipeline output shown"


summ = load_json(OUT / "summary.json")
LIVE = ollama_available(cfg["summary"])
draft = summ.get("draft_sentences", summ["story_sentences"])
draft_chunks = summ.get("draft_chunks", summ["sentence_chunks"])
k = 3                                                         # chunk 4 (zero-based index 3)
chunk = summ["chunks"][k]
previous = [s for s, c in zip(draft, draft_chunks) if c < k][-6:]   # same context window as the pipeline
prompt_notes = NOTES_PROMPT.format(title=cfg["input"]["title"], cast=cfg["input"]["cast"],
                                   context=" ".join(previous) or "(this is the beginning of the film)",
                                   start=fmt_ts(chunk["start"]), end=fmt_ts(chunk["end"]), n="2", text=chunk["text"])
raw, src_notes = llm(prompt_notes, 520)
live_notes, live_story = parse_notes_story(raw) if raw else (chunk.get("notes", []), chunk.get("summary", ""))
if raw and not live_story and live_notes:
    # Same fallback as the pipeline: write the story from the notes
    raw_story, _ = llm(STORY_PROMPT.format(title=cfg["input"]["title"], start=fmt_ts(chunk["start"]), end=fmt_ts(chunk["end"]),
                                           n="2", notes="\\n".join(f"- {x}" for x in live_notes)), 160)
    live_story = _clean(raw_story or "")
""")
md(f"""
### Prompt
**Prompt 2 of 4: editor pass** (`REVISE_PROMPT` in `recap/summarize.py`), sent once for the whole film. `draft` = every draft
sentence tagged with its chunk number; `notes` = the event notes of every chunk; `words` = the draft length in words. Decoding: temperature 0, at most 900 new
tokens, 8,192-token context.

```text
{prompt_text("REVISE_PROMPT")}
```
""")
code("""
# Coding, call 2 of 4: editor pass over the full draft with the prompt shown above
draft_text = "\\n".join(f"[{c + 1}] {s}" for s, c in zip(draft, draft_chunks))
notes_text = "\\n".join(f"[{c['index'] + 1}] " + "; ".join(c.get("notes", [])) for c in summ["chunks"])
prompt_rev = REVISE_PROMPT.format(title=cfg["input"]["title"], cast=cfg["input"]["cast"], draft=draft_text, notes=notes_text,
                                  words=sum(len(s.split()) for s in draft))
raw_rev, src_rev = llm(prompt_rev, 900, ctx=8192)
live_revised = parse_revision(raw_rev or summ.get("revision_raw", ""), len(summ["chunks"]), max(6, len(draft) // 2))
""")
md(f"""
### Prompt
**Prompt 3 of 4: key-event addition** (`KEY_EVENT_PROMPT` in `recap/summarize.py`), sent only for a chunk whose transcript
has strong evidence of a death, killing or murder that the story does not mention. `story` = the story sentences of that
chunk; `lines` = the evidence lines with their timestamps. Decoding: temperature 0, at most 90 new tokens.

```text
{prompt_text("KEY_EVENT_PROMPT")}
```
""")
code("""
# Coding, call 3 of 4: key-event check on run E and on the final story, then a live addition for the climax chunk
run_e = load_json(OUT.parent / "experiments" / "summary_qwen2.5-7b.json")


def key_event_gaps(story):
    # Chunks with strong death/killing evidence in the dialogue but no matching story sentence
    gaps = []
    for c in story["chunks"]:
        lines = key_lines(c, segs)
        told = [s for s, k in zip(story["story_sentences"], story["sentence_chunks"]) if k == c["index"]]
        if lines and not any(KEY_STORY.search(t) for t in told):
            gaps.append({"chunk": c["index"] + 1, "span": f"{fmt_ts(c['start'])}-{fmt_ts(c['end'])}",
                         "evidence": " | ".join(lines[:3]), "story says": " ".join(told)})
    return gaps


gaps_e, gaps_final = key_event_gaps(run_e), key_event_gaps(summ)
c = summ["chunks"][-1]                                     # the climax and ending of the film
told = [s for s, k in zip(run_e["story_sentences"], run_e["sentence_chunks"]) if k == c["index"]]
prompt_key = KEY_EVENT_PROMPT.format(title=cfg["input"]["title"], cast=cfg["input"]["cast"], start=fmt_ts(c["start"]),
                                     end=fmt_ts(c["end"]), story=" ".join(told) or "(none)", lines="\\n".join(key_lines(c, segs)))
raw_key, src_key = llm(prompt_key, 90)
added_live = (split_sentences(_clean(raw_key)) or [_clean(raw_key)])[0] if raw_key else "(Ollama not reachable)"
""")
md(f"""
### Prompt
**Prompt 4 of 4: targeted repair** (`REPAIR_PROMPT` in `recap/summarize.py`), sent only for a sentence that fails the
deterministic character-entry check. `names` = the character named too early; `sentence` = the offending sentence;
`text` = the transcript of its chunk. Decoding: temperature 0, at most 80 new tokens.

```text
{prompt_text("REPAIR_PROMPT")}
```
""")
code("""
# Coding, call 4 of 4: character-entry check on run E and on the final story, then a live repair of the run-E error
chars = cast_characters(cfg["input"]["cast"])
first = first_mentions(segs, chars)

def entry_audit(story):
    # One row per sentence that names a character before the name is first spoken in the film
    return [{"#": i + 1, "chunk ends": fmt_ts(story["chunks"][c]["end"]), "named too early": ", ".join(bad), "sentence": s}
            for i, (s, c) in enumerate(zip(story["story_sentences"], story["sentence_chunks"]))
            if (bad := entry_violations(s, story["chunks"][c]["end"], chars, first))]


audit_e, audit_final = entry_audit(run_e), entry_audit(summ)
repair_demo = None
if audit_e:
    i = audit_e[0]["#"] - 1
    c = run_e["chunks"][run_e["sentence_chunks"][i]]
    names = audit_e[0]["named too early"].replace(", ", " or ")
    prompt_rep = REPAIR_PROMPT.format(title=cfg["input"]["title"], names=names, start=fmt_ts(c["start"]),
                                      end=fmt_ts(c["end"]), sentence=run_e["story_sentences"][i], text=c["text"])
    raw_rep, src_rep = llm(prompt_rep, 80)
    fixed = (split_sentences(_clean(raw_rep)) or [_clean(raw_rep)])[0] if raw_rep else "(Ollama not reachable)"
    repair_demo = {"run-E sentence": run_e["story_sentences"][i], "repaired (live)": fixed, "source": src_rep,
                   "passes the check": not entry_violations(fixed, c["end"], chars, first)}
""")
output("Call 1 (live versus pipeline), call 2 (editor result), call 3 (key-event check and addition), call 4 (entry check and repair), and the final story:")
code("""
print(f"Call 1 - notes for chunk 4 ({fmt_ts(chunk['start'])}-{fmt_ts(chunk['end'])}), {src_notes}")
for n in live_notes:
    print("   NOTE :", n)
print("   LIVE STORY    :", live_story)
print("   PIPELINE STORY:", chunk.get("summary", ""))
print(f"\\nCall 2 - editor, {src_rev}: {len(draft)} draft sentences -> "
      f"{len(live_revised) if live_revised else 0} revised sentences (pipeline final: {len(summ['story_sentences'])})")
print("Call 3 - key-event check. Run E gaps:")
display(pd.DataFrame(gaps_e) if gaps_e else Markdown("*none*"))
print(f"   Live addition for the climax chunk ({src_key}):", added_live)
print("   Final (v2) story gaps:", len(gaps_final), "| additions logged by the pipeline:",
      [a.get("added", a["action"]) for a in summ.get("key_event_check", [])])
display(pd.DataFrame([{"name": ch["name"], "first spoken at": fmt_ts(first[ch["name"]]) if first[ch["name"]] != float("inf") else "never",
                       "exempt (narrator)": ch["narrator"]} for ch in chars]))
print("Call 4 - character-entry check. Run E violations:")
display(pd.DataFrame(audit_e) if audit_e else Markdown("*none*"))
if repair_demo:
    display(pd.DataFrame([repair_demo]).T.rename(columns={0: ""}))
print("Final (v2) story violations:", len(audit_final), "| repairs logged by the pipeline:", len(summ.get("entry_check", [])))
words = sum(len(s["text"].split()) for s in segs)
print(f"\\nBackend: {summ['backend']} ({summ.get('method', 'rolling')})  |  {len(summ['story_sentences'])} sentences, "
      f"{len(summ['story'].split())} words (compression {words / len(summ['story'].split()):.0f}:1 versus the transcript)")
pd.DataFrame([{"#": i + 1, "chunk time span": f"{fmt_ts(summ['chunks'][c]['start'])}-{fmt_ts(summ['chunks'][c]['end'])}",
               "story sentence": s} for i, (s, c) in enumerate(zip(summ["story_sentences"], summ["sentence_chunks"]))])
""")
close("The final story has 24 sentences and 410 words and follows the whole film: the club in New York, Sue leaving for "
      "Hollywood, the hitchhike, Haskell's death and the stolen identity, Vera's blackmail, her threat to call the police, "
      "her accidental death and Al's flight. The character-entry check finds no violation in the final story (run E had "
      "one: Vera named about 14 minutes before her name is first spoken), and the key-event check added the climax that "
      "run E and the first v2 run had left out: \"Al kills Vera accidentally, and now the police suspect him of her murder.\"",
      "* **Grounding works where it is deterministic.** Both checks are plain rules on the transcript and are always right "
      "about *whether* something is missing or premature; the LLM is only asked to fill a narrow, quoted gap.\n"
      "* **The 7B model still makes role errors.** Without speaker labels it reverses who accuses whom in sentences 12 and 14 "
      "(Vera accuses Al, not the other way round), calls Vera \"Haskell\" once (sentence 10) and garbles Haskell's anecdote "
      "about a scratching hitchhiker (sentence 6). These are listed openly in Section 11 and in `SUMMARY_EXPERIMENTS.md`.\n"
      "* **Iteration evidence.** Run v2a (no fallback) lost the story text of 4 of 13 chunks; v2b wrote one sentence per chunk "
      "(213 words, brevity penalty 0.25); v2c fixed length but its added sentences were first-person and partly invented; "
      "the final run rejects first-person or copied additions and keeps the accidental nature of the death.",
      "The summary story is the content backbone of the recap: 24 sentences, each tagged with its 5-minute chunk, ready to be "
      "pinned to exact moments by the next component.")

# ============================================================================ 7 peaks
head("7", "Peak points and time-index mapping",
     "For every story sentence, find the single moment in the film that best shows it, and record its timestamp.",
     "Peak points turn the abstract story into concrete positions in the film; their timestamps drive the clip extraction.",
     "1. **Alignment.** The transcript is cut into 30-second windows (10-second hop). Story sentences and windows are embedded "
     "with `all-MiniLM-L6-v2` (cosine similarity). A **monotonic dynamic-programming alignment** chooses one window $j_i$ per "
     "sentence $i$ maximising $\\sum_i \\text{sim}(i, j_i)$ subject to $j_i \\ge j_{i-1} + g$, so windows follow story order; each "
     "sentence may only align inside its own chunk's time span.\n"
     "2. **Peak picking (post-fusion).** Within ±45 s of the aligned window, every transcript segment is scored "
     "$F = 0.40\\,\\text{semantic} + 0.15\\,\\text{centrality} + 0.15\\,\\text{emotion} + 0.15\\,\\text{audio} + 0.15\\,\\text{visual}$. "
     "The best segment is the peak point and its Whisper start time is the **time index**.\n\n"
     "This component is algorithmic (embeddings plus dynamic programming), so there is no prompt.")
implementation("The first code cell runs the alignment component on a toy matrix to show that order is enforced. The second "
               "recomputes the sentence–window similarity matrix for the real film and loads the peaks chosen by the pipeline.")
code("""
# Component test: the DP alignment keeps story order even when a later column scores higher
from recap.peaks import monotonic_alignment, build_windows
from recap.summarize import embed

toy = np.array([[0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.5]])
greedy, aligned = toy.argmax(axis=1).tolist(), monotonic_alignment(toy, min_gap=1)
""")
code("""
# Real film: story-sentence x transcript-window similarity and the peaks chosen by the pipeline
pk = load_json(OUT / "peaks.json")
windows = build_windows(segs, cfg["peaks"]["window_seconds"], cfg["peaks"]["window_hop_seconds"])
model = cfg["peaks"]["embedding_model"]
sim = embed(summ["story_sentences"], model) @ embed([w["text"] for w in windows], model).T
centres = np.array([(w["start"] + w["end"]) / 2 for w in windows]) / 60
""")
output("Toy alignment, the similarity heatmap with the alignment path and chosen peaks, the peak table, and the film-wide timeline:")
code("""
print("Greedy best window per sentence:", greedy, "(out of order)")
print("Monotonic DP alignment:        ", aligned, "(order preserved)")
plt.figure(figsize=(14, 4.2))
plt.imshow(sim, aspect="auto", cmap="Blues", extent=[centres[0], centres[-1], len(sim) + 0.5, 0.5])
plt.colorbar(label="cosine similarity")
plt.plot([centres[j] for j in pk["alignment_path"]], np.arange(1, len(sim) + 1), "o-", color=ORANGE, ms=5, lw=1.2,
         label="DP alignment path")
plt.scatter([p["timestamp"] / 60 for p in pk["peaks"]], np.arange(1, len(sim) + 1), marker="x", color="black",
            s=50, label="selected peak point", zorder=3)
plt.title("Story sentence vs. transcript window similarity, with the alignment path and peak points")
plt.xlabel("Film time (minutes)")
plt.ylabel("Story sentence #")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
display(pd.DataFrame([{"#": p["index"], "time index": p["timestamp_hms"], "peak line (transcript)": p["transcript"][:80],
                       "emotion": p["emotion_label"], **p["scores"]} for p in pk["peaks"]]))
display(Image(filename=str(OUT / "timeline.png"), width=1100))
""")
chart_note("The orange alignment path climbs steadily from the top-left to the bottom-right of the heatmap, and every "
           "selected peak (x) sits on or next to it. Fused peak scores range from 0.39 to 0.85, and the 24 peaks spread from "
           "03:02 to 01:05:56.",
           "Story order and film order agree, so no sentence was matched to an unrelated scene that only shares a name. "
           "Pairs of sentences from the same chunk land a few seconds apart (for example peaks 10 and 11 at 36:23 and 36:26), "
           "because both describe the same scene.",
           "The climax sentence lands exactly on the line \"after killing Vera without really meaning to do it\" (01:03:19), "
           "which shows the time index pointing at the right moment.")
close("The toy test confirms that the dynamic-programming alignment keeps story order where a greedy match does not. On the "
      "real film every story sentence receives one peak point with a precise Whisper timestamp, and the timeline shows the "
      "clips sitting on those peaks across the whole running time.",
      "Without the order constraint, a sentence about the ending could match an early scene that mentions \"Haskell\" or "
      "\"the police\". Tying each sentence to its own chunk keeps the recap balanced across the story arc. A limitation is that "
      "two sentences of one chunk can choose neighbouring peaks, which later produces two short clips of the same scene.",
      "Two levels of time index now exist for every sentence, the chunk span and the exact peak timestamp, both stored in "
      "`peaks.csv`. The next component cuts the film at these timestamps.")

# ============================================================================ 8 segments & recap
head("8", "Video segment extraction (OpenCV) and the movie recap",
     "Cut the film at the peak timestamps and assemble a recap shorter than ten minutes.",
     "This is the final product of the assignment: the summary, peak points and time indices become a watchable video.",
     "`recap/segments.py` grows each peak line into a clip using the neighbouring dialogue that best matches its story sentence "
     "(up to about 32 s), pads it, snaps both ends to the nearest detected shot cut, and keeps clips chronological and "
     "non-overlapping within a 570-second budget. `recap/render.py` uses **OpenCV** to seek frame-accurately "
     "(`CAP_PROP_POS_FRAMES`), read and annotate every frame (original timecode, peak badge, white flash at the exact peak, "
     "film-position bar) and to draw chapter cards that narrate each story sentence with offline text-to-speech, "
     "loudness-matched to the film. Audio is assembled sample-locked to the number of written frames; ffmpeg only muxes and "
     "encodes. The narration engine reads fixed text and takes no instructions, so there is no prompt.")
implementation("The first cell re-extracts the frame at each peak's time index with OpenCV. The second loads the edit decision "
               "list (EDL) and asserts the duration limit. The third grabs three frames from the finished recap.")
code("""
# Re-extract the frame at every peak's time index with OpenCV
clips = load_json(OUT / "clips.json")
cap = cv2.VideoCapture(VIDEO)
peak_frames = []
for p in pk["peaks"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(p["timestamp"] * info["fps"]))
    ok, frame = cap.read()
    peak_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
""")
code("""
# Edit decision list and the hard duration check required by the assignment
edl = pd.read_csv(OUT / "recap_edl.csv")
recap_seconds = load_json(OUT / "render_info.json")["duration_seconds"]
assert recap_seconds < 600, "recap must be shorter than 10 minutes"
footage = edl[edl.type == "clip"].eval("recap_end - recap_start").sum()
cards = edl[edl.type != "clip"].eval("recap_end - recap_start").sum()
""")
code("""
# Three frames from the finished recap: a chapter card, a clip, and the clip at its exact peak moment
cap = cv2.VideoCapture(str(OUT / "movie_recap.mp4"))
fps_r = cap.get(cv2.CAP_PROP_FPS)
row_card, row_clip = edl[edl.type == "chapter"].iloc[4], edl[edl.type == "clip"].iloc[4]
at_peak = row_clip.recap_start + (pk["peaks"][4]["timestamp"] - row_clip.source_start)
recap_frames = []
for t in (row_card.recap_start + 3, row_clip.recap_start + 1, at_peak + 0.1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps_r))
    ok, frame = cap.read()
    recap_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
""")
output()
code("""
cols = 5
fig, axes = plt.subplots(int(np.ceil(len(peak_frames) / cols)), cols, figsize=(16, 3.1 * np.ceil(len(peak_frames) / cols)))
for ax in axes.flat:
    ax.axis("off")
for ax, p, frame in zip(axes.flat, pk["peaks"], peak_frames):
    ax.imshow(frame)
    ax.set_title(f"Peak {p['index']} @ {p['timestamp_hms']}", fontsize=9)
fig.suptitle("Frame at every peak point (time index)", x=0.01, ha="left", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"Recap: {fmt_ts(recap_seconds)} ({recap_seconds:.1f} s) -> under 10 minutes: {recap_seconds < 600}")
print(f"Footage {footage:.0f} s, cards {cards:.0f} s, compression {info['duration_seconds'] / recap_seconds:.1f}x")
display(edl.assign(text=edl.text.str.slice(0, 70)))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, frame, title in zip(axes, recap_frames, ["Chapter card (story sentence 5)", "Clip 5 with time-index badges",
                                                 "Exact peak moment (white border)"]):
    ax.imshow(frame)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
""")
chart_note("The peak frames show the right characters at the right points (the club, the road, the car, the apartment), "
           "and the three recap frames show a chapter card, a clip with its original-time badge, and the white border that "
           "marks the exact peak moment.",
           "Every clip carries its own provenance on screen: original timecode, peak number and position bar, so a viewer can "
           "check each claim of the story against the film.",
           "The relationship between summary, peak point, time index and segment, which the assignment asks to demonstrate, "
           "is visible in the video itself, not only in the code.")
close("The recap runs **8 min 44 s** (524 s), under the 10-minute limit, with 5.5 minutes of original footage and narrated "
      "chapter cards for the 24 story sentences; it compresses the 67-minute film 7.7 times. The duration assertion passes.",
      "Snapping cuts to shot boundaries avoids starting a clip on a stray frame of the previous shot. More story sentences "
      "mean shorter clips: with 24 chapters the average clip is about 14 s (run E: 13 chapters, about 30 s), a trade-off "
      "between story completeness and how much of each scene is shown. The limit is enforced in code (planning budget plus "
      "an assertion after rendering), not left to chance.",
      "The movie recap is complete; the next component measures how good the summary is and whether the recap contains the "
      "intended content.")

# ============================================================================ 9 evaluation
head("9", "Evaluation: BLEU, recap fidelity and ablation",
     "Quantify the quality of the summary story and verify that the recap contains the intended scenes.",
     "Without measurement the pipeline's claims cannot be trusted; the assignment specifically asks for a BLEU comparison.",
     "* **BLEU** (Papineni et al., 2002): $\\text{BLEU} = BP \\cdot \\exp\\big(\\sum_{n=1}^{4} \\tfrac14 \\log p_n\\big)$, with clipped "
     "n-gram precision $p_n$ and brevity penalty $BP = \\min(1, e^{1 - r/c})$; implemented from first principles in "
     "`recap/bleu.py` and cross-checked against NLTK and sacreBLEU. ROUGE-1/2/L and embedding similarity are reported alongside.\n"
     "* **Reference:** the human-written Wikipedia plot (526 words). Four baselines of equal length: extractive TextRank, "
     "the first N words, uniformly spaced lines and random lines.\n"
     "* **Recap fidelity:** each clip is cut from the *finished recap* and transcribed again; its BLEU against the original "
     "transcript at the mapped timestamps tests the whole chain. The session suggested about 0.7 as a \"good enough\" BLEU.\n"
     "* **Ablation:** peaks chosen with text-only weights are compared with the multimodal fusion.\n\n"
     "The evaluation is algorithmic, so there is no prompt.")
implementation("The first cell runs a worked BLEU example through the three implementations. The second loads the stored "
               "evaluation (`evaluation.json`) produced by `recap/evaluate.py`.")
code("""
# Worked BLEU example: own implementation vs NLTK vs sacreBLEU on one sentence pair
from recap.bleu import bleu, tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

cand = "Al picks up Vera, a hitchhiker, who threatens to turn him over to the police."
ref = "Al crosses into California and picks up a hitchhiker named Vera. She blackmails Al by threatening to turn him over to the police."
own = bleu(cand, [ref])
nltk_score = corpus_bleu([[tokenize(ref)]], [tokenize(cand)], smoothing_function=SmoothingFunction().method2)
sacre_score = sacrebleu.sentence_bleu(cand, [ref]).score / 100
""")
code("""
# Stored evaluation results from the pipeline run
ev = load_json(OUT / "evaluation.json")
fid, ab = ev["recap_fidelity"], ev["ablation"]
metric_cols = ["bleu4", "bleu4_nltk", "sacrebleu", "bleu1", "rouge1", "rouge2", "rougeL", "semantic_similarity", "plot_coverage@0.5", "words"]
""")
output("Worked example, summary versus reference with baselines, recap fidelity and the ablation:")
code("""
print("n-gram precisions:", [round(p, 3) for p in own["precisions"]], "| brevity penalty:", round(own["bp"], 3))
print(f"BLEU own = {own['bleu']:.4f} | NLTK = {nltk_score:.4f} | sacreBLEU = {sacre_score:.4f}")
display(pd.DataFrame(ev["summary_vs_reference"]).T[metric_cols])
display(Image(filename=str(OUT / "metrics.png"), width=1100))
display(pd.DataFrame([{k: fid[k] for k in ("mean_clip_bleu", "median_clip_bleu", "peak_hit_rate@0.6", "overall_bleu", "mean_narration_bleu")}]))
display(Image(filename=str(OUT / "fidelity.png"), width=950))
display(pd.DataFrame({"text only (semantic + centrality)": ab["text_only_profile"],
                      "multimodal fusion (proposed)": ab["multimodal_profile"]}).T)
print(f"Peaks moved by adding audio, visual and emotion signals: {ab['peaks_changed']} of {len(pk['peaks'])} "
      f"(mean shift {ab['mean_shift_seconds']} s)")
""")
chart_note("In `metrics.png` the summary story has the tallest bar for every metric. In `fidelity.png` most clips score "
           "between 0.6 and 1.0, with a few low outliers (the lowest is clip 4 at 0.15).",
           "The baselines are close to each other and far below the story, so the gap comes from the summariser, not from "
           "length or chance. Low-fidelity clips are short scenes where music or overlapping speech changes what Whisper hears "
           "the second time.",
           "Both the summary (against the human plot) and the recap (against the original film) are measured, so quality is "
           "checked at both ends of the pipeline.")
close("* The hand-written BLEU matches NLTK exactly (0.0403); sacreBLEU (0.0353) differs slightly because of its own tokenizer.\n"
      "* The summary story beats all four baselines on every metric: BLEU-4 0.040 vs at most 0.021, BLEU-1 0.41 vs 0.32, "
      "ROUGE-L 0.19 vs 0.13, semantic similarity 0.81 vs 0.47, and plot coverage 0.56 vs 0.16.\n"
      "* Recap fidelity: mean clip BLEU 0.73 (median 0.77), above the 0.7 threshold suggested in class; 92 % of peak lines "
      "are heard in their clip, and the narration is recognised almost perfectly (0.97).\n"
      "* Fusion moves 13 of 24 peaks (mean shift 12 s) and raises the emotion intensity of the chosen lines from 0.50 to 0.77, "
      "at a cost in semantic match (0.87 to 0.75).",
      "Absolute BLEU is low for every system because BLEU counts exact word sequences and an independent human plot rarely "
      "shares phrasing. Writing two sentences per chunk raised the brevity penalty from 0.50 (run E) to 0.76 and BLEU-4 from "
      "0.030 to 0.040. Semantic similarity fell slightly (0.86 to 0.81) because the longer story includes more scene detail "
      "that the short reference does not mention. Fidelity is a little lower than run E (0.80) because 24 shorter clips leave "
      "less context around each peak line.",
      "The evaluation confirms both the quality of the story and the correctness of the time-index chain; the next section "
      "compares the summariser versions behind these numbers.")

# ============================================================================ 10 model comparison
head("10", "Model comparison (summarisation stage)",
     "Compare the candidate summarisers and justify the one used for the submission.",
     "The summariser has the largest influence on recap quality, so its choice must be backed by evidence.",
     "All variants run on the same transcript and are scored against the same reference. No model is trained (all are "
     "pre-trained and used zero-shot), so training time is not applicable; testing time is the measured CPU time to summarise "
     "the whole film. TextRank is timed live; LLM times come from the pipeline logs. The LLM variants use the same prompt "
     "shown in Section 6, so no new prompt is introduced here.")
implementation("The cell times the TextRank baseline live, loads the stored scores of the 3B run, and assembles the comparison table.")
code("""
# Time the extractive baseline live; LLM figures come from the logged pipeline runs
from recap.summarize import textrank_scores, extractive_summary

t0 = time.time()
cent = textrank_scores(segs, model)
_ = extractive_summary(segs, cent, target_words=len(summ["story"].split()))
textrank_seconds = time.time() - t0

ev3 = load_json(OUT.parent / "experiments" / "evaluation_qwen2.5-3b_grounded.json")["summary_vs_reference"]["Summary story (ours)"]
ev_e = load_json(OUT.parent / "experiments" / "evaluation_qwen2.5-7b_runE.json")["summary_vs_reference"]["Summary story (ours)"]
base = ev["summary_vs_reference"]
v2_minutes = load_json(OUT.parent / "experiments" / "timings_v2.json")["v2_full_llm_summary_minutes"]
ev_c = load_json(OUT.parent / "experiments" / "evaluation_qwen2.5-7b_v2c.json")["summary_vs_reference"]["Summary story (ours)"]
rows = [
    ["Qwen2.5-7B, notes + editor + key-event and entry checks (v2, proposed)", base["Summary story (ours)"], "none (pre-trained)",
     f"{v2_minutes:.1f} min", "Grounded in quoted evidence; role errors and repeats corrected; climax and ending kept",
     "Slowest variant (extra editor and repair calls)", "Offline recap generation where factual accuracy matters"],
    ["Qwen2.5-7B, v2c (same notes, earlier key-event prompt)", ev_c, "none (pre-trained)", f"{v2_minutes:.1f} min",
     "Highest plot coverage (0.64)", "Added sentences in first person and partly invented ('in a fit of rage')",
     "Not submitted: factual errors outweigh the coverage gain"],
    ["Qwen2.5-7B, rolling context (run E)", ev_e, "none (pre-trained)", "30.6 min",
     "Coherent arc, good semantic similarity", "Role errors (Vera named before she enters the film), repeated events, climax and ending missing",
     "Quick drafts when an editor pass is too slow"],
    ["Qwen2.5-3B, rolling context (run D)", ev3, "none (pre-trained)", "~14.5 min",
     "2x faster; higher BLEU-1 because it writes more words", "Repeats context sentences; invents events", "Low-RAM machines"],
    ["Extractive TextRank", base["Extractive TextRank"], "none", f"{textrank_seconds:.0f} s",
     "No hallucination; legally safest (original lines only)", "Dialogue lines do not tell a story", "Fallback when no LLM is available"],
    ["Uniform segments", base["Uniform segments"], "none", "< 1 s", "Trivial, unbiased coverage", "Ignores content", "Baseline only"],
    ["Random segments", base["Random segments"], "none", "< 1 s", "Trivial", "Ignores content and order", "Baseline only"],
]
comparison = pd.DataFrame([{"Model": r[0], "Performance (BLEU-4 / ROUGE-L / semantic / plot coverage)": f"{r[1]['bleu4']:.3f} / {r[1]['rougeL']:.3f} / {r[1]['semantic_similarity']:.3f} / {r[1]['plot_coverage@0.5']:.2f}",
                            "Training time": r[2], "Testing time": r[3], "Strengths": r[4], "Limitations": r[5],
                            "Suitable use cases": r[6]} for r in rows])
""")
output()
code("""
comparison
""")
close("The proposed v2 summariser has the best BLEU-4 (0.040), BLEU-1 and ROUGE-L of all variants and every baseline, "
      "and it is the only one whose story contains the correct climax. Run E keeps a slightly higher semantic similarity "
      "(0.86 vs 0.81) with a shorter story, and v2c reaches the highest plot coverage (0.64) but with invented details.",
      "No single metric picks the right model: BLEU rewards length and wording, semantic similarity rewards brevity, plot "
      "coverage rewards mentioning many events even with wrong details. The choice therefore weighs metrics together with "
      "factual checks of the story against the film. The 7B LLM costs about 33 minutes of CPU time for the summary, roughly "
      "twice the 3B model; TextRank stays the automatic fallback because it needs no LLM and cannot hallucinate.",
      "**Recommended model: Qwen2.5-7B-Instruct with grounded notes, editor pass, key-event check and character-entry "
      "check (v2)**, used for the submitted recap.")

# ============================================================================ conclusion & references
md("""
## 11. Conclusion

**Overall findings.** The pipeline turns a 67-minute YouTube film into an **8 min 44 s** recap (7.7× shorter, under the
10-minute limit) on a CPU with open models only. The summary story beats all four baselines on BLEU-4 (0.040), BLEU-1,
ROUGE and semantic similarity, the recap passes an end-to-end fidelity check (mean clip BLEU 0.73, 92 % of peak lines
heard), and 27 automated tests pass.

**Best-performing configuration.** Whisper small.en → Qwen2.5-7B with grounded notes, an editor pass and two deterministic
checks (key events, character entry) → monotonic DP alignment → five-signal post-fusion for peaks → OpenCV rendering with
shot-snapped cuts.

**Why it works.** Each stage carries the time index forward explicitly (chunk span → aligned window → peak line → clip →
recap position), so every sentence of the summary can be traced to a moment in the film, and the checks that matter most
(is a key event missing, is a character named too early) are rules on the transcript rather than model guesses.

**Error analysis of the final story (honest limitations).**

| Sentence | What the story says | What the film shows | Cause |
|---|---|---|---|
| 6 | Al throws a hitchhiker out of his car | Haskell tells Al about a hitchhiker who scratched him | Narrated anecdote taken as a current event |
| 10 | Al realises Vera is Haskell | Al realises Vera knew Haskell | Pronoun and role confusion |
| 12, 14 | Al suspects Vera of the killing | Vera accuses Al of killing Haskell | No speaker labels in the transcript |

The two deterministic checks removed the earlier errors of the same kind that they can detect (a name before its first
mention, a missing death); the remaining ones need speaker identity, which the transcript does not provide.

**Practical applications.** Film and series recaps, "previously on" segments for lectures, highlight reels of meetings and
interviews, and browsing public-domain archives.

**Future improvements.** Speaker diarisation (e.g. pyannote) so the model knows who speaks each line, which addresses all
remaining role errors; a vision-language model (CLIP) to match story sentences to frames; merging same-scene peaks into one
longer clip; a larger or fine-tuned LLM; a FastAPI service around the stages, as recommended in the Day 4 session.

## 12. References

1. A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022; SYSTRAN faster-whisper / CTranslate2.
2. K. Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation", ACL 2002.
3. C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries", 2004; C.-Y. Lin and F. J. Och, "ORANGE", COLING 2004 (BLEU smoothing).
4. R. Mihalcea and P. Tarau, "TextRank: Bringing Order into Texts", EMNLP 2004.
5. N. Reimers and I. Gurevych, "Sentence-BERT", EMNLP 2019 (`all-MiniLM-L6-v2`).
6. J. Hartmann, "Emotion English DistilRoBERTa-base", Hugging Face, 2022.
7. Qwen Team, "Qwen2.5 Technical Report", 2024; Ollama.
8. B. McFee et al., "librosa: Audio and Music Signal Analysis in Python", SciPy 2015; OpenCV; yt-dlp; NLTK; M. Post, "A Call for Clarity in Reporting BLEU Scores" (sacreBLEU), 2018.
9. Wikipedia, "Detour (1945 film)", Plot section (CC BY-SA 4.0), used as the reference summary.
10. *Detour* (1945), Producers Releasing Corporation, public domain; YouTube upload https://www.youtube.com/watch?v=tap67KjjPu8.
11. AI11 Module 8 Day 4 class notes, Dr. J. B. Simha (multimodal fusion, BLEU threshold, movie summarisation assignment).
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("Movie_Recap_Walkthrough.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
