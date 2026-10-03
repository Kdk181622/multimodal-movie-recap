"""Stage 5: transcript -> summary story.

Two summarisers are provided:
* Abstractive "rolling-context" summarisation with a small local LLM served by Ollama
  (qwen2.5:3b, temperature 0). The transcript is cut into ~5 minute chunks; each chunk is
  summarised in 1-2 sentences while the story written so far is passed in as context, which keeps
  character roles consistent (the transcript has no speaker labels). The story is the ordered list
  of chunk sentences, and every sentence remembers the time span of the chunk it came from - the
  first, coarse level of the time index.
  A free-form map-reduce ("rewrite all chunk summaries as one story") was tried first and rejected:
  the 3B model ignored the length limit (46 sentences) and mixed up who did what.
* Extractive TextRank over sentence embeddings. It needs no LLM, is used as a fallback,
  and acts as a baseline in the evaluation.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

from .utils import fmt_ts, load_json, log, save_json

MAP_PROMPT = """You are writing a plot summary of the film "{title}" from its speech transcript.
There are no speaker labels. Much of the film is first-person voice-over narration by the main
character, so "I" in the narration usually means him.

Cast list from the credits, for spelling and roles only: {cast}
Name a character only if that name is spoken in this excerpt or already appears in the story so far;
otherwise describe the person ("a waitress", "a man"). Characters enter the film at different times.

Story so far (earlier parts of the film):
{context}

Now summarise the NEXT part of the film ({start} to {end}) in {n} plain sentences of at most
25 words each. Describe what happens: who does what and why it matters to the plot.
Third person, present tense. Only facts supported by the excerpt; if it is unclear who speaks,
describe the event without guessing. Do not repeat the story so far. No headings or lists.

Transcript excerpt:
\"\"\"{text}\"\"\"

Next sentences:"""


# ----------------------------------------------------------------------------- embeddings
@lru_cache(maxsize=2)
def get_embedder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """L2-normalised sentence embeddings, so a dot product equals cosine similarity."""
    return get_embedder(model_name).encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


# ----------------------------------------------------------------------------- helpers
def split_sentences(text: str) -> list[str]:
    """Light regex sentence splitter (avoids an NLTK data download)."""
    text = re.sub(r"\s+", " ", text.strip())
    # Protect titles so "Mr. Haskell" or "Charles Haskell Jr." are not split into fragments.
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if len(p.strip()) > 2]


def chunk_segments(segments: list[dict], chunk_seconds: float) -> list[dict]:
    """Group consecutive transcript segments into time chunks of about `chunk_seconds`."""
    chunks, cur = [], []
    for s in segments:
        if cur and s["end"] - cur[0]["start"] > chunk_seconds:
            chunks.append(cur)
            cur = []
        cur.append(s)
    if cur:
        # A short tail (e.g. the last 30 s of a film) is merged into the previous chunk.
        if chunks and cur[-1]["end"] - cur[0]["start"] < 0.4 * chunk_seconds:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return [{"index": i, "start": c[0]["start"], "end": c[-1]["end"],
             "segment_ids": [s["id"] for s in c], "text": " ".join(s["text"] for s in c)}
            for i, c in enumerate(chunks)]


def ollama_generate(prompt: str, scfg: dict, num_predict: int = 400) -> str:
    """Call the local Ollama REST API with deterministic decoding."""
    resp = requests.post(
        f"{scfg['ollama_url']}/api/generate",
        json={"model": scfg["ollama_model"], "prompt": prompt, "stream": False,
              "options": {"temperature": scfg.get("temperature", 0.0), "seed": scfg.get("seed", 42),
                          "num_predict": num_predict, "num_ctx": 4096}},
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def ollama_available(scfg: dict) -> bool:
    try:
        tags = requests.get(f"{scfg['ollama_url']}/api/tags", timeout=5).json()
        return any(m["name"].startswith(scfg["ollama_model"]) for m in tags.get("models", []))
    except Exception:
        return False


def _is_repeat(sentence: str, previous: list[str], model_name: str, threshold: float = 0.88) -> bool:
    """Small LLMs sometimes copy a sentence from the context back out; drop near-duplicates."""
    if not previous:
        return False
    emb = embed([sentence] + previous, model_name)
    return bool((emb[1:] @ emb[0]).max() >= threshold)


def _clean(text: str) -> str:
    text = re.sub(r"^(summary( story)?:?\s*)", "", text.strip(), flags=re.I)
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


# ----------------------------------------------------------------------------- TextRank
def textrank_scores(segments: list[dict], model_name: str, damping: float = 0.85, iters: int = 100) -> np.ndarray:
    """PageRank on the cosine-similarity graph of transcript segments (centrality in [0, 1])."""
    emb = embed([s["text"] for s in segments], model_name)
    sim = np.clip(emb @ emb.T, 0, None)
    np.fill_diagonal(sim, 0.0)
    # Very short utterances ("Yeah.", "What?") are similar to everything; down-weight them.
    lengths = np.array([len(s["text"].split()) for s in segments])
    sim *= np.minimum(1.0, lengths / 6.0)[None, :]
    row = sim.sum(axis=1, keepdims=True)
    trans = np.divide(sim, row, out=np.full_like(sim, 1.0 / len(sim)), where=row > 0)
    rank = np.full(len(sim), 1.0 / len(sim))
    for _ in range(iters):
        rank = (1 - damping) / len(sim) + damping * trans.T @ rank
    return (rank - rank.min()) / (rank.max() - rank.min() + 1e-12)


def extractive_summary(segments: list[dict], centrality: np.ndarray, target_words: int) -> dict:
    """Pick the most central segments (min 6 words) until the word budget is met, then reorder by time."""
    order = np.argsort(-centrality)
    chosen, words = [], 0
    for i in order:
        n = len(segments[i]["text"].split())
        if n < 6:
            continue
        chosen.append(int(i))
        words += n
        if words >= target_words:
            break
    chosen.sort()
    return {"segment_ids": chosen, "text": " ".join(segments[i]["text"] for i in chosen)}


# ----------------------------------------------------------------------------- stage runner
def summarize(cfg: dict, segments: list[dict]) -> dict:
    """Produce the summary story (cached to summary.json)."""
    out = Path(cfg["output_dir"]) / "summary.json"
    if out.exists():
        log.info("Using cached summary: %s", out.name)
        return load_json(out)

    scfg, title = cfg["summary"], cfg["input"]["title"]
    chunks = chunk_segments(segments, scfg["chunk_seconds"])
    centrality = textrank_scores(segments, cfg["peaks"]["embedding_model"])
    result: dict = {"title": title, "chunks": chunks, "centrality": centrality.round(4).tolist()}

    backend = scfg["backend"]
    if backend == "ollama" and not ollama_available(scfg):
        log.warning("Ollama model %s not reachable - falling back to extractive summary", scfg["ollama_model"])
        backend = "extractive"

    sentences: list[dict] = []
    if backend == "ollama":
        n = scfg.get("sentences_per_chunk", 2)
        for c in chunks:
            # The last few story sentences are the "story so far" context for the next chunk.
            context = " ".join(x["text"] for x in sentences[-6:]) or "(this is the beginning of the film)"
            # The final chunk holds the climax and the ending, so it may use one extra sentence.
            k = n + 1 if c["index"] == len(chunks) - 1 else n
            log.info("Chunk %d/%d (%s-%s)", c["index"] + 1, len(chunks), fmt_ts(c["start"]), fmt_ts(c["end"]))
            c["summary"] = _clean(ollama_generate(MAP_PROMPT.format(
                title=title, cast=cfg["input"].get("cast", "not provided"), context=context,
                start=fmt_ts(c["start"]), end=fmt_ts(c["end"]), n=f"1 to {k}" if k > 1 else "1",
                text=c["text"]), scfg, 200))
            for t in split_sentences(c["summary"])[:k]:
                if not _is_repeat(t, [x["text"] for x in sentences], cfg["peaks"]["embedding_model"]):
                    sentences.append({"text": t, "chunk": c["index"]})
        result["backend"] = f"ollama:{scfg['ollama_model']}"
    else:
        # Extractive fallback: the most central substantial line of every chunk, in film order.
        for c in chunks:
            best = max(c["segment_ids"], key=lambda i: (len(segments[i]["text"].split()) >= 6, centrality[i]))
            sentences.append({"text": segments[best]["text"], "chunk": c["index"]})
        result["backend"] = "extractive-textrank"

    result["story_sentences"] = [x["text"] for x in sentences]
    result["sentence_chunks"] = [x["chunk"] for x in sentences]
    result["story"] = " ".join(result["story_sentences"])
    # The extractive summary is always stored so it can be evaluated as a baseline.
    n_words = len(result["story"].split())
    result["extractive_baseline"] = extractive_summary(segments, centrality, target_words=n_words)
    save_json(result, out)
    log.info("Summary story: %d sentences, %d words (%s)", len(result["story_sentences"]), n_words, result["backend"])
    return result
