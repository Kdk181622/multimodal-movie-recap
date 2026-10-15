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
  Version 2 (final) adds two grounding steps. Each chunk call first writes short event NOTES in the
  form "who - does what - to whom (quote)", then writes the story sentences from those notes only.
  A final EDITOR pass reads all draft sentences with all notes, corrects role errors, removes
  repeated events and restores missing key events (deaths, arrests, the ending). Run E (rolling
  context without notes or editor) is kept selectable with `summary.method: rolling` for ablation.
  A deterministic CHARACTER-ENTRY CHECK then runs on the final story: a sentence may not name a
  character before that character's name is first spoken in the transcript. Offending sentences are
  repaired by one targeted LLM call (REPAIR_PROMPT); this catches the run-E error where Vera was
  named in a scene that happens 20 minutes before she enters the film.
  A deterministic KEY-EVENT CHECK runs before it: transcript lines about deaths, killings, murder or
  the police are located in every chunk; if the story says nothing about them for that chunk, one
  targeted call (KEY_EVENT_PROMPT) writes the missing sentence from those quoted lines. This restores
  the climax (Vera's death) and the ending that both run E and the first v2 run left out.
* Extractive TextRank over sentence embeddings. It needs no LLM, is used as a fallback,
  and acts as a baseline in the evaluation.
"""

from __future__ import annotations

import os
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

NOTES_PROMPT = """You are writing a plot summary of the film "{title}" from its speech transcript.
There are no speaker labels. Much of the film is first-person voice-over narration by the main
character, so "I" in the narration usually means him.

Cast list from the credits, for spelling and roles only: {cast}
Name a character only if that name is spoken in this excerpt or already appears in the story so far;
otherwise describe the person ("a waitress", "a man"). Characters enter the film at different times.

Story so far (earlier parts of the film):
{context}

Transcript excerpt for the NEXT part of the film ({start} to {end}):
\"\"\"{text}\"\"\"

Work in two steps.

NOTES: list the 2 to 4 most important events of this part, one per line, in the form
"who - does what - to whom (quote)", where the quote is a few words from the excerpt that prove it.
Check who performs each action before writing it. A character telling a story about the past is not
doing it now, and a person who is only mentioned in someone's story is not present.
Deaths, crimes, arrests, threats and decisions that change the plot matter most.

STORY: then write {n} plain sentences of 15 to 25 words each, based only on your NOTES.
Be concrete: say who does what, where, and with what result. Avoid vague phrases such as
"grapples with his feelings" or "struggles with his past". Third person, present tense.
Do not repeat the story so far. No headings or lists.

NOTES:"""

STORY_PROMPT = """You are writing a plot summary of the film "{title}".
Event notes for the part of the film from {start} to {end}, each written as "who - does what - to whom (quote)":
{notes}

Write {n} plain sentences of 15 to 25 words each that tell these events as a story, in time order.
Be concrete: who does what, where, and with what result. Use only these notes.
Third person, present tense. No headings or lists.

Sentences:"""

REVISE_PROMPT = """You are the editor of a plot summary of the film "{title}".
Cast list from the credits, for spelling and roles only: {cast}

Draft story. Each line starts with the [part] of the film it comes from:
{draft}

Event notes written from the transcript of each part:
{notes}

Revise the draft into a clean story:
- Keep the time order and keep the [part] tag at the start of every line.
- If a sentence disagrees with the notes of its part (for example the wrong person doing an action), correct it.
- If two sentences tell the same event, keep only the better one.
- If an important event in the notes is missing (a death, a crime, an arrest, how the film ends), add one sentence for it with that part's tag.
- If a sentence is vague, replace it with the concrete events from the notes of its part.
- Keep the details: do not shorten sentences. The revised story should be about {words} words long, like the draft.
- Use only facts from the notes. 15 to 25 words per sentence, third person, present tense.
Output one sentence per line in the form "[part] sentence" and nothing else.

Revised story:
"""

KEY_EVENT_PROMPT = """You are completing a plot summary of the film "{title}".
Cast list from the credits, for spelling and roles only: {cast}

Story sentences already written for this part of the film ({start} to {end}):
{story}

These transcript lines from the same part show an important event (a death, a killing, a crime or
the police) that the story does not mention yet. There are no speaker labels; the main character
narrates in the first person, so "I" in narration means him.
{lines}

Write ONE sentence of 15 to 25 words that states this event plainly: who does what to whom and the
result. Rules:
- Use only what these lines and the story support. Do not add places, motives or emotions they do not state.
- Keep the nature of the event exactly as the lines give it: if they say it happened by accident or
  "without meaning to", say so.
- An accusation is not a confession: if one person accuses another, write it as an accusation.
- The narrator is Al Roberts. Write about him as "Al": never use "I", "me" or "my". Use character names.
- Do not quote the lines; describe what happens in your own words. Speech recognition may mishear a
  name (for example "Bill" for "Vera"); use the story and the cast list to tell who is meant.
Third person, present tense. Output only the sentence.

Sentence:"""

REPAIR_PROMPT = """You are correcting one sentence of a plot summary of the film "{title}".
The sentence names {names}, but at this point in the film ({start} to {end}) that character has not
appeared yet, so the sentence is wrong about who is involved.

Sentence to correct:
{sentence}

Transcript excerpt of this part of the film:
\"\"\"{text}\"\"\"

Rewrite the sentence so that it describes what really happens in this excerpt. Do not name {names}.
If a person's name is not spoken in the excerpt, describe the person instead ("a driver", "a man").
One sentence, at most 25 words, third person, present tense. Output only the sentence.

Corrected sentence:"""


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


def ollama_generate(prompt: str, scfg: dict, num_predict: int = 400, num_ctx: int = 4096) -> str:
    """Call the local Ollama REST API with deterministic decoding."""
    resp = requests.post(
        f"{scfg['ollama_url']}/api/generate",
        json={"model": scfg["ollama_model"], "prompt": prompt, "stream": False,
              "options": {"temperature": scfg.get("temperature", 0.0), "seed": scfg.get("seed", 42),
                          "num_predict": num_predict, "num_ctx": num_ctx}},
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


def parse_notes_story(text: str) -> tuple[list[str], str]:
    """Split a NOTES/STORY response into the event-note lines and the story text.

    If the model forgets the STORY marker, note-like lines (bullets, numbers, "who - does what")
    are treated as notes and everything else as the story, so a chunk is never lost.
    """
    text = text.replace("**", "").strip()
    parts = re.split(r"\bSTORY\s*:", text, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        note_block, story = parts
    else:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        is_note = [bool(re.match(r"^([-*\u2022]|\d+[.)])", l)) or " - " in l for l in lines]
        note_block = "\n".join(l for l, n in zip(lines, is_note) if n)
        story = " ".join(l for l, n in zip(lines, is_note) if not n)
    notes = [re.sub(r"^([-*\u2022]|\d+[.)])\s*", "", l.strip()) for l in note_block.splitlines()]
    notes = [n for n in notes if len(n) > 3 and not re.match(r"^notes\s*:?$", n, flags=re.I)]
    return notes, _clean(story)


def parse_revision(text: str, n_chunks: int, min_sentences: int) -> list[tuple[int, str]] | None:
    """Read "[part] sentence" lines from the editor; return None when the output is unusable.

    Part tags are 1-based in the prompt and converted back to 0-based chunk indices. Lines are
    kept in chunk order (a stable sort) so the time index stays monotonic.
    """
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*\[(\d+)\]\s*(.+?)\s*$", line.replace("**", ""))
        if m and 1 <= int(m.group(1)) <= n_chunks and len(m.group(2).split()) >= 4:
            out.append((int(m.group(1)) - 1, _clean(m.group(2))))
    if len(out) < min_sentences:
        return None
    return sorted(out, key=lambda x: x[0])


# ----------------------------------------------------------------------------- key-event check
# Strong evidence only: a killing, a murder, a strangling, "is/was dead" or "died". Negated forms such as
# "didn't kill" are ignored, and a chunk needs at least MIN_KEY_LINES hits, so casual talk ("dead tired",
# "the cops are no friends of mine") does not trigger the check.
KEY_LINE = re.compile(r"\b(killing|killed|kills|murder\w*|strangl\w*|(?:is|was) dead|died)\b", re.I)
NEGATED = re.compile(r"\b(didn't|did not|never|don't|wouldn't)\s+\w*\s*kill", re.I)
KEY_STORY = re.compile(r"\b(dead|death|dies|died|kill\w*|murder\w*|strangl\w*|police|cops?|arrest\w*|blackmail\w*|"
                       r"body|corpse|turn\w* (him|Al) in)\b", re.I)
MIN_KEY_LINES = 2
FIRST_PERSON = re.compile(r"\b(I|me|my|myself|we|us|our)\b")


def key_lines(chunk: dict, segments: list[dict], limit: int = 6) -> list[str]:
    """Transcript lines of one chunk with strong evidence of a death, killing or murder (with timestamps)."""
    ids = set(chunk["segment_ids"])
    hits = [s for s in segments if s["id"] in ids and KEY_LINE.search(s["text"]) and not NEGATED.search(s["text"])]
    return [f'[{fmt_ts(s["start"])}] {s["text"].strip()}' for s in hits[:limit]] if len(hits) >= MIN_KEY_LINES else []


def check_key_events(sentences: list[dict], chunks: list[dict], segments: list[dict], cfg: dict,
                     use_llm: bool) -> list[dict]:
    """Make sure every chunk whose dialogue contains key-event lines has a story sentence about them.

    Detection is deterministic (regular expressions on the transcript and on the story); only a chunk
    that fails the check gets one LLM call (KEY_EVENT_PROMPT). The new sentence is inserted after the
    chunk's existing sentences, so story order and the time index stay monotonic.
    """
    audit = []
    for c in chunks:
        lines = key_lines(c, segments)
        story = [x["text"] for x in sentences if x["chunk"] == c["index"]]
        if not lines or any(KEY_STORY.search(t) for t in story):
            continue
        row = {"chunk": c["index"], "span": f'{fmt_ts(c["start"])}-{fmt_ts(c["end"])}', "evidence": lines,
               "action": "flagged"}
        if use_llm:
            raw = ollama_generate(KEY_EVENT_PROMPT.format(
                title=cfg["input"]["title"], cast=cfg["input"].get("cast", ""), start=fmt_ts(c["start"]),
                end=fmt_ts(c["end"]), story=" ".join(story) or "(none)", lines="\n".join(lines)), cfg["summary"], 90)
            new = (split_sentences(_clean(raw)) or [_clean(raw)])[0]
            if FIRST_PERSON.search(new) or new.strip('"').lower() in " ".join(lines).lower():
                # Deterministic guard: a first-person or copied line is not a story sentence; reject it.
                row["rejected"], new = new, ""
            if len(new.split()) >= 5:
                # Place the sentence where its evidence occurs: before the chunk's other sentences if the first
                # evidence line falls in the first 60 % of the chunk, otherwise after them.
                t0 = min(s["start"] for s in segments if s["id"] in set(c["segment_ids"]) and KEY_LINE.search(s["text"]))
                early = (t0 - c["start"]) < 0.6 * (c["end"] - c["start"])
                same = [i for i, x in enumerate(sentences) if x["chunk"] == c["index"]]
                if early and same:
                    pos = same[0]
                else:
                    pos = max([i for i, x in enumerate(sentences) if x["chunk"] <= c["index"]], default=-1) + 1
                sentences.insert(pos, {"text": new, "chunk": c["index"]})
                row["action"], row["added"] = "added", new
        log.info("Key-event check: chunk %d (%s) -> %s", c["index"] + 1, row["span"], row["action"])
        audit.append(row)
    return audit


# ----------------------------------------------------------------------------- character-entry check
def cast_characters(cast: str) -> list[dict]:
    """Parse the cast string into characters with the name tokens that identify them.

    "Al Roberts (the narrator, ...); Vera (a hitchhiker)" -> [{"name": "Al Roberts", "tokens": ["Al", "Roberts"],
    "narrator": True}, ...]. The narrator is exempt from the check because he is present from the first scene.
    """
    chars = []
    for entry in [e.strip() for e in (cast or "").split(";") if e.strip()]:
        name, _, desc = entry.partition("(")
        tokens = [t for t in re.findall(r"[A-Z][a-z]+", name) if t not in {"Jr", "Sr", "Mr", "Mrs", "Ms", "Dr"}]
        if tokens:
            chars.append({"name": name.strip(), "tokens": tokens, "narrator": "narrator" in desc.lower()})
    return chars


def first_mentions(segments: list[dict], chars: list[dict]) -> dict[str, float]:
    """Time (s) at which any name token of each character is first spoken; inf if never spoken."""
    first = {}
    for ch in chars:
        pat = re.compile(r"\b(" + "|".join(map(re.escape, ch["tokens"])) + r")\b")
        first[ch["name"]] = next((s["start"] for s in segments if pat.search(s["text"])), float("inf"))
    return first


def entry_violations(sentence: str, chunk_end: float, chars: list[dict], first: dict[str, float]) -> list[str]:
    """Characters named in `sentence` whose name is first spoken after the end of the sentence's chunk."""
    bad = []
    for ch in chars:
        if ch["narrator"] or first[ch["name"]] <= chunk_end:
            continue
        if re.search(r"\b(" + "|".join(map(re.escape, ch["tokens"])) + r")\b", sentence):
            bad.append(ch["name"])
    return bad


def check_and_repair_entries(sentences: list[dict], chunks: list[dict], segments: list[dict], cfg: dict,
                             use_llm: bool) -> list[dict]:
    """Apply the character-entry check to the final story; repair violations with REPAIR_PROMPT.

    Returns an audit list (one row per checked violation) that is stored in summary.json. If the repair
    still names the character, the sentence is dropped when its chunk has another sentence, otherwise kept
    and flagged, so the time index never loses a part of the film.
    """
    chars = cast_characters(cfg["input"].get("cast", ""))
    first = first_mentions(segments, chars)
    audit = []
    for i, x in enumerate(sentences):
        c = chunks[x["chunk"]]
        bad = entry_violations(x["text"], c["end"], chars, first)
        if not bad:
            continue
        row = {"sentence_index": i, "chunk": x["chunk"], "original": x["text"], "names": bad,
               "first_spoken": {b: (fmt_ts(first[b]) if first[b] != float("inf") else "never") for b in bad},
               "action": "flagged"}
        if use_llm:
            names = " or ".join(bad)
            fixed = _clean(ollama_generate(REPAIR_PROMPT.format(
                title=cfg["input"]["title"], names=names, start=fmt_ts(c["start"]), end=fmt_ts(c["end"]),
                sentence=x["text"], text=c["text"]), cfg["summary"], 80))
            fixed = (split_sentences(fixed) or [fixed])[0]
            if fixed and not entry_violations(fixed, c["end"], chars, first):
                x["text"], row["action"], row["repaired"] = fixed, "repaired", fixed
        if row["action"] == "flagged" and sum(1 for y in sentences if y["chunk"] == x["chunk"]) > 1:
            x["drop"], row["action"] = True, "dropped"
        log.info("Entry check: sentence %d names %s before entry -> %s", i + 1, bad, row["action"])
        audit.append(row)
    sentences[:] = [x for x in sentences if not x.get("drop")]
    return audit


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
    # Optional: reuse the chunk notes, draft and editor output of an earlier run (env RECAP_REUSE_DRAFT=<summary.json>)
    # so that a change to the deterministic checks can be re-run in minutes instead of 30 min of LLM time.
    reuse = os.environ.get("RECAP_REUSE_DRAFT", "")
    if backend == "ollama" and reuse and Path(reuse).exists():
        prev = load_json(reuse)
        chunks = prev["chunks"]
        result.update({"chunks": chunks, "draft_sentences": prev["draft_sentences"], "draft_chunks": prev["draft_chunks"],
                       "revision_raw": prev.get("revision_raw", ""), "backend": prev["backend"], "reused_draft_from": reuse})
        revised = parse_revision(prev.get("revision_raw", ""), len(chunks), max(6, len(prev["draft_sentences"]) // 2))
        sentences = ([{"text": t, "chunk": k} for k, t in revised] if revised else
                     [{"text": t, "chunk": k} for t, k in zip(prev["draft_sentences"], prev["draft_chunks"])])
        result["method"] = "notes+revise" if revised else "notes"
        log.info("Reusing chunk notes, draft and editor output from %s", reuse)
        if scfg.get("key_event_check", True):
            result["key_event_check"] = check_key_events(sentences, chunks, segments, cfg, use_llm=True)
        if scfg.get("entry_check", True):
            result["entry_check"] = check_and_repair_entries(sentences, chunks, segments, cfg, use_llm=True)
        backend = "reused"
    if backend == "ollama":
        n = scfg.get("sentences_per_chunk", 2)
        method = scfg.get("method", "notes")
        cast = cfg["input"].get("cast", "not provided")
        emb_model = cfg["peaks"]["embedding_model"]
        for c in chunks:
            # The last few story sentences are the "story so far" context for the next chunk.
            context = " ".join(x["text"] for x in sentences[-6:]) or "(this is the beginning of the film)"
            # Exactly k sentences per chunk keeps the story close to the reference length (BLEU brevity penalty);
            # the final chunk holds the climax and the ending, so it gets one extra sentence.
            k = n + 1 if c["index"] == len(chunks) - 1 else n
            fill = dict(title=title, cast=cast, context=context, start=fmt_ts(c["start"]), end=fmt_ts(c["end"]),
                        n=str(k), text=c["text"])
            log.info("Chunk %d/%d (%s-%s) [%s]", c["index"] + 1, len(chunks), fmt_ts(c["start"]), fmt_ts(c["end"]), method)
            if method == "notes":
                c["notes"], c["summary"] = parse_notes_story(ollama_generate(NOTES_PROMPT.format(**fill), scfg, 520))
                if not c["summary"] and c["notes"]:
                    # The model sometimes stops after the notes; write the story from the notes in one short call.
                    c["story_from_notes"] = True
                    c["summary"] = _clean(ollama_generate(STORY_PROMPT.format(
                        title=title, start=fill["start"], end=fill["end"], n=fill["n"],
                        notes="\n".join(f"- {x}" for x in c["notes"])), scfg, 160))
            else:
                c["summary"] = _clean(ollama_generate(MAP_PROMPT.format(**fill), scfg, 200))
            for t in split_sentences(c["summary"])[:k]:
                if not _is_repeat(t, [x["text"] for x in sentences], emb_model):
                    sentences.append({"text": t, "chunk": c["index"]})
        result["backend"] = f"ollama:{scfg['ollama_model']}"
        result["method"] = method

        # Editor pass: check the whole draft against the notes of every chunk.
        if method == "notes" and scfg.get("revise", True):
            result["draft_sentences"] = [x["text"] for x in sentences]
            result["draft_chunks"] = [x["chunk"] for x in sentences]
            draft = "\n".join(f"[{x['chunk'] + 1}] {x['text']}" for x in sentences)
            notes = "\n".join(f"[{c['index'] + 1}] " + "; ".join(c.get("notes", [])) for c in chunks)
            log.info("Editor pass over %d draft sentences", len(sentences))
            n_words = sum(len(x["text"].split()) for x in sentences)
            raw = ollama_generate(REVISE_PROMPT.format(title=title, cast=cast, draft=draft, notes=notes, words=n_words),
                                  scfg, 900, num_ctx=8192)
            result["revision_raw"] = raw
            revised = parse_revision(raw, len(chunks), max(6, len(sentences) // 2))
            if revised:
                sentences = [{"text": t, "chunk": k} for k, t in revised]
                result["method"] = "notes+revise"
            else:
                log.warning("Editor output unusable - keeping the draft story")
        # Deterministic key-event check: deaths, crimes and the police must not be missing from the story.
        if scfg.get("key_event_check", True):
            result["key_event_check"] = check_key_events(sentences, chunks, segments, cfg, use_llm=True)
        # Deterministic character-entry check on the final story, with a targeted LLM repair.
        if scfg.get("entry_check", True):
            result["entry_check"] = check_and_repair_entries(sentences, chunks, segments, cfg, use_llm=True)
    elif backend != "reused":
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
