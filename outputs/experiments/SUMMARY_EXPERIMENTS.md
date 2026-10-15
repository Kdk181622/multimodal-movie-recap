# Summarisation experiments (model and prompt selection)

All runs use the same transcript (faster-whisper small.en, 946 segments), the same features and the
same peak/render/evaluation code. Only the summary stage changes. Reference = Wikipedia plot (526 words).
Metrics are taken from the pipeline logs / `evaluation.json` of each run.

| Run | LLM | Summary method | Story | BLEU-4 | BLEU-1 | ROUGE-L | Semantic sim. | Recap | Clip BLEU (fidelity) | Peak hit rate | Observed problems |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | qwen2.5:3b | Map (2-4 sentences / 5-min chunk) + free-form reduce to 12-18 sentences | 46 sentences, 752 words | - | - | - | - | - | - | - | Reduce step ignored the length limit; invented events (e.g. Haskell "covering the body"); sentence splitter broke on "Mr." Stopped before rendering. |
| B | qwen2.5:3b | Rolling context, 1-2 sentences per chunk | 26 sentences, 649 words | 0.0472 | 0.4410 | 0.1792 | 0.779 | 8:47 | 0.177* | 0.19* | Role confusion ("the narrator and his companion, Al"). *Fidelity was low because loud TTS narration masked the quiet 1945 soundtrack; fixed by loudness matching and per-clip transcription. |
| C | qwen2.5:3b | B + cast list from credits | 24 sentences, 366 words | 0.0497 | 0.3827 | 0.1839 | 0.848 | 9:02 | 0.703 | 1.00 | Names inserted where they do not belong (Vera placed in the opening club scene). Ending missing. |
| D | qwen2.5:3b | C + "name only if spoken" grounding, 3 sentences for the final chunk | 24 sentences, 361 words | 0.0399 | 0.3741 | 0.1911 | 0.812 | 8:40 | 0.740 | 0.96 | Copied sentences from its own context (chunk 5 repeated chunk 2); invented a hit-and-run. |
| **E (final)** | **qwen2.5:7b** | D + near-duplicate filter | **13 sentences, 315 words** | 0.0297 | 0.2894 | 0.1814 | **0.856** | **8:51** | **0.795** | **1.00** | Coherent full arc. Two role errors remain (sentence 4 names Vera where Haskell gives the ride; sentence 12 has Al, not Vera, threatening to call the police). |

Selection: **run E**. It has the highest semantic similarity to the human plot and the best recap
fidelity, and it is the only story a reader can follow from start to end. Its BLEU-4 is lower than C
largely because the story is shorter (brevity penalty 0.50); 57% of its words still appear in the
reference (unigram precision 0.574). BLEU rewards n-gram overlap and length, not factual order, so it was not used alone to
choose the model.

Cost: 7B summary took 30.6 min on CPU (13 chunks, about 2.3 min each) vs. about 15 min for 3B.

Saved artefacts: `summary_qwen2.5-3b_grounded.json` + `evaluation_qwen2.5-3b_grounded.json` (run D),
`summary_qwen2.5-7b.json` (run E), `run_qwen2.5-7b.log`, `run_final.log`.

## Version 2: grounding the story (15 Oct 2026)

Run E had three factual problems found on review against the Wikipedia plot: Vera is named in the scene where Haskell
gives Al a ride (her name is first spoken 14 minutes later), Al rather than Vera threatens to call the police, and the
climax (Vera's accidental death) and the ending are missing. Version 2 adds grounded notes, an editor pass and two
deterministic checks (`recap/summarize.py`). Same transcript, features, peak, render and evaluation code.

| Run | Change | Story | BLEU-4 | BLEU-1 | ROUGE-L | Semantic | Plot coverage | Recap | Clip BLEU | Observed problems |
|---|---|---|---|---|---|---|---|---|---|---|
| v2a | notes + editor (Python 3.10 without sacreBLEU) | 13 sentences, 125 words | - | - | - | - | - | - | - | Vague sentences; evaluation crashed (missing library). Saved as `summary_qwen2.5-7b_v2a.json` |
| v2b | + key-event and character-entry checks | 15 sentences, 213 words | 0.0195 | 0.158 | 0.189 | 0.851 | 0.52 | 8:41 | 0.786 | 4 of 13 chunks returned notes but no story; added sentences invented details ("in a fit of rage, on the roadside", "Al confesses") |
| v2c | + story-from-notes fallback, exactly 2 sentences per chunk, stricter key-event prompt | 24 sentences, 405 words | 0.0406 | 0.406 | 0.192 | 0.838 | 0.64 | 8:42 | 0.715 | Added sentences written in the first person ("I kill Vera accidentally") and one invented ("knocked him over the railing") |
| **Final** | + third-person/no-quote rules and a deterministic guard that rejects first-person or copied additions (checks re-run on the v2c notes and editor output) | **24 sentences, 410 words** | **0.0403** | **0.407** | **0.191** | 0.810 | 0.56 | **8:44** | **0.731** | 3 role errors remain (see below) |

Selection: **final**. It is the only version with a correct climax ("Al kills Vera accidentally, and now the police
suspect him of her murder"), no character named before entry, and no first-person or invented additions. v2c has a higher
plot coverage, but it comes from sentences with invented details, so it was rejected on factual review.

Remaining errors (no speaker labels in the transcript): sentence 6 treats Haskell's anecdote about a hitchhiker as Al's
own action; sentence 10 calls Vera "Haskell"; sentences 12 and 14 reverse who accuses whom of killing Haskell.

Cost: v2 summary 32.8 min of CPU time for the full LLM run (`run_v2c.log`, 16:56:55 to 17:29:45); re-running only the
checks on cached notes (`RECAP_REUSE_DRAFT`) takes about 4 minutes.
