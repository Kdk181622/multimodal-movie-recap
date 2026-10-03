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
