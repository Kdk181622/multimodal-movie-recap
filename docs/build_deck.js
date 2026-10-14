// Builds docs/Movie_Recap_Presentation.pptx from the pipeline outputs.
// Usage (from the project root): NODE_PATH=<dir with pptxgenjs> node docs/build_deck.js <apply_theme.js path>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "outputs", "detour_1945");
const A = (f) => path.join(ROOT, "docs", "slide_assets", f);
const notes = JSON.parse(fs.readFileSync(path.join(__dirname, "speaker_notes.json"), "utf8"));
const ev = JSON.parse(fs.readFileSync(path.join(OUT, "evaluation.json"), "utf8"));
const summ = JSON.parse(fs.readFileSync(path.join(OUT, "summary.json"), "utf8"));
const timings = JSON.parse(fs.readFileSync(path.join(OUT, "..", "experiments", "timings_reference.json"), "utf8"));

const THEME = {
  name: "Film Noir Amber",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "141414", lt1: "FFFFFF", dk2: "2B2B2B", lt2: "F2F1EE",
    accent1: "E8A33D", accent2: "2A78D6", accent3: "1BAF7A", accent4: "EB6834",
    accent5: "8A8984", accent6: "4A3AA7", hlink: "2A78D6", folHlink: "4A3AA7",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Karthikeyan K";
pres.title = "Multimodal Movie Recap";
pres.subject = "AI11 Module 8 Assignment 1";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.35, w: 5.6, h: 1.5, fontSize: 38, bold: true,
      color: C.background1, valign: "bottom", align: "left", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.95, w: 5.6, h: 0.9, fontSize: 16,
      color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9.0, h: 0.75, fontSize: 28, bold: true,
      color: C.text1, valign: "middle", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
    { text: { text: "Multimodal Movie Recap  |  AI11 Module 8", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: HEX.accent5, margin: 0 } } },
  ],
});
pres.defineSlideMaster({
  title: "DARK_END",
  background: { color: HEX.dk1 },
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, bold: true,
      color: C.background1, valign: "middle", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
  ],
});

// ---------------------------------------------------------------- helpers
let slideNo = 0;
function addSlide(master, section) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  s.addNotes(notes[slideNo].script);
  slideNo += 1;
  return s;
}
const body = (extra = {}) => Object.assign({ fontSize: 14, color: C.text2, margin: 0, isTextBox: true, valign: "top" }, extra);
// Motif: amber numbered disc.
function badge(s, x, y, label, d = 0.42) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `badge ${label}` });
  s.addText(String(label), { x, y, w: d, h: d, fontSize: 13, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
}
function stat(s, x, y, w, value, label, dark = false, size = 36) {
  s.addText(value, { x, y, w, h: 0.75, fontSize: size, bold: true, fontFace: THEME.headFontFace, color: dark ? C.accent1 : C.text1, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.75, w, h: 0.45, fontSize: 12, color: dark ? C.background2 : C.accent5, margin: 0, isTextBox: true, valign: "top" });
}
const fmt = (x, d = 3) => Number(x).toFixed(d);
const sv = ev.summary_vs_reference;
const fid = ev.recap_fidelity;
const ab = ev.ablation;
const cons = ev.constraints;
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// ---------------------------------------------------------------- 1 title
pres.addSection({ title: "Introduction" });
let s = addSlide("DARK_TITLE", "Introduction");
s.addText("Multimodal Movie Recap", { placeholder: "title" });
s.addText("YouTube film → summary story → peak points → time index → a recap under 10 minutes", { placeholder: "body" });
s.addText("AI11 · Module 8 · Multimodal AI Systems: Speech, Text and Conversational Agents · Assignment 1 · Dr. J. B. Simha",
  { x: 0.6, y: 4.35, w: 5.6, h: 0.6, fontSize: 11, color: C.accent5, margin: 0, isTextBox: true });
s.addShape(pres.shapes.OVAL, { x: 6.85, y: 0.95, w: 2.5, h: 2.5, fill: { color: C.accent1 }, line: { color: C.accent1 } });
s.addImage({ path: A("presenter_photo.jpg"), x: 6.93, y: 1.03, w: 2.34, h: 2.34, rounding: true, altText: "Photo of the presenter, Karthikeyan K" });
s.addText("Karthikeyan K", { x: 6.35, y: 3.6, w: 3.5, h: 0.5, fontSize: 20, bold: true, fontFace: THEME.headFontFace, color: C.background1, align: "center", margin: 0, isTextBox: true });
s.addText("Presenter", { x: 6.35, y: 4.08, w: 3.5, h: 0.3, fontSize: 12, color: C.accent1, align: "center", margin: 0, isTextBox: true });

// ---------------------------------------------------------------- 2 problem
s = addSlide("CONTENT", "Introduction");
s.addText("Problem statement and objectives", { placeholder: "title" });
card(s, 0.5, 1.2, 4.1, 3.7);
s.addText("The problem", body({ x: 0.75, y: 1.4, w: 3.6, h: 0.4, fontSize: 18, bold: true, color: C.text1, fontFace: THEME.headFontFace }));
s.addText("A feature film takes over an hour to watch. Producing a recap by hand means watching it, writing a summary and editing clips together. The goal is an AI system that does all of this automatically from a YouTube link.",
  body({ x: 0.75, y: 1.9, w: 3.6, h: 2.8, fontSize: 15 }));
const objectives = [
  "Process video, audio and text in one multimodal pipeline",
  "Speech-to-text + summarisation → a concise summary story",
  "Find peak points and map them to time indices in the film",
  "Extract the segments (OpenCV) → recap under 10 min, scored with BLEU",
];
objectives.forEach((t, i) => {
  badge(s, 5.0, 1.3 + i * 0.9, i + 1);
  s.addText(t, body({ x: 5.6, y: 1.3 + i * 0.9, w: 3.9, h: 0.75, fontSize: 15, valign: "middle" }));
});

// ---------------------------------------------------------------- 3 input film
s = addSlide("CONTENT", "Introduction");
s.addText("Input: Detour (1945), a public-domain film noir", { placeholder: "title" });
s.addImage({ path: A("recap_intro.jpg"), x: 0.5, y: 1.2, w: 4.2, h: 3.15, altText: "Intro card of the generated recap" });
const reasons = [
  ["Public domain", "recap clips can be shared with no copyright risk"],
  ["Voice-over narration", "rich, clear speech for speech-to-text"],
  ["Human reference", "Wikipedia plot summary for BLEU"],
];
reasons.forEach(([h, t], i) => {
  s.addText([{ text: h + ": ", options: { bold: true, color: C.text1 } }, { text: t }], body({ x: 5.0, y: 1.25 + i * 0.55, w: 4.5, h: 0.5, fontSize: 14 }));
});
stat(s, 5.0, 3.05, 1.2, "67:38", "film length", false, 28);
stat(s, 6.2, 3.05, 1.1, "946", "transcript lines", false, 28);
stat(s, 7.3, 3.05, 1.0, String(cons.n_peaks), "peak points", false, 28);
stat(s, 8.3, 3.05, 1.2, mmss(cons.recap_duration_seconds), "recap length", false, 28);
s.addText("youtube.com/watch?v=tap67KjjPu8  ·  original English audio track selected (upload also carries auto-dubbed tracks)",
  body({ x: 0.5, y: 4.5, w: 9.0, h: 0.4, fontSize: 11, color: C.accent5 }));

// ---------------------------------------------------------------- 4 architecture
pres.addSection({ title: "Solution" });
s = addSlide("CONTENT", "Solution");
s.addText("System architecture and workflow", { placeholder: "title" });
s.addImage({ path: path.join(ROOT, "docs", "architecture.png"), x: 0.5, y: 1.15, w: 9.0, h: 2.7, altText: "Pipeline: YouTube URL to evaluation in ten stages" });
s.addText([
  { text: "Post-fusion: ", options: { bold: true, color: C.text1 } },
  { text: "audio, video and text-emotion signals are computed independently and combined only when choosing peak points. Every stage is a separate, tested module that caches its output." },
], body({ x: 0.5, y: 4.1, w: 9.0, h: 0.8, fontSize: 14 }));

// ---------------------------------------------------------------- 5 technologies
s = addSlide("CONTENT", "Solution");
s.addText("Technologies, models and libraries", { placeholder: "title" });
const tech = [
  ["Ingest", "yt-dlp · ffmpeg (bundled via imageio-ffmpeg)"],
  ["Speech-to-text", "faster-whisper small.en, int8 on CPU, word timestamps"],
  ["Summary story", "Qwen2.5-7B-Instruct via Ollama, temperature 0"],
  ["Embeddings", "sentence-transformers all-MiniLM-L6-v2"],
  ["Text emotion", "DistilRoBERTa emotion classifier"],
  ["Audio features", "librosa: RMS, onset strength, ZCR"],
  ["Video", "OpenCV: shot cuts, motion, seek / read / write"],
  ["Evaluation", "own BLEU + NLTK + sacreBLEU, ROUGE"],
];
tech.forEach(([h, t], i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = 0.5 + col * 4.6, y = 1.2 + row * 0.93;
  card(s, x, y, 4.4, 0.8);
  s.addText(h, body({ x: x + 0.2, y: y + 0.08, w: 4.0, h: 0.32, fontSize: 14, bold: true, color: C.text1 }));
  s.addText(t, body({ x: x + 0.2, y: y + 0.4, w: 4.0, h: 0.34, fontSize: 12 }));
});

// ---------------------------------------------------------------- 6 implementation
s = addSlide("CONTENT", "Solution");
s.addText("Implementation: test, cache, then integrate", { placeholder: "title" });
const steps = [
  ["Test each component", "23 automated tests; a synthetic movie with a known cut (4 s) and loud burst (5 s)"],
  ["Cache slow stages", "transcript, features, summary and peaks saved as JSON"],
  ["One config file", "models, fusion weights and the duration budget in config.yaml"],
];
steps.forEach(([h, t], i) => {
  badge(s, 0.5, 1.25 + i * 1.15, i + 1);
  s.addText(h, body({ x: 1.1, y: 1.22 + i * 1.15, w: 3.4, h: 0.35, fontSize: 15, bold: true, color: C.text1 }));
  s.addText(t, body({ x: 1.1, y: 1.57 + i * 1.15, w: 3.4, h: 0.7, fontSize: 13 }));
});
s.addChart(pres.charts.BAR, [{ name: "minutes", labels: timings.labels, values: timings.minutes }], {
  x: 4.9, y: 1.1, w: 4.6, h: 3.8, barDir: "bar", chartColors: [HEX.accent1],
  showTitle: true, title: "Measured CPU time per stage (minutes)", titleFontSize: 12, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: HEX.dk2, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
});

// ---------------------------------------------------------------- 7 signals
s = addSlide("CONTENT", "Solution");
s.addText("Speech-to-text and multimodal signals", { placeholder: "title" });
s.addImage({ path: path.join(OUT, "timeline.png"), x: 0.5, y: 1.1, w: 5.5, h: 4.02, altText: "Audio, visual, emotion and fused curves across the film with peak points" });
const sig = [
  ["Transcript", "946 lines, 9k words, word timestamps; 22 min on CPU"],
  ["Audio", "librosa arousal: loudness, onsets, amplitude fluctuation"],
  ["Video", "OpenCV: 312 shot cuts + frame-difference motion"],
  ["Text emotion", "48% of lines non-neutral (surprise, disgust, anger lead)"],
];
sig.forEach(([h, t], i) => {
  s.addText(h, body({ x: 6.3, y: 1.15 + i * 0.95, w: 3.2, h: 0.32, fontSize: 14, bold: true, color: C.text1 }));
  s.addText(t, body({ x: 6.3, y: 1.47 + i * 0.95, w: 3.2, h: 0.6, fontSize: 12 }));
});

// ---------------------------------------------------------------- 8 summary
pres.addSection({ title: "Method" });
s = addSlide("CONTENT", "Method");
s.addText("Summary story: rolling-context summarisation", { placeholder: "title" });
const sm = [
  "13 chunks of ~5 min; 1-2 sentences each",
  "Story so far passed as context (no speaker labels in the transcript)",
  "Cast list from the credits; a name is used only if spoken",
  "Near-duplicate sentences filtered out",
];
sm.forEach((t, i) => {
  badge(s, 0.5, 1.25 + i * 0.75, i + 1, 0.38);
  s.addText(t, body({ x: 1.05, y: 1.25 + i * 0.75, w: 3.3, h: 0.65, fontSize: 13, valign: "middle" }));
});
card(s, 4.6, 1.15, 4.9, 3.8);
s.addText("From the generated story (13 sentences, 315 words)", body({ x: 4.8, y: 1.28, w: 4.5, h: 0.3, fontSize: 12, bold: true, color: C.accent5 }));
const pick = [1, 4, 6, 12];
const chunks = summ.chunks;
const ts = (sec) => `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
pick.forEach((k, i) => {
  const c = chunks[summ.sentence_chunks[k]];
  s.addText([
    { text: `${ts(c.start)}-${ts(c.end)}  `, options: { bold: true, color: C.text1 } },
    { text: summ.story_sentences[k] },
  ], body({ x: 4.8, y: 1.65 + i * 0.82, w: 4.5, h: 0.8, fontSize: 11.5 }));
});

// ---------------------------------------------------------------- 9 peaks
s = addSlide("CONTENT", "Method");
s.addText("Peak-point identification: alignment + post-fusion", { placeholder: "title" });
s.addImage({ path: A("alignment_heatmap.png"), x: 0.5, y: 1.1, w: 5.6, h: 2.24, altText: "Story sentence by transcript window similarity with the alignment path" });
s.addText([
  { text: "1  Alignment. ", options: { bold: true, color: C.text1 } },
  { text: "Monotonic dynamic programming keeps story order, inside each sentence's own chunk." },
], body({ x: 0.5, y: 3.5, w: 5.6, h: 0.6, fontSize: 13 }));
s.addText([
  { text: "2  Peak picking. ", options: { bold: true, color: C.text1 } },
  { text: "Within ±45 s, the highest fused score wins; its Whisper timestamp is the time index." },
], body({ x: 0.5, y: 4.15, w: 5.6, h: 0.6, fontSize: 13 }));
s.addChart(pres.charts.BAR, [{ name: "weight", labels: ["Semantic", "Centrality", "Text emotion", "Audio", "Visual"], values: [0.40, 0.15, 0.15, 0.15, 0.15] }], {
  x: 6.4, y: 1.1, w: 3.1, h: 3.7, barDir: "bar", chartColors: [HEX.accent1],
  showTitle: true, title: "Fusion weights", titleFontSize: 12, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: HEX.dk2, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valAxisMaxVal: 0.5, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
});

// ---------------------------------------------------------------- 10 mapping
s = addSlide("CONTENT", "Method");
s.addText("Timestamp mapping and segment extraction", { placeholder: "title" });
const chain = [["Story sentence 5", "\"Al realizes he must hide Haskell's body…\""], ["Source chunk", "22:10-27:02"], ["Peak line", "\"…fell out of the car…\""],
  ["Time index", "00:24:13"], ["Film clip", "23:55-24:28"], ["Recap slot", "OpenCV seek / read / write"]];
chain.forEach(([h, t], i) => {
  const x = 0.5 + i * 1.53;
  card(s, x, 1.15, 1.38, 1.15, i === 3 ? C.accent1 : C.background2);
  s.addText(h, body({ x: x + 0.1, y: 1.22, w: 1.18, h: 0.3, fontSize: 11, bold: true, color: C.text1 }));
  s.addText(t, body({ x: x + 0.1, y: 1.52, w: 1.18, h: 0.75, fontSize: 10.5 }));
  if (i < chain.length - 1) s.addText("›", { x: x + 1.36, y: 1.5, w: 0.2, h: 0.4, fontSize: 18, bold: true, color: C.accent5, margin: 0, align: "center", isTextBox: true });
});
s.addImage({ path: A("peak_keyframes.jpg"), x: 0.5, y: 2.5, w: 6.4, h: 2.4, altText: "Frames at the first eight peak points with their time indices" });
s.addText("Clips grow with on-topic neighbouring dialogue (≤ 32 s), snap to the nearest shot cut, stay chronological and fit the budget.",
  body({ x: 7.1, y: 2.55, w: 2.4, h: 2.3, fontSize: 13 }));

// ---------------------------------------------------------------- 11 recap
pres.addSection({ title: "Results" });
s = addSlide("CONTENT", "Results");
s.addText("Final movie recap output", { placeholder: "title" });
s.addImage({ path: A("recap_card.jpg"), x: 0.5, y: 1.1, w: 3.6, h: 2.7, altText: "Chapter card narrating a story sentence" });
// Embedded 42 s excerpt (chapter card 5 + clip 5) so the recap plays inside the slide show while recording.
s.addMedia({ type: "video", path: A("recap_excerpt.mp4"), x: 4.25, y: 1.1, w: 3.6, h: 2.7,
  cover: "image/jpeg;base64," + fs.readFileSync(A("recap_clip.jpg")).toString("base64") });
s.addText("Chapter card (narrated story sentence)", body({ x: 0.5, y: 3.85, w: 3.6, h: 0.3, fontSize: 11, color: C.accent5 }));
s.addText("▶ Click to play: 42 s excerpt (chapter 5 + its clip)", body({ x: 4.25, y: 3.85, w: 3.6, h: 0.3, fontSize: 11, color: C.accent5 }));
stat(s, 8.1, 1.05, 1.5, mmss(cons.recap_duration_seconds), "recap (< 10:00)");
stat(s, 8.1, 2.3, 1.5, (cons.footage_seconds / 60).toFixed(1), "minutes of original footage");
stat(s, 8.1, 3.55, 1.5, `${cons.compression_ratio.toFixed(1)}×`, "compression");
s.addText("outputs/detour_1945/movie_recap.mp4  ·  movie_recap.srt  ·  recap_edl.csv", body({ x: 0.5, y: 4.4, w: 7.4, h: 0.35, fontSize: 11, color: C.accent5 }));

// ---------------------------------------------------------------- 12 BLEU
s = addSlide("CONTENT", "Results");
s.addText("Evaluation: summary vs. the human plot (BLEU)", { placeholder: "title" });
const names = Object.keys(sv);
const short = { "Summary story (ours)": "Our story", "Extractive TextRank": "TextRank", "Lead-N (first words)": "Lead-N", "Uniform segments": "Uniform", "Random segments": "Random" };
function twoSeries(metric) {
  const labels = names.map((n) => `${short[n]} (${fmt(sv[n][metric], metric === "bleu4" ? 3 : 2)})`);
  return [
    { name: "Proposed", labels, values: names.map((n) => (n.includes("ours") ? sv[n][metric] : 0)) },
    { name: "Baselines", labels, values: names.map((n) => (n.includes("ours") ? 0 : sv[n][metric])) },
  ];
}
const chartOpts = (title, x) => ({
  x, y: 1.1, w: 4.4, h: 2.9, barDir: "bar", barGrouping: "stacked", chartColors: [HEX.accent1, "C9C8C2"],
  showTitle: true, title, titleFontSize: 12, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
});
s.addChart(pres.charts.BAR, twoSeries("bleu4"), chartOpts("BLEU-4 (higher is better)", 0.5));
s.addChart(pres.charts.BAR, twoSeries("semantic_similarity"), chartOpts("Semantic similarity to the reference", 5.1));
s.addText([
  { text: "Why BLEU is low for everyone: ", options: { bold: true, color: C.text1 } },
  { text: `BLEU counts exact 4-word overlaps; independent summaries rarely share phrasing, and the story (${sv["Summary story (ours)"].words} words) is shorter than the 526-word reference (brevity penalty ${fmt(sv["Summary story (ours)"].brevity_penalty, 2)}). The ranking against baselines is the meaningful signal.` },
], body({ x: 0.5, y: 4.1, w: 9.0, h: 0.85, fontSize: 12.5 }));

// ---------------------------------------------------------------- 13 fidelity + ablation
s = addSlide("CONTENT", "Results");
s.addText("Recap fidelity and multimodal ablation", { placeholder: "title" });
card(s, 0.5, 1.15, 4.3, 3.75);
s.addText("Recap re-transcribed clip by clip", body({ x: 0.75, y: 1.3, w: 3.9, h: 0.35, fontSize: 14, bold: true, color: C.text1 }));
stat(s, 0.75, 1.8, 1.9, fmt(fid.mean_clip_bleu, 2), "mean clip BLEU (> 0.7 target)");
stat(s, 2.75, 1.8, 1.9, `${Math.round(fid["peak_hit_rate@0.6"] * 100)}%`, "peak lines heard in their clip");
stat(s, 0.75, 3.1, 1.9, fmt(fid.mean_narration_bleu, 2), "narration intelligibility");
stat(s, 2.75, 3.1, 1.9, fmt(fid.median_clip_bleu, 2), "median clip BLEU");
s.addText("Text-only vs. multimodal peak selection", body({ x: 5.1, y: 1.3, w: 4.4, h: 0.35, fontSize: 14, bold: true, color: C.text1 }));
const tp = ab.text_only_profile, mp = ab.multimodal_profile;
const rowsAb = [["Signal of chosen peaks", "Text only", "Fusion"]].concat(
  [["semantic", "Semantic"], ["text_emotion", "Text emotion"], ["audio", "Audio arousal"], ["visual", "Visual activity"]]
    .map(([k, l]) => [l, fmt(tp[k], 2), fmt(mp[k], 2)]));
s.addTable(rowsAb.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: HEX.dk2, fill: { color: i === 0 ? HEX.lt2 : HEX.lt1 }, align: j ? "center" : "left" } }))),
  { x: 5.1, y: 1.8, w: 4.4, colW: [2.0, 1.2, 1.2], fontSize: 12, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D9D8D3" }, rowH: 0.36 });
s.addText(`Fusion moved ${ab.peaks_changed} of ${cons.n_peaks} peaks (mean ${ab.mean_shift_seconds} s) toward the more dramatic line in the same scene.`,
  body({ x: 5.1, y: 3.85, w: 4.4, h: 0.9, fontSize: 13 }));

// ---------------------------------------------------------------- 14 model comparison
s = addSlide("CONTENT", "Results");
s.addText("Model comparison: choosing the summariser", { placeholder: "title" });
const head = ["Configuration", "Story", "BLEU-4", "Semantic", "Clip BLEU", "Verdict"];
const rowsM = [
  ["3B, single rewrite step", "46 sentences", "-", "-", "-", "Ignored length, invented events"],
  ["3B, rolling context", "26 sentences", "0.047", "0.78", "-", "Role confusion"],
  ["3B, + cast list", "24 sentences", "0.050", "0.85", "0.70", "Names in wrong scenes"],
  ["3B, + name grounding", "24 sentences", "0.040", "0.81", "0.74", "Copies its own sentences"],
  ["7B, rolling + grounding (chosen)", `${summ.story_sentences.length} sentences`, fmt(sv["Summary story (ours)"].bleu4), fmt(sv["Summary story (ours)"].semantic_similarity, 2), fmt(fid.mean_clip_bleu, 2), "Coherent full plot arc"],
  ["TextRank (extractive)", "same length", fmt(sv["Extractive TextRank"].bleu4), fmt(sv["Extractive TextRank"].semantic_similarity, 2), "-", "No story; fallback only"],
];
s.addTable([head, ...rowsM].map((r, i) => r.map((c, j) => ({ text: c, options: {
  bold: i === 0 || i === 5, color: HEX.dk2, fill: { color: i === 0 ? HEX.lt2 : i === 5 ? "FBE9CC" : HEX.lt1 }, align: j === 0 || j === 5 ? "left" : "center" } }))),
  { x: 0.5, y: 1.15, w: 9.0, colW: [2.45, 1.25, 0.9, 1.0, 1.0, 2.4], fontSize: 12, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D9D8D3" }, rowH: 0.42 });
s.addText("The 7B model is ~2× slower (31 vs ~14 min) but is the only version whose story follows the film from start to end. Its BLEU-4 is lower only because it writes fewer words.",
  body({ x: 0.5, y: 4.25, w: 9.0, h: 0.7, fontSize: 13 }));

// ---------------------------------------------------------------- 15 conclusion
pres.addSection({ title: "Conclusion" });
s = addSlide("DARK_END", "Conclusion");
s.addText("Key observations and conclusion", { placeholder: "title" });
stat(s, 0.5, 1.15, 2.1, mmss(cons.recap_duration_seconds), "recap from a 67:38 film, CPU only", true);
stat(s, 0.5, 2.45, 2.1, fmt(fid.mean_clip_bleu, 2), "clip fidelity BLEU", true);
stat(s, 0.5, 3.75, 2.1, "9 / 9", "metrics where the story beats every baseline", true);
const obs = [
  ["Carry the time index through every stage", "chunk span → aligned window → peak line → clip → recap slot makes every claim traceable"],
  ["Modalities complement each other", "text picks the scene; audio, video and emotion pick the moment"],
  ["BLEU needs company", "use baselines and semantic similarity to judge abstractive summaries"],
  ["Next steps", "speaker diarisation, CLIP frame matching, a FastAPI service"],
];
obs.forEach(([h, t], i) => {
  badge(s, 3.1, 1.2 + i * 0.95, i + 1, 0.38);
  s.addText(h, body({ x: 3.65, y: 1.15 + i * 0.95, w: 5.85, h: 0.35, fontSize: 15, bold: true, color: C.background1 }));
  s.addText(t, body({ x: 3.65, y: 1.5 + i * 0.95, w: 5.85, h: 0.45, fontSize: 12.5, color: C.background2 }));
});

// ---------------------------------------------------------------- write + theme
const outFile = path.join(ROOT, "docs", "Movie_Recap_Presentation.pptx");
(async () => {
  await pres.writeFile({ fileName: outFile });
  const { applyTheme } = require(process.argv[2]);
  await applyTheme(outFile, THEME);
  console.log("wrote", outFile, "slides:", slideNo);
})();
