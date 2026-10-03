"""Renders the images used in the presentation deck from the pipeline outputs (run from the project root)."""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recap.peaks import build_windows  # noqa: E402
from recap.summarize import embed  # noqa: E402
from recap.utils import fmt_ts, load_config, load_json  # noqa: E402

cfg = load_config()
OUT = Path(cfg["output_dir"])
ASSETS = ROOT / "docs" / "slide_assets"
ASSETS.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})


def frame_at(path, t):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * cap.get(cv2.CAP_PROP_FPS)))
    ok, f = cap.read()
    cap.release()
    return f


# 1. Recap frames: chapter card, clip with badges, intro card
edl = pd.read_csv(OUT / "recap_edl.csv")
card, clip = edl[edl.type == "chapter"].iloc[4], edl[edl.type == "clip"].iloc[4]
recap = OUT / "movie_recap.mp4"
cv2.imwrite(str(ASSETS / "recap_card.jpg"), frame_at(recap, card.recap_start + 3))
cv2.imwrite(str(ASSETS / "recap_clip.jpg"), frame_at(recap, clip.recap_start + 2))
cv2.imwrite(str(ASSETS / "recap_intro.jpg"), frame_at(recap, 2.5))

# 2. Peak keyframes strip (first 8 peaks) with time index labels
pk = load_json(OUT / "peaks.json")
video = cfg["input"]["local_path"]
tiles = []
for p in pk["peaks"][:8]:
    f = cv2.resize(frame_at(video, p["timestamp"]), (320, 240))
    cv2.rectangle(f, (0, 208), (320, 240), (0, 0, 0), -1)
    cv2.putText(f, f"PEAK {p['index']}  {p['timestamp_hms']}", (8, 231), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (64, 186, 255), 1, cv2.LINE_AA)
    tiles.append(f)
grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
cv2.imwrite(str(ASSETS / "peak_keyframes.jpg"), grid)

# 3. DP alignment heatmap (story sentence x transcript window)
tr = load_json(OUT / "transcript.json")["segments"]
summ = load_json(OUT / "summary.json")
windows = build_windows(tr, cfg["peaks"]["window_seconds"], cfg["peaks"]["window_hop_seconds"])
model = cfg["peaks"]["embedding_model"]
sim = embed(summ["story_sentences"], model) @ embed([w["text"] for w in windows], model).T
centres = np.array([(w["start"] + w["end"]) / 2 for w in windows]) / 60
fig, ax = plt.subplots(figsize=(11, 4.4))
im = ax.imshow(sim, aspect="auto", cmap="Greys", extent=[centres[0], centres[-1], len(sim) + 0.5, 0.5])
fig.colorbar(im, ax=ax, label="cosine similarity")
ax.plot([centres[j] for j in pk["alignment_path"]], np.arange(1, len(sim) + 1), "o-", color="#eb6834", ms=6, lw=1.6,
        label="monotonic DP alignment")
ax.scatter([p["timestamp"] / 60 for p in pk["peaks"]], np.arange(1, len(sim) + 1), marker="x", color="#2a78d6", s=70,
           linewidths=2.2, label="selected peak point", zorder=3)
ax.set_xlabel("Film time (minutes)")
ax.set_ylabel("Story sentence #")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
ax.grid(False)
fig.tight_layout()
fig.savefig(ASSETS / "alignment_heatmap.png", dpi=170)
plt.close(fig)
print("assets written to", ASSETS)
