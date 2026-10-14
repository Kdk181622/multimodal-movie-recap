"""Builds a narrated explanation video from the slide images and docs/speaker_notes.json.

Usage (project root):
    python docs/make_explanation_video.py <slides_png_dir> <output.mp4>

<slides_png_dir> holds slide01.png ... slideNN.png exported from Movie_Recap_Presentation.pptx at 1920x1080.
Narration uses the neural voice en-IN-PrabhatNeural via edge-tts (online); if that fails, the offline
Windows voice is used. After the slide 11 narration, the embedded recap excerpt plays full screen.
"""

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

DOCS = Path(__file__).resolve().parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VOICE = "en-IN-PrabhatNeural"
EXCERPT_AFTER_SLIDE = 11
PAUSE_BEFORE, PAUSE_AFTER = 0.5, 0.9  # seconds of silence around each slide's narration

# Spoken forms for names a speech engine tends to mispronounce (narration only; slides are unchanged).
SAY = {"Qwen": "Chwen", "BLEU": "blue", "sacreBLEU": "sacre-blue", "MiniLM": "Mini L M", "DistilRoBERTa": "Distil Roberta",
       "librosa": "lib-rosa", "yt-dlp": "Y T D L P", "Ollama": "Oh-llama", "OpenCV": "Open C V", "ffmpeg": "F F mpeg",
       "TextRank": "Text Rank", "Ulmer": "Ulmer"}


def spoken(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)  # drop stage directions such as [Click the video ...]
    for k, v in sorted(SAY.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return re.sub(r"\s+", " ", text).strip()


def run(args):
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def duration(path: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


async def tts_neural(text: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, VOICE, rate="-4%").save(str(path))


def tts_offline(text: str, path: Path) -> None:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.save_to_file(text, str(path))
    engine.runAndWait()


def main(slides_dir: Path, output: Path) -> None:
    notes = json.loads((DOCS / "speaker_notes.json").read_text(encoding="utf-8"))
    work = output.parent / "_video_build"
    work.mkdir(parents=True, exist_ok=True)
    segments = []
    # Shared encode settings so the segments can be joined without re-encoding.
    venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30"]
    aenc = ["-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2"]

    for i, note in enumerate(notes, 1):
        audio = work / f"narr{i:02d}.mp3"
        if not audio.exists():
            try:
                asyncio.run(tts_neural(spoken(note["script"]), audio))
            except Exception as exc:  # no internet: fall back to the offline OS voice
                print(f"slide {i}: neural TTS failed ({exc}); using offline voice")
                audio = audio.with_suffix(".wav")
                tts_offline(spoken(note["script"]), audio)
        seconds = PAUSE_BEFORE + duration(audio) + PAUSE_AFTER
        seg = work / f"seg{i:02d}.mp4"
        run(["-loop", "1", "-framerate", "30", "-i", str(slides_dir / f"slide{i:02d}.png"), "-i", str(audio),
             "-filter_complex", f"[1:a]adelay={int(PAUSE_BEFORE * 1000)}:all=1,apad[a]", "-map", "0:v", "-map", "[a]",
             "-t", f"{seconds:.2f}", *venc, "-tune", "stillimage", *aenc, str(seg)])
        segments.append(seg)
        print(f"slide {i:2d}: {seconds:5.1f} s")

        if i == EXCERPT_AFTER_SLIDE:
            seg = work / "seg_excerpt.mp4"
            run(["-i", str(DOCS / "slide_assets" / "recap_excerpt.mp4"),
                 "-vf", "scale=1440:1080:flags=lanczos,pad=1920:1080:240:0:black,setsar=1", *venc, *aenc, str(seg)])
            segments.append(seg)
            print(f"recap excerpt: {duration(seg):5.1f} s")

    listing = work / "segments.txt"
    listing.write_text("".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="utf-8")
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(output)])
    total = duration(output)
    print(f"wrote {output} ({total / 60:.1f} min)")
    if not 8 * 60 <= total <= 20 * 60:
        raise SystemExit("video length is outside the required 8-20 minutes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
