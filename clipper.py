import json
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

SRC = os.environ["VIDEO_URL"]  # link ya repo ke andar ki file ka path
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "5"))
MIN_LEN, MAX_LEN = 25, 55  # clip length seconds
OUT = Path("output")
OUT.mkdir(exist_ok=True)

KEYWORDS = ["wow", "crazy", "no way", "oh my god", "omg", "insane", "unbelievable",
            "bro", "what", "never", "secret", "best", "worst", "shocked", "scared",
            "love", "hate", "money", "wait", "stop", "mistake", "truth"]


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


# 1) Video lao
if SRC.startswith("http"):
    run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
         "-o", "source.mp4", SRC])
    video = "source.mp4"
else:
    video = SRC

# 2) Transcribe (Whisper, free, CPU)
print("Transcribing...")
model = WhisperModel("small", device="cpu", compute_type="int8")
segs, _ = model.transcribe(video, word_timestamps=True)
segs = list(segs)
words = [w for s in segs for w in (s.words or [])]

# 3) Audio loudness har second ke liye
print("Analysing audio...")
raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", video, "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
    capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
n_sec = len(audio) // 8000
rms = np.array([np.sqrt(np.mean(audio[i * 8000:(i + 1) * 8000] ** 2)) for i in range(n_sec)])
rms = rms / (rms.max() + 1e-9)

# 4) Candidate windows ko score do
cands = []
for i, s in enumerate(segs):
    start = s.start
    end = None
    text = ""
    for t in segs[i:]:
        text += " " + t.text
        if t.end - start >= MIN_LEN:
            end = t.end
            break
    if end is None or end - start > MAX_LEN or end > n_sec:
        continue
    loud = float(rms[int(start):int(end)].mean())
    n_words = len(text.split())
    speed = min(n_words / (end - start) / 4.0, 1.0)  # ~4 words/sec = fast
    low = text.lower()
    excite = min(low.count("!") * 0.1 + low.count("?") * 0.05
                 + sum(low.count(k) for k in KEYWORDS) * 0.1, 1.0)
    score = loud + 0.7 * speed + 1.0 * excite
    cands.append({"start": start, "end": end, "score": score, "text": text.strip()})

cands.sort(key=lambda c: c["score"], reverse=True)
clips = []
for c in cands:
    if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in clips):
        clips.append(c)
    if len(clips) == NUM_CLIPS:
        break

for c in clips:
    first = c["text"].replace("\n", " ").split(".")[0].split("?")[0].split("!")[0]
    c["hook"] = " ".join(first.split()[:8]) + "..."

(OUT / "clips.json").write_text(json.dumps(clips, indent=2, ensure_ascii=False))


def ts(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


# 5) Har clip: captions + 9:16 crop + hook
for i, c in enumerate(clips, 1):
    start, end = float(c["start"]), float(c["end"])
    ws = [w for w in words if w.start >= start and w.end <= end]
    if not ws:
        continue
    lines = []
    for k in range(0, len(ws), 3):
        chunk = ws[k:k + 3]
        a, b = chunk[0].start - start, chunk[-1].end - start
        txt = " ".join(w.word.strip() for w in chunk).upper()
        lines.append(f"{len(lines) + 1}\n{ts(a)} --> {ts(b)}\n{txt}\n")
    Path("cap.srt").write_text("\n".join(lines), encoding="utf-8")
    Path("hook.txt").write_text(textwrap.fill(c["hook"].upper(), 22), encoding="utf-8")

    vf = (
        "crop=ih*9/16:ih,scale=1080:1920,"
        "subtitles=cap.srt:force_style='Alignment=2,FontSize=10,Bold=1,Outline=2,MarginV=40',"
        "drawtext=textfile=hook.txt:fontcolor=white:fontsize=60:box=1:"
        "boxcolor=0xA020F0@0.85:boxborderw=20:x=(w-text_w)/2:y=220:enable='lt(t,3)'"
    )
    run(["ffmpeg", "-y", "-ss", str(start), "-t", str(end - start), "-i", video,
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", str(OUT / f"clip_{i:02d}.mp4")])

print("Done. Clips in output/")
