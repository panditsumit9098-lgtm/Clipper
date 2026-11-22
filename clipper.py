import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import anthropic
from faster_whisper import WhisperModel

SRC = os.environ["VIDEO_URL"]  # link ya repo ke andar ki file ka path
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "5"))
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
OUT = Path("output")
OUT.mkdir(exist_ok=True)


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
segments, _ = model.transcribe(video, word_timestamps=True)
segments = list(segments)
words = [w for s in segments for w in (s.words or [])]
transcript = "\n".join(f"[{s.start:.1f}] {s.text.strip()}" for s in segments)

# 3) Claude se best moments + hooks
print("Picking moments...")
client = anthropic.Anthropic()
prompt = f"""Below is a timestamped transcript of a long video.
Pick the {NUM_CLIPS} most engaging, self-contained moments for short-form clips (20-60 seconds each, no overlap).
For each give a scroll-stopping hook (max 8 words) shown at the start of the clip.
Reply with ONLY a JSON array: [{{"start": seconds, "end": seconds, "hook": "text"}}]

Transcript:
{transcript[:150000]}"""
resp = client.messages.create(
    model=MODEL, max_tokens=2000,
    messages=[{"role": "user", "content": prompt}],
)
text = resp.content[0].text
clips = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
(OUT / "clips.json").write_text(json.dumps(clips, indent=2, ensure_ascii=False))


def ts(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


# 4) Har clip: captions + crop + hook
for i, c in enumerate(clips, 1):
    start, end = float(c["start"]), float(c["end"])
    ws = [w for w in words if w.start >= start and w.end <= end]
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
