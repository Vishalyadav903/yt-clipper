#!/usr/bin/env python3
"""
yt-clipper: paste a YouTube link -> get 6-10 viral-style vertical clips with captions.

Pipeline:
  1. yt-dlp downloads the video
  2. Groq Whisper transcribes it (word-level timestamps)
  3. Groq LLM picks the most engaging moments
  4. ffmpeg cuts, crops to 9:16 and burns animated captions (ASS subtitles)

Needs: python3, ffmpeg, yt-dlp, requests, and a free GROQ_API_KEY (console.groq.com)
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import requests

GROQ_URL = "https://api.groq.com/openai/v1"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
GF = "https://github.com/google/fonts/raw/main/ofl"
FONTS = {  # family name (used in the subtitle style) -> download url
    "Anton": f"{GF}/anton/Anton-Regular.ttf",
    "Bebas Neue": f"{GF}/bebasneue/BebasNeue-Regular.ttf",
    "Luckiest Guy": f"{GF}/luckiestguy/LuckiestGuy-Regular.ttf",
    "Bangers": f"{GF}/bangers/Bangers-Regular.ttf",
}

# ASS colours are BGR: &HBBGGRR&
HIGHLIGHT = "&H0000F0FF&"  # yellow
WHITE = "&HFFFFFF&"


# ------------------------------------------------------------------ helpers
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"Command failed: {' '.join(map(str, cmd))}\n{r.stderr[-1500:]}")
    return r.stdout


def need_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("Set GROQ_API_KEY first (free key at https://console.groq.com)")
    return key


def post_with_retry(url, retries=5, **kw):
    for i in range(retries):
        r = requests.post(url, headers={"Authorization": f"Bearer {need_key()}"}, timeout=300, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            wait = 5 * (i + 1)
            print(f"  API busy ({r.status_code}), retrying in {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code != 200:
            sys.exit(f"API error {r.status_code}: {r.text[:500]}")
        return r.json()
    sys.exit("API kept failing, try again later.")


# ----------------------------------------------------------------- download
def download(url, out_root, cookies=None):
    base = ["yt-dlp", "--no-playlist"]
    if cookies:
        base += ["--cookies", cookies]
    vid = run(base + ["--get-id", url]).strip().splitlines()[-1]
    work = out_root / vid
    work.mkdir(parents=True, exist_ok=True)
    src = work / "source.mp4"
    if not src.exists():
        print("Downloading video...")
        run(base + [
            "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format", "mp4",
            "-o", str(work / "source.%(ext)s"), url,
        ])
    return work, src


# ------------------------------------------------------------ transcription
def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(path)])
    return float(out.strip())


def transcribe(video, work, lang=None):
    cache = work / "transcript.json"
    if cache.exists():
        print("Using cached transcript.")
        return json.loads(cache.read_text())

    print("Transcribing...")
    total = duration(video)
    chunk = 600  # 10 min mp3 @32kbps ~ 2.4MB, far below API limits
    words, segments = [], []
    for i, off in enumerate(range(0, int(total) + 1, chunk)):
        mp3 = work / f"audio_{i}.mp3"
        run(["ffmpeg", "-y", "-v", "error", "-ss", str(off), "-t", str(chunk), "-i", str(video),
             "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(mp3)])
        if not mp3.exists() or mp3.stat().st_size < 2000:
            continue
        data = {"model": WHISPER_MODEL, "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"]}
        if lang:
            data["language"] = lang
        with open(mp3, "rb") as f:
            res = post_with_retry(f"{GROQ_URL}/audio/transcriptions",
                                  files={"file": (mp3.name, f, "audio/mpeg")}, data=data)
        for w in res.get("words", []):
            words.append({"w": w["word"].strip(), "s": w["start"] + off, "e": w["end"] + off})
        for s in res.get("segments", []):
            segments.append({"s": s["start"] + off, "e": s["end"] + off, "t": s["text"].strip()})
        mp3.unlink()
        print(f"  transcribed {min(off + chunk, int(total))}/{int(total)}s")

    result = {"words": words, "segments": segments, "duration": total}
    cache.write_text(json.dumps(result))
    return result


# ------------------------------------------------------- picking the clips
PROMPT = """You are an expert short-form video editor for TikTok, Reels and YouTube Shorts.
Below is a timestamped transcript ([seconds] text). Pick the {k} MOST ENGAGING moments.

Rules:
- Each clip must be {mn}-{mx} seconds long and make sense on its own (no missing context).
- It must start with a strong hook (bold claim, question, surprising fact, story start, emotional line).
- It must end on a complete thought or payoff, never mid-sentence.
- Prefer: surprising insights, strong opinions, emotional/funny moments, stories, actionable tips, controversy.
- Avoid: intros, sponsor reads, small talk, rambling.
- Clips must not overlap.
- score = 0-100 predicted viral potential.

Return ONLY JSON: {{"clips":[{{"start":<sec>,"end":<sec>,"title":"<catchy title <60 chars>","hook":"<first line>","score":<int>,"reason":"<short>"}}]}}

TRANSCRIPT:
{text}
"""


def transcript_blocks(segments, limit=14000):
    blocks, cur, size = [], [], 0
    for s in segments:
        line = f"[{int(s['s'])}] {s['t']}"
        if size + len(line) > limit and cur:
            blocks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def ask_llm(text, k, mn, mx):
    res = post_with_retry(f"{GROQ_URL}/chat/completions", json={
        "model": LLM_MODEL,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": PROMPT.format(k=k, mn=mn, mx=mx, text=text)}],
    })
    raw = res["choices"][0]["message"]["content"]
    try:
        return json.loads(raw).get("clips", [])
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)).get("clips", []) if m else []


def overlaps(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return inter / max(1, min(a["end"] - a["start"], b["end"] - b["start"])) > 0.3


def pick_clips(tr, n, mn, mx):
    blocks = transcript_blocks(tr["segments"])
    per = max(3, math.ceil(n * 1.5 / len(blocks)))
    cands = []
    print(f"Finding best moments ({len(blocks)} section(s))...")
    for b in blocks:
        for c in ask_llm(b, per, mn, mx):
            try:
                c["start"], c["end"], c["score"] = float(c["start"]), float(c["end"]), int(c.get("score", 50))
            except (KeyError, ValueError, TypeError):
                continue
            if c["end"] > c["start"]:
                cands.append(c)
    cands.sort(key=lambda c: c["score"], reverse=True)
    chosen = []
    for c in cands:
        if not any(overlaps(c, o) for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return [snap(c, tr["words"], tr["duration"], mn, mx) for c in chosen]


def snap(c, words, total, mn, mx):
    """Align clip edges to real word boundaries and enforce duration limits."""
    inside = [w for w in words if w["s"] >= c["start"] - 0.4 and w["e"] <= c["end"] + 1.0]
    if inside:
        c["start"] = max(0, inside[0]["s"] - 0.15)
        c["end"] = inside[-1]["e"] + 0.3
    if c["end"] - c["start"] > mx:
        c["end"] = c["start"] + mx
    if c["end"] - c["start"] < mn:
        c["end"] = c["start"] + mn
    c["end"] = min(c["end"], total)
    return c


# ---------------------------------------------------------------- captions
def ass_time(t):
    t = max(0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def clean_word(w):
    return re.sub(r"^[\"'(]+|[,.;:\"')]+$", "", w).upper()


def group_words(words, per):
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or w["s"] - cur[-1]["e"] > 0.6):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.?!]$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words, start, end, path, font, size, per, margin_v):
    ws = [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.05 and clean_word(w["w"])]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,2,0,1,8,4,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for g in group_words(ws, per):
        for j, w in enumerate(g):
            s = w["s"] - start
            e = (g[j + 1]["s"] - start) if j + 1 < len(g) else (w["e"] - start + 0.08)
            if e <= s:
                e = s + 0.1
            parts = []
            for k, x in enumerate(g):
                t = clean_word(x["w"])
                if k == j:
                    parts.append(f"{{\\c{HIGHLIGHT}\\fscx115\\fscy115}}{t}{{\\c{WHITE}\\fscx100\\fscy100}}")
                else:
                    parts.append(t)
            lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Cap,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


def ensure_font(name):
    FONT_DIR.mkdir(exist_ok=True)
    if name in FONTS:
        dest = FONT_DIR / Path(FONTS[name]).name
        if not dest.exists():
            print(f"Downloading font {name}...")
            try:
                urllib.request.urlretrieve(FONTS[name], dest)
            except Exception as e:  # noqa
                print(f"  could not download font ({e}); falling back to system font")


# ------------------------------------------------------------------ render
def esc(p):
    return str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def render(video, ass, start, end, out, mode):
    sub = f"ass={esc(ass)}:fontsdir={esc(FONT_DIR)}"
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{end - start:.2f}"]
    if mode == "blur":  # full video centred over a blurred copy of itself
        fc = ("[0:v]split=2[a][b];"
              "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
              "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1," + sub + "[v]")
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "0:a?"]
    else:  # centre crop to 9:16
        vf = f"crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1,{sub}"
        cmd += ["-vf", vf]
    run(cmd + enc + [str(out)])


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Turn a YouTube video into viral short clips")
    ap.add_argument("url")
    ap.add_argument("-n", "--clips", type=int, default=8, help="number of clips (6-10)")
    ap.add_argument("--min", type=int, default=20, help="min clip seconds")
    ap.add_argument("--max", type=int, default=60, help="max clip seconds")
    ap.add_argument("--font", default="Anton", help=f"one of {list(FONTS)} or any font in ./fonts")
    ap.add_argument("--font-size", type=int, default=100)
    ap.add_argument("--words", type=int, default=3, help="words shown at once")
    ap.add_argument("--mode", choices=["crop", "blur"], default="crop",
                    help="crop = fill 9:16, blur = whole frame over blurred background")
    ap.add_argument("--caption-height", type=int, default=480, help="caption distance from bottom (px)")
    ap.add_argument("--lang", default=None, help="audio language code, e.g. en, hi")
    ap.add_argument("--cookies", default=None, help="cookies.txt for age-restricted / bot-checked videos")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    n = max(6, min(10, a.clips))
    out_root = Path(a.out)
    work, video = download(a.url, out_root, a.cookies)
    tr = transcribe(video, work, a.lang)
    if not tr["words"]:
        sys.exit("No speech detected in this video.")
    clips = pick_clips(tr, n, a.min, a.max)
    ensure_font(a.font)

    clips_dir = work / "clips"
    clips_dir.mkdir(exist_ok=True)
    meta = []
    for i, c in enumerate(clips, 1):
        name = f"clip_{i:02d}"
        ass = work / f"{name}.ass"
        mp4 = clips_dir / f"{name}.mp4"
        print(f"[{i}/{len(clips)}] {c['title']}  ({c['end'] - c['start']:.0f}s, score {c['score']})")
        build_ass(tr["words"], c["start"], c["end"], ass, a.font, a.font_size, a.words, a.caption_height)
        render(video, ass, c["start"], c["end"], mp4, a.mode)
        meta.append({"file": mp4.name, **c})
    (clips_dir / "clips.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nDone! {len(meta)} clips saved in: {clips_dir}")


if __name__ == "__main__":
    main()
