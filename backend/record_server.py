# record_server.py - training-data recorder. Run from backend:
#   uvicorn record_server:app --port 8001
# then open http://localhost:8001/record
# Standalone on purpose: server.py is not touched, and the GPU is not used.
# Output: backend/train_data/wavs/*.wav (16 kHz mono PCM16) + manifest.jsonl (UTF-8).
import collections, json, math, os, random, re, sqlite3, time, wave

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "quran.db")
DATA = os.path.join(HERE, "train_data")
WAVS = os.path.join(DATA, "wavs")
MANIFEST = os.path.join(DATA, "manifest.jsonl")
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.wav$")
os.makedirs(WAVS, exist_ok=True)

# The 30 letter openings (surah, ayah). Self-contained so this file needs nothing else.
MUQ = {(2, 1), (3, 1), (7, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1),
       (19, 1), (20, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1),
       (36, 1), (38, 1), (40, 1), (41, 1), (42, 1), (42, 2), (43, 1), (44, 1), (45, 1),
       (46, 1), (50, 1), (68, 1)}


def load_ayahs():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT surah, ayah, pos, text_uthmani, text_clean "
                       "FROM words ORDER BY surah, ayah, pos").fetchall()
    con.close()
    acc = {}
    for s, a, p, u, c in rows:
        d = acc.setdefault((s, a), ([], []))
        d[0].append(u)
        d[1].append(c)
    return {k: (" ".join(v[0]), " ".join(v[1])) for k, v in acc.items()}


AY = load_ayahs()
app = FastAPI()


def read_manifest():
    rows = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        pass
    return rows


def write_manifest(rows):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, MANIFEST)


def stats(rows=None):
    rows = read_manifest() if rows is None else rows
    return {"clips": len(rows), "minutes": round(sum(r.get("dur", 0) for r in rows) / 60, 1)}


def parse_surahs(s):
    out = set()
    try:
        for part in s.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    except ValueError:
        raise HTTPException(400, "bad surah list; use e.g. 78-114,1,36")
    return sorted(x for x in out if 1 <= x <= 114)


@app.get("/")
def home():
    return RedirectResponse("/record")


@app.get("/record")
def page():
    return FileResponse(os.path.join(HERE, "record.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/rec/queue")
def queue(surahs: str = "78-114", muq: int = 1, order: str = "shuffle", seed: int = 1, muq_takes: int = 1):
    rows = read_manifest()
    counts = collections.Counter((r["surah"], r["ayah"]) for r in rows)
    sel = set(parse_surahs(surahs))
    keys = {k for k in AY if k[0] in sel}
    if muq:
        keys |= {k for k in MUQ if k in AY}
    keys = sorted(keys)
    items = []
    for k in keys:
        kind = "muq" if k in MUQ else ("bas" if k == (1, 1) else "ayah")
        want = max(1, min(20, muq_takes))      # total takes wanted per ayah (patch_takes)
        for _ in range(max(0, want - counts[k])):
            items.append({"surah": k[0], "ayah": k[1], "kind": kind,
                          "uthmani": AY[k][0], "clean": AY[k][1]})
    if order == "shuffle":
        random.Random(seed).shuffle(items)
    return {"items": items, "stats": stats(rows)}


@app.post("/api/rec/save")
async def save(request: Request, surah: int, ayah: int, kind: str = "ayah"):
    if (surah, ayah) not in AY:
        raise HTTPException(400, "unknown ayah")
    raw = await request.body()
    raw = raw[: len(raw) // 2 * 2]
    if len(raw) < 2 * 8000:
        raise HTTPException(400, "clip shorter than 0.5 s")
    a = np.frombuffer(raw, np.int16)
    dur = len(a) / 16000
    rms = float(np.sqrt(np.mean((a.astype(np.float32) / 32768.0) ** 2)))
    db = 20 * math.log10(rms + 1e-9)
    peak = int(np.abs(a.astype(np.int32)).max())
    warn = []
    if peak >= 32000:
        warn.append("clipping")
    if db < -45:
        warn.append("very quiet")
    if dur > 29:
        warn.append("over 29 s: skipped in training (Whisper window is 30 s)")
    rows = read_manifest()
    take = 1 + sum(1 for r in rows if r["surah"] == surah and r["ayah"] == ayah)
    name = "s%03d_a%03d_t%d_%d.wav" % (surah, ayah, take, int(time.time()))
    with wave.open(os.path.join(WAVS, name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(raw)
    u, c = AY[(surah, ayah)]
    rows.append({"file": "wavs/" + name, "surah": surah, "ayah": ayah, "kind": kind, "take": take,
                 "uthmani": u, "clean": c, "dur": round(dur, 2), "rms_db": round(db, 1),
                 "peak": peak, "ts": int(time.time())})
    write_manifest(rows)
    return {"file": name, "dur": round(dur, 1), "rms_db": round(db), "warn": warn, "stats": stats(rows)}


@app.post("/api/rec/delete")
def delete(file: str):
    if not NAME_RE.match(file):
        raise HTTPException(400, "bad name")
    rows = [r for r in read_manifest() if os.path.basename(r["file"]) != file]
    write_manifest(rows)
    try:
        os.remove(os.path.join(WAVS, file))
    except OSError:
        pass
    return {"stats": stats(rows)}


@app.get("/api/rec/wav/{name}")
def wav(name: str):
    if not NAME_RE.match(name) or not os.path.exists(os.path.join(WAVS, name)):
        raise HTTPException(404)
    return FileResponse(os.path.join(WAVS, name), media_type="audio/wav")
