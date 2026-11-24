# trainutil.py - shared helpers for baseline / fine-tune scripts. Lives in backend.
import hashlib, json, os, re, wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "train_data")
HELDOUT_PCT = 20          # share of ayahs held out; the split is by ayah, deterministic


def load_manifest(max_dur=29.0):
    rows = []
    skip = int(os.environ.get("MANIFEST_SKIP", "0") or 0)   # ignore the first N manifest lines
    with open(os.path.join(DATA, "manifest.jsonl"), encoding="utf-8") as f:
        for _i, line in enumerate(f):
            if _i < skip:
                continue
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("dur", 0) <= max_dur:      # Whisper's window is 30 s
                    rows.append(r)
    return rows


def is_heldout(surah, ayah, pct=HELDOUT_PCT):
    h = int(hashlib.md5(("%d:%d" % (surah, ayah)).encode()).hexdigest(), 16)
    return h % 100 < pct


def fold_of(surah, ayah, n):
    """Cross-validation fold of an ayah (by ayah, deterministic, independent of is_heldout)."""
    return int(hashlib.md5(("f%d:%d" % (surah, ayah)).encode()).hexdigest(), 16) % n


def read_wav(rel):
    with wave.open(os.path.join(DATA, rel), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, rel
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def norm(t):
    """Same normalisation the aligner uses (falls back to a simple one if import fails)."""
    try:
        from alignment import normalize
        return normalize(t)
    except Exception:
        out = []
        for ch in t:
            o = ord(ch)
            if 0x064B <= o <= 0x065F or o == 0x0670 or 0x06D6 <= o <= 0x06ED or o == 0x0640:
                continue
            out.append({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}.get(ch, ch))
        return "".join(out)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


KEEP_DIAC = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652"   # the 8 marks the model actually writes


def to_model_style(u, shadda_first=True):
    """text_uthmani -> the diacritized style the base model writes: Arabic letters + the 8 standard
    marks only, wasla -> plain alef; dagger alef, maddah and all small Quranic signs dropped."""
    out = []
    for ch in u:
        if ch == "\u0671":
            ch = "\u0627"
        if ch == "\u0649":                      # Uthmani writes a yeh after kasra as alef maksura; the model writes yeh
            j = len(out) - 1
            while j >= 0 and out[j] in KEEP_DIAC:
                if out[j] == "\u0650":
                    ch = "\u064A"
                j -= 1
        if ch == " " or ("\u0621" <= ch <= "\u064A" and ch != "\u0640") or ch in KEEP_DIAC:
            out.append(ch)
    s = re.sub(" +", " ", "".join(out)).strip()

    def reorder(m):
        run = m.group()
        rest = run.replace("\u0651", "")
        return ("\u0651" + rest) if shadda_first else (rest + "\u0651")
    return re.sub("[\u064B-\u0652]{2,}", lambda m: reorder(m) if "\u0651" in m.group() else m.group(), s)


def sim(a, b):
    """1 - levenshtein/maxlen, the aligner's word similarity (my reading of the handoff)."""
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1.0 - edit_distance(a, b) / m


def tolerant_ops(ref, hyp, thr=0.75):
    """Word alignment where a substitution with sim >= thr counts as CORRECT (as the live aligner
    would accept it). Returns only the real errors: [(kind, ref_word, hyp_word)], kind in S/D/I."""
    n, m = len(ref), len(hyp)
    ok = [[sim(ref[i], hyp[j]) >= thr for j in range(m)] for i in range(n)]
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (0 if ok[i - 1][j - 1] else 1))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if ok[i - 1][j - 1] else 1):
            if not ok[i - 1][j - 1]:
                out.append(("S", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            out.append(("I", "", hyp[j - 1]))
            j -= 1
    return out[::-1]


def take_of(r):
    """Take number of a manifest row (older rows lack the field: read it from the file name)."""
    if r.get("take"):
        return int(r["take"])
    m = re.search(r"_t(\d+)_", r["file"])
    return int(m.group(1)) if m else 1


def is_heldout_take(r):
    """Closed-set split for letter openings: takes 3, 6, 9... are held out (about a third)."""
    return take_of(r) % 3 == 0
