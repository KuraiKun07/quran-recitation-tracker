"""Streaming word-level alignment of ASR output against canonical Quran text.

align()        read-only: semi-global DP of spoken words vs the next `window` reference words (free tail, so stopping
               mid-ayah is fine). Skipping reference words costs DEL, extra spoken words cost INS.
commit()       accept a leading slice of events and advance the cursor.
suggest_jump() detect a RESTART/REWIND/JUMP: the spoken words match a different place in the surah far better than
               they match the expected place.
"""
import re
from collections import defaultdict
from math import ceil
from typing import Dict, List, Optional, Tuple

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")

_NORM_HA = __import__("os").environ.get("NORM_HAMZA_ALEF", "1") == "1"
def normalize(text: str) -> str:
    t = _TASHKEEL.sub("", text)
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\u0621-\u064A ]", "", t).strip()
    if _NORM_HA:  # reference u0621+u0627 -> u0627 (ASR writes one alef-madda)
        t = t.replace("\u0621\u0627", "\u0627")
    return t

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def _similarity(a: str, b: str) -> float:
    return 1 - levenshtein(a, b) / max(len(a), len(b), 1)

# similarity of two strings never changes -> cache it (the jump search re-scores the same
# pairs thousands of times per pass). ALIGN_MEMO=0 switches the cache off.
from functools import lru_cache as _lru
_MEMO = __import__("os").environ.get("ALIGN_MEMO", "1") != "0"
similarity = _lru(maxsize=1 << 16)(_similarity) if _MEMO else _similarity

def skeleton(w: str) -> str:
    """Drop weak letters (ا و ي) after the first letter: ASR often drops/garbles word endings like كفوا -> كف."""
    return w[:1] + "".join(c for c in w[1:] if c not in "اوي")

def public(e: Dict) -> Dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}

# INS is slightly dearer than SUB so a wrong word is reported as "wrong", not as an "extra word", when costs tie
DEL, INS, SUB = 0.9, 1.05, 1.0
# split fold: two spoken tokens may be ONE reference word the ASR split; a joined pair must reach SPLIT_MIN and beat each token by SPLIT_GAIN
import os as _os
SPLIT_FOLD = _os.environ.get("SPLIT_FOLD", "1") == "1"
SPLIT_MIN = float(_os.environ.get("SPLIT_MIN", "0.6"))
SPLIT_GAIN = float(_os.environ.get("SPLIT_GAIN", "0.15"))
SPLIT_MIN_LEN = int(_os.environ.get("SPLIT_MIN_LEN", "4"))
SUB_GRADE = float(_os.environ.get("SUB_GRADE", "0.4"))   # a near-miss wrong word costs less than a far one
JUMP_PENALTY = 0.5
NEAR_PREFIX = int(_os.environ.get("NEAR_PREFIX", "1"))   # letters that must match; 2 = old rule
NEAR_MIN = float(_os.environ.get("NEAR_MIN", "0.5"))      # a wrong word that is this similar AND starts the same is reported as 'unsure', not 'wrong'

ALIGN_CHUNK = _os.environ.get("ALIGN_CHUNK", "1") != "0"   # align long readings in slices
ALIGN_MAX_TOKENS = int(_os.environ.get("ALIGN_MAX_TOKENS", "96"))

def _sim_pair(a: str, b: str, skel: bool) -> float:
    """Body of Aligner._sim as a pure (cacheable) function."""
    s = similarity(a, b)
    if skel:                                            # tolerate dropped/garbled weak letters (>=2 consonants must agree)
        ka, kb = skeleton(a), skeleton(b)
        if min(len(ka), len(kb)) >= 2:
            s = max(s, 0.9 * similarity(ka, kb))
    return s
if _MEMO:
    _sim_pair = _lru(maxsize=1 << 16)(_sim_pair)

class Aligner:
    def __init__(self, words: List[Dict], start_index: int = 0, window: int = 10,
                 threshold: float = 0.75, strict: bool = False, lenient: bool = False):
        self.words, self.cursor = words, start_index
        self.window, self.threshold, self.strict, self.lenient = window, threshold, strict, lenient
        self.index = defaultdict(list)                      # normalized word -> positions (for jump detection)
        self._keys = [self._key(w) for w in words]          # reference keys computed once
        for i, k in enumerate(self._keys):
            self.index[k].append(i)

    def _key(self, w):
        return w["text_uthmani"] if self.strict else normalize(w["text_clean"])

    def _spoken(self, transcript: str, cap: int = 0) -> List[Tuple[str, str]]:
        raw = transcript.split()
        sp = [w if self.strict else normalize(w) for w in raw]
        return [(s, r) for s, r in zip(sp, raw) if s][: cap or self.window + 3]

    def _sim(self, a: str, b: str) -> float:
        return _sim_pair(a, b, self.lenient and not self.strict)

    def _split_cand(self, sp, rk, i, base, sg):
        """Split fold: spoken tokens i-2 and i-1 joined may be ONE reference word that the ASR split in two.
        Returns (cost, op) for D[i][j] or None. 'okj' only when the joined form passes the normal threshold."""
        a, b = sp[i - 2][0], sp[i - 1][0]
        jn = a + b
        if len(jn) < SPLIT_MIN_LEN:
            return None
        s2 = self._sim(jn, rk)
        if s2 < SPLIT_MIN or s2 - max(self._sim(a, rk), self._sim(b, rk)) < SPLIT_GAIN:
            return None
        ok = s2 >= self.threshold
        cost = (1 - s2) if ok else (SUB - sg * s2 if s2 >= NEAR_MIN else SUB)
        return base + cost, ("okj" if ok else "subj")

    def _dp(self, sp, start: int):
        ref = self.words[start:start + self.window]
        n, m = len(sp), len(ref)
        if n == 0 or m == 0:
            return None
        ref_k = self._keys[start:start + self.window]
        sg = SUB_GRADE
        D = [[0.0] * (m + 1) for _ in range(n + 1)]
        B = [[""] * (m + 1) for _ in range(n + 1)]
        for j in range(1, m + 1): D[0][j], B[0][j] = j * DEL, "del"
        for i in range(1, n + 1): D[i][0], B[i][0] = i * INS, "ins"
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                s = self._sim(sp[i - 1][0], ref_k[j - 1])
                ok = s >= self.threshold
                D[i][j], B[i][j] = min(
                    (D[i - 1][j - 1] + ((1 - s) if ok else (SUB - sg * s if s >= NEAR_MIN else SUB)), "ok" if ok else "sub"),
                    (D[i][j - 1] + DEL, "del"),
                    (D[i - 1][j] + INS, "ins"),
                    key=lambda x: x[0])
                if SPLIT_FOLD and i >= 2:
                    c = self._split_cand(sp, ref_k[j - 1], i, D[i - 2][j - 1], sg)
                    if c is not None and c[0] < D[i][j]:
                        D[i][j], B[i][j] = c
        best_j = min(range(m + 1), key=lambda j: D[n][j])   # free tail
        i, j, ops = n, best_j, []
        while i > 0 or j > 0:
            op = B[i][j]
            if op in ("okj", "subj"): ops.append((op, j - 1, i - 1)); i, j = i - 2, j - 1
            elif op in ("ok", "sub"): ops.append((op, j - 1, i - 1)); i, j = i - 1, j - 1
            elif op == "del":       ops.append((op, j - 1, None));  j -= 1
            else:                   ops.append((op, j, i - 1));     i -= 1
        ops.reverse()
        return D[n][best_j], ops

    def _align_dp(self, sp, start: int, off: int):
        """One window of reference words. off = window start relative to self.cursor (see align)."""
        r = self._dp(sp, start)
        if r is None:
            return [], -1
        events = []
        for op, rj, si in r[1]:
            jn = op in ("okj", "subj")      # split fold: si = LAST of two joined spoken tokens
            if jn:
                op = op[:-1]
            gi = start + rj
            ev = {"event": "alignment_result", "matched_word_index": gi, "expected_word": None,
                  "spoken_transcription": None, "unmask_next": op != "ok"}
            if op != "ins":
                ev["expected_word"] = self.words[gi]["text_uthmani"]
            if si is not None:
                ev["spoken_transcription"] = (sp[si - 1][1] + " " + sp[si][1]) if jn else sp[si][1]
            status = {"ok": "CORRECT", "sub": "MISTAKE_SUBSTITUTION",
                      "del": "MISTAKE_SKIPPED", "ins": "MISTAKE_INSERTION"}[op]
            if op == "sub":
                a_, b_ = ((sp[si - 1][0] + sp[si][0]) if jn else sp[si][0]), self._key(self.words[gi])
                if similarity(a_, b_) >= NEAR_MIN and a_[:NEAR_PREFIX] == b_[:NEAR_PREFIX]:
                    status = "MISTAKE_UNSURE"
            ev["status"] = status
            ev["_ref"] = None if op == "ins" else off + rj
            ev["_spk"] = 2 if jn else (op != "del")
            events.append(ev)
        last = max((rj for op, rj, si in r[1] if op != "ins"), default=-1)
        return events, last

    def align(self, transcript: str) -> List[Dict]:
        """Align from the cursor without moving it. Events carry private keys _ref / _spk (see commit).
        A reading longer than one window is aligned in slices (ALIGN_CHUNK=1): without that, the words
        past the window were reported as extra words or silently dropped instead of matched."""
        sp = self._spoken(transcript, cap=ALIGN_MAX_TOKENS if ALIGN_CHUNK else 0)
        if ALIGN_CHUNK and len(sp) > self.window:
            events, i, off = [], 0, 0
            step = max(1, self.window - 2)         # spare reference words at the seam so it cannot force an extra word
            while i < len(sp):
                ev, last = self._align_dp(sp[i:i + step], self.cursor + off, off)
                events.extend(ev)
                i += step
                if last >= 0:
                    off += last + 1
            return events
        events, _last = self._align_dp(sp, self.cursor, 0)
        return events

    def commit(self, events: List[Dict]) -> int:
        """Accept a leading slice of align()'s events: advance the cursor, return #spoken words consumed."""
        refs = [e["_ref"] for e in events if e["_ref"] is not None]
        if refs:
            self.cursor += max(refs) + 1
        return sum(e["_spk"] for e in events)

    def feed(self, transcript: str) -> List[Dict]:
        ev = self.align(transcript)
        self.commit(ev)
        return [public(e) for e in ev]

    def _score(self, sp, start: int):
        r = self._dp(sp, start)
        if r is None:
            return float("inf"), 0
        return r[0], sum(1 for o in r[1] if o[0] in ("ok", "okj"))

    def suggest_jump(self, transcript: str) -> Optional[int]:
        """Return a word index to restart from if the spoken words clearly belong somewhere else in the surah
        (user restarted, went back, or started a different ayah beyond the skip window). Needs >= 2 spoken words."""
        sp = self._spoken(transcript)
        if len(sp) < 2:
            return None
        fwd_cost, fwd_ok = self._score(sp, self.cursor)
        cands = set()
        for i in range(min(3, len(sp))):                    # anchor on any of the first 3 spoken words
            for pos in self.index.get(sp[i][0], ()):
                p = pos - i
                if p >= 0 and not (self.cursor <= p < self.cursor + self.window):
                    cands.add(p)
        need = max(2, ceil(0.8 * max(len(sp) - 1, 2)))       # almost every spoken word must match there (the last, often garbled, word is not required)
        best = None
        for p in sorted(cands, key=lambda p: abs(p - self.cursor))[:40]:
            cost, ok = self._score(sp, p)
            if ok >= need and ok > fwd_ok and cost + JUMP_PENALTY < fwd_cost and (best is None or cost < best[0]):
                best = (cost, p)
        return best[1] if best else None
