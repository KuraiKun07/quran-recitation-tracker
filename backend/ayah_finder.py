"""
ayah_finder.py  -  v1.2 prototype: find an ayah from a few (noisy) transcribed words.

Pure Python, no extra dependencies. Reads backend/quran.db (words table) and reuses
normalize / similarity / skeleton from alignment.py, so "similar word" means the same
thing as in the live aligner.

Pipeline
  1. Fuzzy neighbours: each query word -> up to 8 vocabulary words with similarity >= 0.6
     (char-trigram candidates, then edit-distance similarity; trigrams are NOT anchored to
     the first letter, so wrong first letters still find their word).
  2. IDF voting: neighbours vote for the ayahs containing them (rare words count more;
     very common words are skipped at this stage).
  3. Re-rank the top candidates with an order-aware local alignment (free start, free end,
     so mid-ayah phrases work), with a small bonus when the match starts at the ayah opening.

Public API
  f = AyahFinder("quran.db")
  r = f.find("raw transcript text", topn=5)
  f.find(text, topn=5, partial=False)   # partial=True while the utterance is still streaming
  r -> {"n_words", "basmala_stripped", "results": [{"surah","ayah","start","score","conf"}...], "margin", "confident"}
  "start" = 0-based word offset inside the ayah where the spoken words begin
  (hand-off: start message with starting_ayah=ayah, expected_word_index=start).
"""
import collections
import math
import sqlite3
from functools import lru_cache

from alignment import normalize, similarity, skeleton


def _tri(w):
    p = f" {w} "
    return {p[i:i + 3] for i in range(len(p) - 2)}


class AyahFinder:
    def __init__(self, db_path, min_sim=0.6, max_neigh=8, cand_pool=60,
                 rerank_top=20, common_df=400):
        self.min_sim, self.max_neigh = min_sim, max_neigh
        self.cand_pool, self.rerank_top, self.common_df = cand_pool, rerank_top, common_df

        con = sqlite3.connect(db_path)
        rows = con.execute("SELECT surah, ayah, pos, text_clean FROM words "
                           "ORDER BY surah, ayah, pos").fetchall()
        con.close()

        self.ayahs, self.words = [], []          # aid -> (surah, ayah), aid -> [clean words]
        last = None
        for s, a, _p, t in rows:
            if (s, a) != last:
                self.ayahs.append((s, a))
                self.words.append([])
                last = (s, a)
            w = normalize(t)
            if w:
                self.words[-1].append(w)
        self.N = len(self.ayahs)

        post = collections.defaultdict(set)
        for aid, ws in enumerate(self.words):
            for w in ws:
                post[w].add(aid)
        self.post = {w: sorted(v) for w, v in post.items()}
        self.df = {w: len(v) for w, v in self.post.items()}
        self.idf = {w: math.log((self.N + 1) / (d + 0.5)) for w, d in self.df.items()}
        vals = sorted(self.idf.values())
        self.default_idf = vals[len(vals) // 2] if vals else 5.0

        self.tri = collections.defaultdict(list)
        for w in self.post:
            for t in _tri(w):
                self.tri[t].append(w)

        self._ncache = {}
        self._wsim = lru_cache(maxsize=None)(self._wsim_raw)

    # ---------- similarity / neighbours ----------
    def _wsim_raw(self, a, b):
        s = similarity(a, b)
        sa, sb = skeleton(a), skeleton(b)
        if len(sa) >= 2 and len(sb) >= 2:
            s = max(s, 0.95 * similarity(sa, sb))
        return s

    def neighbors(self, q):
        r = self._ncache.get(q)
        if r is not None:
            return r
        cnt = collections.Counter()
        for t in _tri(q):
            for v in self.tri.get(t, ()):
                cnt[v] += 1
        best = []
        for v, _c in cnt.most_common(self.cand_pool):
            if abs(len(v) - len(q)) > 3:
                continue
            s = self._wsim(q, v)
            if s >= self.min_sim:
                best.append((v, s))
        best.sort(key=lambda x: -x[1])
        r = best[:self.max_neigh]
        self._ncache[q] = r
        return r

    # ---------- stage 1: voting ----------
    def _vote(self, qwords, qidf, skip_common):
        scores = collections.defaultdict(float)
        for q, qi in zip(qwords, qidf):
            best = {}
            for v, s in self.neighbors(q):
                if skip_common and self.df[v] > self.common_df:
                    continue
                w = (s * s) * min(self.idf[v], qi)   # a rarer fuzzy neighbour must not beat the exact word
                for a in self.post[v]:
                    if w > best.get(a, 0.0):
                        best[a] = w
            for a, w in best.items():
                scores[a] += w
        return scores

    # ---------- stage 2: order-aware local alignment ----------
    def _align(self, q, qidf, ref):
        n, m = len(q), len(ref)
        rid = [self.idf.get(w, self.default_idf) for w in ref]
        prev = [(0.0, None)] * (m + 1)                     # free start anywhere in the ayah
        for i in range(1, n + 1):
            ins = 0.35 * qidf[i - 1]                        # spoken word with no match
            cur = [None] * (m + 1)
            cur[0] = (prev[0][0] - ins, prev[0][1])
            for j in range(1, m + 1):
                best = (prev[j][0] - ins, prev[j][1])
                c = cur[j - 1]
                cand = (c[0] - 0.25 * rid[j - 1], c[1])     # reference word skipped
                if cand[0] > best[0]:
                    best = cand
                s = self._wsim(q[i - 1], ref[j - 1])
                if s >= self.min_sim:
                    p = prev[j - 1]
                    cand = (p[0] + min(rid[j - 1], qidf[i - 1]) * s * s, p[1] if p[1] is not None else j - 1)
                    if cand[0] > best[0]:
                        best = cand
                cur[j] = best
            prev = cur
        F, start = max(prev, key=lambda x: x[0])           # free end
        return F, start

    # ---------- public ----------
    BASMALA = ["بسم", "الله", "الرحمن", "الرحيم"]

    def _strip_basmala(self, qwords):
        """Drop a spoken basmala when real words follow it (>= 2 more words)."""
        if len(qwords) >= 6 and all(self._wsim(qwords[i], self.BASMALA[i]) >= 0.7 for i in range(4)):
            return qwords[4:], True
        return qwords, False

    def find(self, text, topn=5, partial=False):
        """partial=True while the utterance is still streaming: the last word is often incomplete,
        so it is trusted less."""
        qwords = [w for w in (normalize(t) for t in text.split()) if w]
        qwords, stripped = self._strip_basmala(qwords)
        out = {"n_words": len(qwords), "results": [], "margin": 0.0, "confident": False,
               "basmala_stripped": stripped}
        if not qwords:
            return out

        qidf = []
        for q in qwords:
            nb = self.neighbors(q)
            qidf.append(self.idf[nb[0][0]] if nb else self.default_idf)
        if partial:
            qidf[-1] *= 0.5
        denom = sum(qidf) or 1.0

        scores = self._vote(qwords, qidf, True) or self._vote(qwords, qidf, False)
        if not scores:
            return out
        cands = sorted(scores, key=scores.get, reverse=True)[:self.rerank_top]

        res = []
        for aid in cands:
            F, start = self._align(qwords, qidf, self.words[aid])
            if start is None or F <= 0:
                continue
            bonus = 1.12 if start == 0 else (1.05 if start == 1 else 1.0)
            s, a = self.ayahs[aid]
            res.append({"surah": s, "ayah": a, "start": start,
                        "score": F * bonus, "conf": max(0.0, min(1.0, F / denom))})
        res.sort(key=lambda r: -r["score"])
        out["results"] = res[:topn]
        if res:
            s1 = res[0]["score"]
            s2 = res[1]["score"] if len(res) > 1 else 0.0
            out["margin"] = (s1 - s2) / s1 if s1 > 0 else 0.0
            out["confident"] = res[0]["conf"] >= 0.5 and out["margin"] >= 0.15
        return out
