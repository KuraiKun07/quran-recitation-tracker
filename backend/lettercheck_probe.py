# lettercheck_probe.py - run from backend, uvicorn STOPPED (loads the model on the GPU).
#   python lettercheck_probe.py NAME TAPES... [--mistake SURAH:AYAH:POS ...] [--no-variants]
# Measures, for every word COMMITTED in the tapes, how the ASR model itself scores the expected word against
#   (h) what the ASR wrote for it  : dh = logP(transcript) - logP(transcript with that token = expected word)
#   (v) one-letter confusions       : dv = max over variants logP(variant) - logP(expected)   (ha/Ha, s/S, q/k, ...)
# using the same forced decoding as forced_check (same audio slice and transcript as the live pass).
# --mistake marks the words that were recited WRONG on purpose (all other words are taken as recited correctly).
# Output: replay_out\lc_NAME\  words.jsonl (one line per committed word), summary.txt (ASCII), details.txt (UTF-8).
import argparse, json, os, sys, time, wave

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "replay_out")

# learner letter confusions (each letter -> letters it is commonly replaced with)
CONF = {
    "ح": "ه",            # Ha -> ha
    "ه": "ح",            # ha -> Ha
    "ع": "أ",            # ain -> hamza
    "أ": "ع", "إ": "ع", "ء": "ع",
    "ق": "ك",            # qaf -> kaf
    "ك": "ق",
    "ص": "س",            # Sad -> sin
    "س": "صث",      # sin -> Sad, tha
    "ض": "دظ",      # Dad -> dal, Za
    "د": "ضذ",      # dal -> Dad, dhal
    "ظ": "زضذ",
    "ط": "ت",            # Ta -> ta
    "ت": "ط",
    "ث": "ست",      # tha -> sin, ta
    "ذ": "زد",      # dhal -> zay, dal
    "ز": "ذ",
    "غ": "خ",            # ghain -> kha
    "خ": "غح",
}
MAX_VARIANTS = 10


def variants(s):
    out = []
    for i, ch in enumerate(s):
        for alt in CONF.get(ch, ""):
            out.append(s[:i] + alt + s[i + 1:])
    return out[:MAX_VARIANTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("tapes", nargs="+")
    ap.add_argument("--mistake", action="append", default=[])
    ap.add_argument("--no-variants", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    os.environ.setdefault("PYTHONUTF8", "1")
    import numpy as np
    import server
    from replay import find_tapes, read
    from alignment import normalize, similarity
    fc = server._fc_setup()
    if fc.get("off"):
        print("forced decoding unavailable"); return 1
    m, tok = fc["m"], fc["tok"]
    fe = m.feature_extractor
    mist = set()
    for s in a.mistake:
        su, ay, po = (int(x) for x in s.split(":"))
        mist.add((su, ay, po))
    od = os.path.join(OUT, "lc_" + a.name)
    os.makedirs(od, exist_ok=True)
    words_cache = {}
    rows, t_lp, n_lp = [], 0.0, 0

    def lp(enc, nf, toks):
        nonlocal t_lp, n_lp
        ids = tok.encode(" " + " ".join(toks))
        if len(ids) + len(fc["prefix"]) > 440:
            return None
        t = time.time()
        res = m.model.align(enc, fc["prefix"], [ids], nf)
        t_lp += time.time() - t; n_lp += 1
        return sum(np.log(max(x, 1e-9)) for x in list(res[0].text_token_probs))

    tapes = find_tapes(a.tapes)
    for ti, tp in enumerate(tapes):
        recs = read(tp)
        surah, buf = None, []
        for r in recs:
            if r["k"] == "in" and r["m"].get("event") == "start":
                surah = r["m"]["surah_id"]
            elif r["k"] == "pass":
                buf.append(r)
            elif r["k"] in ("drop", "error"):
                buf = []
            elif r["k"] == "wav":
                with wave.open(os.path.join(tp, r["file"]), "rb") as wf:
                    pcm = wf.readframes(wf.getnframes())
                if surah not in words_cache:
                    words_cache[surah] = server.load_surah(surah)
                W = words_cache[surah]
                for p in buf:
                    com = [e for e in p["ev"][:p["n"]] if e[1] in ("CORRECT", "MISTAKE_SUBSTITUTION", "MISTAKE_UNSURE")]
                    if not com or not p.get("text"):
                        continue
                    raw = p["text"].split()
                    rest = raw[p["fed0"]:]
                    pos, q0 = {}, 0
                    for i, e in enumerate(p["ev"]):          # same token mapping as forced_check
                        sp = e[2]
                        if sp is None:
                            continue
                        q = q0
                        while q < len(rest) and rest[q] != sp:
                            q += 1
                        if q < len(rest):
                            pos[i] = q; q0 = q + 1
                    todo = [i for i, e in enumerate(p["ev"][:p["n"]]) if e in com and i in pos]
                    if not todo:
                        continue
                    audio = np.frombuffer(pcm[:p["len"]], np.int16).astype(np.float32) / 32768.0
                    enc = m.encode(fc["pad"](fe(audio)[:, :fe.nb_max_frames]))
                    nf = min(int(len(audio) / fe.hop_length), fe.nb_max_frames)
                    base = lp(enc, nf, raw)
                    if base is None:
                        continue
                    for i in todo:
                        e = p["ev"][i]
                        w = W[e[0]]
                        key = normalize(w["text_clean"])
                        heard = normalize(e[2])
                        exp = server._fc_style(w["text_uthmani"])
                        k = p["fed0"] + pos[i]
                        t2 = list(raw); t2[k] = exp
                        le = lp(enc, nf, t2)
                        if le is None:
                            continue
                        dv, best = None, None
                        if not a.no_variants:
                            for v in variants(exp):
                                t3 = list(raw); t3[k] = v
                                lv = lp(enc, nf, t3)
                                if lv is not None and (dv is None or lv - le > dv):
                                    dv, best = lv - le, v
                        lab = (surah, w["ayah_number"], w["word_position"]) in mist
                        rows.append(dict(tape=os.path.basename(tp), surah=surah, ayah=w["ayah_number"],
                                         pos=w["word_position"], idx=e[0], status=e[1], final=p["final"],
                                         expected=w["text_uthmani"], heard=e[2], exact=heard == key,
                                         sim=round(similarity(heard, key), 3), dh=round(base - le, 3),
                                         dv=None if dv is None else round(dv, 3), best=best, mistake=lab))
                buf = []
        print("tape %d/%d %s  words %d" % (ti + 1, len(tapes), os.path.basename(tp), len(rows)))

    with open(os.path.join(od, "words.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    L = []
    pos_ = [r for r in rows if r["mistake"]]
    neg = [r for r in rows if not r["mistake"]]
    L.append("committed words %d (mistakes %d, correct %d) | forced decodes %d, %.1f ms each" % (
        len(rows), len(pos_), len(neg), n_lp, 1000 * t_lp / max(n_lp, 1)))
    L.append("")
    L.append("MISTAKES (recited wrong on purpose):")
    for r in pos_:
        L.append("  %3d:%-3d w%-2d %-6s final=%d exact=%d sim %.2f  dh %7.2f  dv %s" % (
            r["surah"], r["ayah"], r["pos"], r["status"][8:11] if r["status"] != "CORRECT" else "OK", r["final"],
            r["exact"], r["sim"], r["dh"], "%7.2f" % r["dv"] if r["dv"] is not None else "   -"))

    def table(title, vals):
        L.append(title)
        vals = sorted(vals)
        if not vals:
            L.append("  (none)"); return
        for t in (1, 2, 3, 4, 5, 6, 8, 10):
            L.append("  > %2d : %4d of %4d" % (t, sum(v > t for v in vals), len(vals)))
        L.append("  max %.2f  p99 %.2f  p95 %.2f" % (vals[-1], vals[int(0.99 * (len(vals) - 1))], vals[int(0.95 * (len(vals) - 1))]))
    L.append("")
    table("CORRECT words, heard != expected, green now: dh (how much likelier the ASR spelling is)",
          [r["dh"] for r in neg if not r["exact"] and r["status"] == "CORRECT"])
    table("CORRECT words, red/orange now: dh (FORCED_CHECK accepts when dh <= FORCED_T=3)",
          [r["dh"] for r in neg if r["status"] != "CORRECT"])
    table("CORRECT words, all: dv (best one-letter confusion vs expected)",
          [r["dv"] for r in neg if r["dv"] is not None])
    open(os.path.join(od, "summary.txt"), "w", encoding="ascii", errors="replace").write("\n".join(L) + "\n")
    D = []
    for r in sorted(rows, key=lambda r: -(r["dv"] or -99)):
        D.append("%s %s %d:%d w%d %s | exp %s | heard %s | dh %.2f | dv %s %s" % (
            "MISTAKE" if r["mistake"] else "ok     ", r["tape"], r["surah"], r["ayah"], r["pos"], r["status"],
            r["expected"], r["heard"], r["dh"], r["dv"], r["best"] or ""))
    open(os.path.join(od, "details.txt"), "w", encoding="utf-8").write("\n".join(D) + "\n")
    print("\n".join(L))
    print("files: replay_out\\lc_%s\\ (summary.txt, details.txt = Arabic, words.jsonl)" % a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
