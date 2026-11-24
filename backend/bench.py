# bench.py - run from backend, uvicorn STOPPED (uses the GPU).   One score card for the whole system.
#   python bench.py NAME [--asr] [--set KEY=VALUE ...]      run + score   -> replay_out\bench_NAME\
#   python bench.py --score NAME                              re-score an existing run (no replay)
#   python bench.py --compare A B                             two cards side by side + every word that changed
# Every tape in bench_labels.json is replayed through the CURRENT code with default settings (+ --set only;
# knob variables in this shell are ignored). --asr re-transcribes the audio with the model (tests ASR changes,
# ~20 min); default text mode reuses the recorded transcripts (tests everything after the ASR, a few minutes).
# Card: CORRECT RECITATION = false flags (lower is better); MISTAKES = deliberate mistakes caught (higher is
# better); SPEED. A change is kept only if no line gets worse.
# Files: card.txt (ASCII), words.jsonl (one line per scored word), details.txt (UTF-8, Arabic; with --compare).
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "replay_out")
PY = sys.executable
GREEN, ORANGE, RED, SKIP = "CORRECT", "MISTAKE_UNSURE", "MISTAKE_SUBSTITUTION", "MISTAKE_SKIPPED"


def labels():
    return json.load(open(os.path.join(HERE, "bench_labels.json"), encoding="utf-8"))


def run(name, asr, sets):
    from replay import find_tapes, read
    L = labels()
    tapes = []
    for p in L["tapes"]:
        tapes += [t for t in find_tapes([os.path.join(HERE, p)])]
    import re
    src_txt = "".join(open(os.path.join(HERE, f), encoding="utf-8").read() for f in ("server.py", "alignment.py"))
    knobs = set(re.findall(r'(?:getenv|environ\.get)\(\s*"([A-Z][A-Z0-9_]+)"', src_txt))
    env = {k: v for k, v in os.environ.items() if k not in knobs}   # current-code defaults, not this shell
    for k, v in L.get("profile", {}).items():           # the production profile (what the app really runs with)
        if not k.startswith("_"):
            env[k] = os.path.join(HERE, v) if k == "CT2_DIR" else v
    env["PYTHONUTF8"] = "1"
    cmd = [PY, os.path.join(HERE, "replay.py"), "bench_" + name] + tapes + ["--cur-env"]
    cmd += ["--asr"] if asr else []
    for s in sets:
        cmd += ["--set", s]
    print("replaying %d tapes (%s mode)%s ..." % (len(tapes), "asr" if asr else "text", " --set " + " ".join(sets) if sets else ""))
    r = subprocess.run(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = [l for l in r.stdout.splitlines() if l.startswith("TOTAL") or "Traceback" in l or "Error" in l]
    print("\n".join(tail[-3:]))
    od = os.path.join(OUT, "bench_" + name)
    os.makedirs(od, exist_ok=True)
    json.dump(dict(asr=asr, sets=sets, tapes=len(tapes)), open(os.path.join(od, "run.json"), "w"))
    return r.returncode


def score(name):
    from replay import find_tapes, read, paints
    sys.path.insert(0, HERE)
    import sqlite3
    con = sqlite3.connect(os.path.join(HERE, "quran.db"))
    L = labels()
    src = {}                                    # original tape basename -> (group, set of mistake (s,a,p))
    for p, v in L["tapes"].items():
        for t in find_tapes([os.path.join(HERE, p)]):
            src[os.path.basename(t)] = (v["group"], {tuple(int(x) for x in m.split(":")) for m in v["mistakes"]})
    rp = os.path.join(OUT, "rp_bench_" + name)
    od = os.path.join(OUT, "bench_" + name)
    meta = json.load(open(os.path.join(od, "run.json")))
    W = {}
    def word(s, i):
        if s not in W:
            W[s] = con.execute("SELECT ayah, pos, text_uthmani FROM words WHERE surah=? ORDER BY ayah, pos", (s,)).fetchall()
        return W[s][i]
    rows, speed = [], dict(asr=[], align=[], lc=[], lag=[], early=0, total=0)
    for t in find_tapes([rp]):
        orig = os.path.basename(os.path.dirname(t))
        if orig not in src:
            continue
        group, mist = src[orig]
        recs = read(t)
        surah = next(r["m"]["surah_id"] for r in recs if r["k"] == "in" and r["m"].get("event") == "start")
        p, ins = paints(recs)
        if p:
            lo, hi = min(p), max(p)
            for i in range(lo, hi + 1):
                a, pos, txt = word(surah, i)
                st = p[i][0] if i in p else None
                rows.append(dict(tape=orig, group=group, surah=surah, ayah=a, pos=pos, idx=i, text=txt,
                                 heard=p[i][2] if i in p else None, status=st, mistake=(surah, a, pos) in mist))
            for (s_, a_, p_) in mist:
                if not any(r["tape"] == orig and (r["ayah"], r["pos"]) == (a_, p_) for r in rows):
                    rows.append(dict(tape=orig, group=group, surah=s_, ayah=a_, pos=p_, idx=None, text=None,
                                     heard=None, status=None, mistake=True))
        rows.append(dict(tape=orig, group=group, extra=ins))
        seen, u = {}, 0                          # speed: compute time + commit lag (first seen -> committed)
        for r in recs:
            if r["k"] == "pass":
                if r.get("asr_ms") is not None: speed["asr"].append(r["asr_ms"])
                if r.get("align_ms") is not None: speed["align"].append(r["align_ms"])
                if "lc_ms" in r: speed["lc"].append(r["lc_ms"])
                for e in r["ev"]:
                    if e[3] is not None and (u, e[0]) not in seen: seen[(u, e[0])] = r["len"]
                for e in r["ev"][:r["n"]]:
                    if e[1] != "MISTAKE_INSERTION" and (u, e[0]) in seen:
                        speed["lag"].append((r["len"] - seen.pop((u, e[0]))) / 32000.0)
                        speed["total"] += 1; speed["early"] += 0 if r["final"] else 1
            elif r["k"] in ("wav", "drop", "error"):
                u += 1; seen = {}
    with open(os.path.join(od, "words.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    card = make_card(name, meta, rows, speed)
    open(os.path.join(od, "card.txt"), "w", encoding="ascii", errors="replace").write(card)
    print(card)
    return rows


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0.0


def make_card(name, meta, rows, speed):
    words = [r for r in rows if "extra" not in r]
    groups = ["sim199", "live", "sessB", "sessC"]
    L = ["BENCH %s | mode %s | settings %s | tapes %d" % (name, "asr" if meta["asr"] else "text",
                                                         " ".join(meta["sets"]) or "profile", meta["tapes"]), ""]
    L.append("CORRECT RECITATION (false flags, lower is better)")
    L.append("  %-7s %6s %7s %7s %5s %5s %9s %6s" % ("set", "words", "green%", "orange", "red", "skip", "unpainted", "extra"))
    def crow(g, ws, ex):
        n = len(ws)
        c = lambda s: sum(1 for r in ws if r["status"] == s)
        return "  %-7s %6d %6.2f%% %7d %5d %5d %9d %6d" % (g, n, 100.0 * c(GREEN) / max(n, 1), c(ORANGE), c(RED), c(SKIP), c(None), ex)
    for g in groups + ["ALL"]:
        ws = [r for r in words if not r["mistake"] and (g == "ALL" or r["group"] == g)]
        ex = sum(r["extra"] for r in rows if "extra" in r and (g == "ALL" or r["group"] == g))
        L.append(crow(g, ws, ex))
    L.append("")
    L.append("DELIBERATE MISTAKES (caught = red or orange, higher is better)")
    L.append("  %-7s %4s %7s %5s %7s %6s %9s" % ("set", "n", "caught", "red", "orange", "green", "unpainted"))
    for g in ["live", "sessB", "sessC", "ALL"]:
        ms = [r for r in words if r["mistake"] and (g == "ALL" or r["group"] == g)]
        c = lambda s: sum(1 for r in ms if r["status"] == s)
        L.append("  %-7s %4d %7d %5d %7d %6d %9d" % (g, len(ms), c(RED) + c(ORANGE) + c(SKIP), c(RED), c(ORANGE), c(GREEN), c(None)))
    L.append("")
    L.append("SPEED")
    L.append("  asr ms %s | align ms p95 %.1f | letter check ms p50 %.0f p95 %.0f (%d passes)" % (
        "p50 %.0f p95 %.0f" % (q(speed["asr"], .5), q(speed["asr"], .95)) if meta["asr"] else "n/a (text mode)",
        q(speed["align"], .95), q(speed["lc"], .5), q(speed["lc"], .95), len(speed["lc"])))
    L.append("  commit lag (audio s from first seen to painted) p50 %.2f p95 %.2f | painted before the pause %.1f%%" % (
        q(speed["lag"], .5), q(speed["lag"], .95), 100.0 * speed["early"] / max(speed["total"], 1)))
    return "\n".join(L) + "\n"


def compare(a, b):
    def load(n):
        return [json.loads(l) for l in open(os.path.join(OUT, "bench_" + n, "words.jsonl"), encoding="utf-8")]
    ca = open(os.path.join(OUT, "bench_" + a, "card.txt"), encoding="ascii").read().splitlines()
    cb = open(os.path.join(OUT, "bench_" + b, "card.txt"), encoding="ascii").read().splitlines()
    print("A = %s\nB = %s\n" % (a, b))
    for x, y in zip(ca, cb):
        print("A " + x)
        if y != x:
            print("B " + y)
    key = lambda r: (r["tape"], r["surah"], r["ayah"], r["pos"])
    wa = {key(r): r for r in load(a) if "extra" not in r}
    wb = {key(r): r for r in load(b) if "extra" not in r}
    D, better, worse = [], 0, 0
    for k in sorted(set(wa) | set(wb), key=str):
        x, y = wa.get(k), wb.get(k)
        sx, sy = x and x["status"], y and y["status"]
        if sx == sy:
            continue
        m = (x or y)["mistake"]
        good = (sy not in (GREEN, None)) if m else (sy == GREEN)
        bad = (sx not in (GREEN, None)) if m else (sx == GREEN)
        better += good and not bad; worse += bad and not good
        D.append("%s %s %d:%d w%d %s | %s -> %s | heard %s -> %s" % (
            "MISTAKE" if m else "correct", k[0], k[1], k[2], k[3], (x or y)["text"], sx, sy,
            x and x["heard"], y and y["heard"]))
    open(os.path.join(OUT, "bench_" + b, "details.txt"), "w", encoding="utf-8").write("\n".join(D) + "\n")
    print("\nwords changed %d: better %d, worse %d  (list: replay_out\\bench_%s\\details.txt)" % (len(D), better, worse, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--score", metavar="NAME")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    if a.compare:
        compare(*a.compare); return 0
    if a.score:
        score(a.score); return 0
    if not a.name:
        ap.print_help(); return 1
    rc = run(a.name, a.asr, a.set)
    if rc:
        print("replay failed (exit %d)" % rc); return rc
    score(a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
