"""mistake_eval.py - inject synthetic reading mistakes into clean ep4 transcripts
and measure how many the real Aligner catches.

Run from backend:
    python mistake_eval.py
    python mistake_eval.py --outputs model_outputs_new_ep4.jsonl other_ep4_outputs.jsonl

Standalone: imports alignment.py, reads quran.db and the outputs file(s),
changes nothing. Console output is ASCII only (this file too); Arabic examples
go to mistake_eval_report.txt (UTF-8, open in VS Code).

Method
  - Only clips that are fully CLEAN under the default Aligner are used, so any
    flag on an undamaged clip is a false alarm and any flag on a damaged clip
    is caused by the damage.
  - Each clip is damaged ONE word at a time, at EVERY position, in seven ways:
      sub_same  : word replaced by a different word from the same transcript
      sub_rand  : word replaced by a word from another clip
      delete    : word removed (skipped word)
      repeat    : word said twice
      near_swap : ONE letter swapped for a confusable one (s/S, t/T, z/Z, k/q ...)
      near_drop : ONE letter removed from a word of 3+ letters
      near_add  : ONE letter (alef / waw / ya) added inside the word
  - Each damaged transcript is fed to the Aligner (window covers the ayah,
    same as aligner_eval.py).
  - hard   = at least one MISTAKE_SUBSTITUTION / SKIPPED / INSERTION event
    unsure = only MISTAKE_UNSURE events (shown doubtful, not wrong)
    missed = every event CORRECT (damage not noticed)
    extra  = more flagged events than the one damaged spot (collateral)
  - Settings compared: default (0.75), lenient, thresholds 0.60 0.70 0.80 0.85,
    strict.
  - The four original kinds use the same random draws as the first version of
    this script, so their numbers are reproducible.
"""
import argparse
import collections
import json
import os
import random
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from alignment import Aligner, normalize, public  # noqa: E402

FILE_RE = re.compile(r"s(\d{3})_a(\d{3})")
HARD = ("MISTAKE_SUBSTITUTION", "MISTAKE_SKIPPED", "MISTAKE_INSERTION")
CONFIGS = collections.OrderedDict([
    ("default", {}),
    ("lenient", dict(lenient=True)),
    ("thr0.60", dict(threshold=0.60)),
    ("thr0.70", dict(threshold=0.70)),
    ("thr0.80", dict(threshold=0.80)),
    ("thr0.85", dict(threshold=0.85)),
    ("strict", dict(strict=True)),
])
BASE_KINDS = ("sub_same", "sub_rand", "delete", "repeat")
NEAR_KINDS = ("near_swap", "near_drop", "near_add")
KINDS = BASE_KINDS + NEAR_KINDS
LENS = ("<=3", "4-5", ">=6")
EXAMPLE_CAP = 40  # missed examples kept per kind (default config)

# confusable letter pairs, as unicode escapes (keeps this file ASCII)
_PAIRS = [("0633", "0635"), ("062A", "0637"), ("0630", "0632"), ("0643", "0642"),
          ("062B", "0633"), ("062D", "062E"), ("0647", "062D"), ("0639", "063A"),
          ("062F", "0636"), ("0638", "0636")]
CONF = collections.defaultdict(list)
for _a, _b in _PAIRS:
    _a, _b = chr(int(_a, 16)), chr(int(_b, 16))
    CONF[_a].append(_b)
    CONF[_b].append(_a)
ADD_LETTERS = ["\u0627", "\u064A", "\u0648"]  # alef, ya, waw


def is_mark(ch):
    return "\u064b" <= ch <= "\u0652"


def ref_words(con, surah, ayah):
    return [dict(ayah_number=a, word_position=p, text_uthmani=u, text_clean=c, page=pg, line=ln)
            for a, p, u, c, pg, ln in con.execute(
                "SELECT ayah, pos, text_uthmani, text_clean, page, line FROM words "
                "WHERE surah=? AND ayah=? ORDER BY pos", (surah, ayah))]


def run(words, text, **kw):
    al = Aligner(words, 0, window=max(10, len(words)), **kw)
    return [public(e) for e in al.feed(text)]


def classify(ev):
    st = collections.Counter(e.get("status") for e in ev)
    noncorrect = sum(v for k, v in st.items() if k != "CORRECT")
    hard = sum(st.get(k, 0) for k in HARD)
    if hard:
        return "hard", noncorrect
    if noncorrect:
        return "unsure", noncorrect
    return "missed", 0


def near_variant(tok, kind, rng):
    letters = [j for j, ch in enumerate(tok) if not is_mark(ch)]
    if kind == "near_swap":
        cands = [j for j in letters if tok[j] in CONF]
        if not cands:
            return None
        j = rng.choice(cands)
        return tok[:j] + rng.choice(CONF[tok[j]]) + tok[j + 1:]
    if kind == "near_drop":
        if len(letters) < 3:
            return None
        j = rng.choice(letters)
        return tok[:j] + tok[j + 1:]
    if kind == "near_add":
        j = rng.choice(letters)
        return tok[:j + 1] + rng.choice(ADD_LETTERS) + tok[j + 1:]
    return None


def damage(kind, toks, i, pool, rng, rng2):
    cur = normalize(toks[i])
    if kind == "delete":
        return toks[:i] + toks[i + 1:]
    if kind == "repeat":
        return toks[:i + 1] + [toks[i]] + toks[i + 1:]
    if kind in NEAR_KINDS:
        new = near_variant(toks[i], kind, rng2)
        if new is None or normalize(new) == cur or not normalize(new):
            return None
        return toks[:i] + [new] + toks[i + 1:]
    if kind == "sub_same":
        cands = [t for t in toks if normalize(t) != cur]
    else:
        cands = [t for t in pool if normalize(t) != cur]
    if not cands:
        return None
    return toks[:i] + [rng.choice(cands)] + toks[i + 1:]


def len_class(tok):
    n = len(normalize(tok))
    return "<=3" if n <= 3 else ("4-5" if n <= 5 else ">=6")


def pct(n, d):
    return "%5.1f%%" % (100.0 * n / d) if d else "   n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", default=["model_outputs_new_ep4.jsonl"],
                    help="baseline.py output files made with the ep4 model")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rng2 = random.Random(args.seed + 1)  # near kinds only, keeps old kinds reproducible

    outs = []
    for fn in args.outputs:
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print("missing outputs file:", fn)
            sys.exit(1)
        outs += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    con = sqlite3.connect(os.path.join(HERE, "quran.db"))
    clips, pool, skipped = [], [], collections.Counter()
    for o in outs:
        m = FILE_RE.search(o["file"])
        toks = o["out"].split()
        pool += toks
        if not m:
            skipped["bad filename"] += 1
            continue
        s, a = int(m.group(1)), int(m.group(2))
        words = ref_words(con, s, a)
        if len(words) < args.min_words:
            skipped["short ayah"] += 1
            continue
        ev = run(words, o["out"])
        st = collections.Counter(e.get("status") for e in ev)
        if not (st.get("CORRECT", 0) == len(words) and sum(st.values()) == st.get("CORRECT", 0)):
            skipped["not clean"] += 1
            continue
        clips.append((o["file"], s, a, words, toks))

    print("clips read       :", len(outs))
    print("clean and used   :", len(clips))
    print("skipped          :", dict(skipped))
    if not clips:
        print("nothing to test")
        sys.exit(1)

    res = {c: {k: collections.defaultdict(collections.Counter) for k in KINDS} for c in CONFIGS}
    lens = {c: {k: collections.defaultdict(collections.Counter) for k in KINDS} for c in CONFIGS}
    undamaged = {c: [0, 0] for c in CONFIGS}  # flagged, total
    examples = collections.defaultdict(list)

    for fname, s, a, words, toks in clips:
        for cname, kw in CONFIGS.items():
            ev = run(words, " ".join(toks), **kw)
            undamaged[cname][1] += 1
            if any(e.get("status") != "CORRECT" for e in ev):
                undamaged[cname][0] += 1
        n = len(toks)
        for kind in KINDS:
            for i in range(n):
                dm = damage(kind, toks, i, pool, rng, rng2)
                if dm is None:
                    continue
                pc = "first" if i == 0 else ("last" if i == n - 1 else "mid")
                lc = len_class(toks[i])
                text = " ".join(dm)
                for cname, kw in CONFIGS.items():
                    ev = run(words, text, **kw)
                    verdict, nc = classify(ev)
                    for c in (res[cname][kind][pc], lens[cname][kind][lc]):
                        c["trials"] += 1
                        c[verdict] += 1
                        if nc > 1:
                            c["extra"] += 1
                    if (cname == "default" and verdict == "missed"
                            and len(examples[kind]) < EXAMPLE_CAP):
                        what = ("%s -> %s" % (toks[i], dm[i])) if kind != "delete" else toks[i]
                        examples[kind].append("%s %d:%d %s pos %d/%d (%s)\n  word: %s\n  damaged: %s" % (
                            kind, s, a, pc, i + 1, n, fname, what, text))

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    def agg(cname, kind):
        tot = collections.Counter()
        for pc in res[cname][kind].values():
            tot.update(pc)
        return tot

    emit("")
    emit("SUMMARY - % of damaged words NOT noticed at all (lower is better)")
    emit("false_alarm = % of undamaged clean clips that got any flag (lower is better)")
    head = "%-8s %11s" % ("setting", "false_alarm") + "".join(" %9s" % k for k in KINDS)
    emit(head)
    for cname in CONFIGS:
        f, t = undamaged[cname]
        row = "%-8s %11s" % (cname, pct(f, t).strip())
        for kind in KINDS:
            tot = agg(cname, kind)
            row += " %9s" % pct(tot["missed"], tot["trials"]).strip()
        emit(row)
    emit("")
    emit("SUMMARY - near-miss kinds: % flagged WRONG (hard) / % only doubtful (unsure)")
    emit("%-8s" % "setting" + "".join(" %15s" % k for k in NEAR_KINDS))
    for cname in CONFIGS:
        row = "%-8s" % cname
        for kind in NEAR_KINDS:
            tot = agg(cname, kind)
            row += " %15s" % ("%s/%s" % (pct(tot["hard"], tot["trials"]).strip(),
                                          pct(tot["unsure"], tot["trials"]).strip()))
        emit(row)
    emit("")
    emit("trials per kind (default): " + ", ".join(
        "%s %d" % (k, agg("default", k)["trials"]) for k in KINDS))

    for cname in CONFIGS:
        emit("")
        emit("== %s ==" % cname)
        emit("%-10s %7s %8s %8s %8s %8s" % ("damage", "trials", "hard", "unsure", "missed", "extra"))
        for kind in KINDS:
            tot = agg(cname, kind)
            t = tot["trials"]
            emit("%-10s %7d %8s %8s %8s %8s" % (
                kind, t, pct(tot["hard"], t), pct(tot["unsure"], t),
                pct(tot["missed"], t), pct(tot["extra"], t)))
        emit("missed by position (first / mid / last):")
        for kind in KINDS:
            parts = []
            for pc in ("first", "mid", "last"):
                c = res[cname][kind][pc]
                parts.append("%s %s (%d)" % (pc, pct(c["missed"], c["trials"]).strip(), c["trials"]))
            emit("  %-10s %s" % (kind, " | ".join(parts)))
        emit("near-miss missed by word length in letters (<=3 / 4-5 / >=6):")
        for kind in NEAR_KINDS:
            parts = []
            for lc in LENS:
                c = lens[cname][kind][lc]
                parts.append("%s %s (%d)" % (lc, pct(c["missed"], c["trials"]).strip(), c["trials"]))
            emit("  %-10s %s" % (kind, " | ".join(parts)))

    emit("")
    emit("hard = flagged wrong, unsure = flagged doubtful only, missed = not noticed,")
    emit("extra = more flagged events than the one damaged spot.")

    with open("mistake_eval_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\nMISSED EXAMPLES (default config, up to %d per kind):\n" % EXAMPLE_CAP)
        for kind in KINDS:
            fh.write("\n--- %s: %d shown ---\n\n" % (kind, len(examples[kind])))
            fh.write("\n".join(examples[kind]) + "\n")
    print("Missed examples written to mistake_eval_report.txt (open in VS Code).")


if __name__ == "__main__":
    main()
