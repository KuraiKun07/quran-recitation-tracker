# rescore.py - re-score existing outputs with a tolerant metric. Run from backend.
#   python rescore.py                      # BASELINE: real errors of the production model, all clips
#   python rescore.py cv_B_lr3e-06         # a finished cv.py run (tag = the "===" name cv.py prints)
# "Tolerant" = a word the model wrote is accepted when its similarity to the reference word is
# >= 0.75 (1 - levenshtein/maxlen on normalised text), which is how I read the live aligner's
# CORRECT rule in your handoff. The real Aligner code was not available to me, so this is a proxy.
# Console is ASCII only; Arabic goes to real_errors_report.txt (baseline mode).
import collections, glob, json, os, re, sys

from trainutil import HERE, edit_distance, load_manifest, norm, tolerant_ops

THR = 0.75


def baseline(outs_name="model_outputs.jsonl"):
    _tag = os.path.basename(outs_name).replace("model_outputs", "").replace(".jsonl", "").strip("_")
    rep_name = "real_errors_report_" + _tag + ".txt" if _tag else "real_errors_report.txt"
    rows = {r["file"]: r for r in load_manifest()}
    outs = [json.loads(l) for l in open(os.path.join(HERE, outs_name), encoding="utf-8") if l.strip()]
    cnt, kinds, lines = collections.Counter(), collections.Counter(), []
    words = bad = 0
    for o in outs:
        r = rows.get(o["file"])
        if not r:
            continue
        ref, hyp = norm(r["clean"]).split(), norm(o["out"]).split()
        words += len(ref)
        ev = tolerant_ops(ref, hyp, THR)
        bad += bool(ev)
        for k, a, b in ev:
            cnt[k] += 1
            kinds[r["kind"]] += 1
            lines.append("%d:%d [%s] %s  ref=%s  heard=%s" % (r["surah"], r["ayah"], r["kind"], k, a, b))
    tot = sum(cnt.values())
    print("BASELINE, tolerant (similarity >= %.2f counts as correct)" % THR)
    print("words %d | real errors %d = %.1f%% (substitutions %d, deletions %d, insertions %d)"
          % (words, tot, 100 * tot / max(1, words), cnt["S"], cnt["D"], cnt["I"]))
    print("clips with a real error: %d of %d | by kind: %s" % (bad, len(outs), dict(kinds)))
    with open(os.path.join(HERE, rep_name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Arabic list: backend\\" + rep_name)


def parse_heldout(path):
    out, cur = [], {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line and not line.startswith(" "):
            if cur:
                out.append(cur)
            cur = {}
        elif line.startswith("  clean :"):
            cur["clean"] = line.split(":", 1)[1].strip()
        elif line.startswith("  beam4 :"):
            cur["beam4"] = line.split(":", 1)[1].strip()
        elif line.startswith("  greedy:"):
            cur["greedy"] = line.split(":", 1)[1].strip()
    if cur:
        out.append(cur)
    return out


def cv(base):
    dirs = sorted(glob.glob(os.path.join(HERE, "runs", base + "_f*")))
    if not dirs:
        print("no runs named %s_f* under runs\\" % base)
        return
    per = {}      # epoch -> fold -> dict of counts
    for d in dirs:
        for f in glob.glob(os.path.join(d, "heldout_ep*.txt")):
            e = int(re.search(r"heldout_ep(\d+)\.txt", f).group(1))
            c = {"words": 0, "s4": 0, "t4": 0, "sg": 0, "tg": 0}
            for x in parse_heldout(f):
                ref = x["clean"].split()
                c["words"] += len(ref)
                for key, name in (("beam4", "4"), ("greedy", "g")):
                    hyp = x.get(key, "").split()
                    c["s" + name] += edit_distance(ref, hyp)
                    c["t" + name] += len(tolerant_ops(ref, hyp, THR))
            per.setdefault(e, {})[os.path.basename(d)] = c
    print("=== %s | tolerant metric (similarity >= %.2f counts as correct); strict in brackets" % (base, THR))
    for e in sorted(per):
        folds = [k for k in per[e] if k in per.get(0, {})]
        if not folds:
            continue
        w = sum(per[e][k]["words"] for k in folds)
        t4, tg = sum(per[e][k]["t4"] for k in folds), sum(per[e][k]["tg"] for k in folds)
        s4, sg = sum(per[e][k]["s4"] for k in folds), sum(per[e][k]["sg"] for k in folds)
        line = "epoch %d: beam4 %3d/%d = %.1f%% [strict %d] | greedy %3d/%d = %.1f%% [strict %d]" % (
            e, t4, w, 100 * t4 / w, s4, tg, w, 100 * tg / w, sg)
        if e > 0:
            d = [(per[e][k]["t4"] + per[e][k]["tg"]) - (per[0][k]["t4"] + per[0][k]["tg"]) for k in folds]
            line += " | folds better/same/worse: %d/%d/%d" % (sum(x < 0 for x in d), sum(x == 0 for x in d), sum(x > 0 for x in d))
        else:
            line += "   <- production"
        print(line)


if __name__ == "__main__":
    (baseline(sys.argv[1]) if sys.argv[1].endswith(".jsonl") else cv(sys.argv[1])) if len(sys.argv) > 1 else baseline()
