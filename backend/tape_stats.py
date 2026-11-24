# tape_stats.py - run from backend.   python tape_stats.py [TAPES ...]      (default: tapes)
# One ASCII summary per recorded session (TAPE=1) and a total: passes, ASR and alignment time per pass,
# paints by colour, extra words, jumps, look-backs, dropped clicks, errors. Read-only.
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay import find_tapes, paints, read   # noqa: E402


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else "-"


def stats(recs):
    P = [r for r in recs if r["k"] == "pass"]
    asr = [p["asr_ms"] for p in P if p.get("asr_ms") is not None]
    aln = [p["align_ms"] for p in P if p.get("align_ms") is not None]
    w, ins = paints(recs)
    col = {}
    for st, _e, _h in w.values():
        col[st] = col.get(st, 0) + 1
    return dict(passes=len(P), asr=asr, aln=aln, col=col, ins=ins,
                jumps=sum(1 for p in P if p.get("jumped")), lookbacks=sum(1 for p in P if p.get("lookback") is not None),
                drops=sum(1 for r in recs if r["k"] == "drop"), errors=sum(1 for r in recs if r["k"] == "error"),
                utts=sum(1 for r in recs if r["k"] == "wav"),
                beams=recs[0].get("final_beams"), step=recs[0].get("step_bytes"))


def line(name, s):
    c = s["col"]
    return ("%-24s utt %3d pass %4d | asr ms p50 %4s p95 %4s max %4s | align ms p50 %5s p95 %5s max %5s | "
            "ok %d sub %d uns %d skip %d extra %d | jump %d lookback %d drop %d err %d" % (
                name, s["utts"], s["passes"], pct(s["asr"], .5), pct(s["asr"], .95), max(s["asr"], default="-"),
                pct(s["aln"], .5), pct(s["aln"], .95), max(s["aln"], default="-"),
                c.get("CORRECT", 0), c.get("MISTAKE_SUBSTITUTION", 0), c.get("MISTAKE_UNSURE", 0),
                c.get("MISTAKE_SKIPPED", 0), s["ins"], s["jumps"], s["lookbacks"], s["drops"], s["errors"]))


def main():
    tapes = find_tapes(sys.argv[1:] or [os.path.join(HERE, "tapes")])
    if not tapes:
        print("no tapes found (start uvicorn with $env:TAPE=\"1\" first)"); return
    tot = None
    for t in tapes:
        s = stats(read(t))
        print(line(os.path.basename(t), s))
        if tot is None:
            tot = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in s.items()}
        else:
            for k, v in s.items():
                if isinstance(v, list): tot[k] += v
                elif isinstance(v, dict):
                    for kk, vv in v.items(): tot[k][kk] = tot[k].get(kk, 0) + vv
                elif isinstance(v, int) and k not in ("beams", "step"): tot[k] += v
    if len(tapes) > 1:
        print(line("TOTAL (%d sessions)" % len(tapes), tot))


if __name__ == "__main__":
    main()
