# passdiff.py - run from backend.   python passdiff.py A B
# Strict equivalence check of two bench runs (replay_out\rp_bench_A vs rp_bench_B): every pass of every tape must
# have the same transcript, events, commit count, cursor, fed count and letter-check scores. Timings are ignored.
# Use it to prove that a clean-up changed nothing (bench --compare only looks at the final paint).
import os, sys
from replay import find_tapes, read

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("len", "final", "text", "ev", "n", "cur", "fed", "cur0", "fed0", "lc", "lookback", "jump", "jumped", "short")


def passes(t):
    return [tuple(r.get(k) for k in KEYS) for r in read(t) if r["k"] == "pass"]


def main():
    a, b = sys.argv[1:3]
    ra = {os.path.basename(os.path.dirname(t)): t for t in find_tapes([os.path.join(HERE, "replay_out", "rp_bench_" + a)])}
    rb = {os.path.basename(os.path.dirname(t)): t for t in find_tapes([os.path.join(HERE, "replay_out", "rp_bench_" + b)])}
    same = diff = 0
    for k in sorted(set(ra) | set(rb)):
        if k not in ra or k not in rb:
            print("only in one run:", k); diff += 1; continue
        pa, pb = passes(ra[k]), passes(rb[k])
        if pa == pb:
            same += len(pa)
        else:
            first = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
            print("DIFF %s: pass %d of %d/%d" % (k, first, len(pa), len(pb))); diff += 1
    print("tapes %d | passes identical %d | tapes different %d -> %s" % (len(ra), same, diff, "IDENTICAL" if not diff else "CHANGED"))
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
