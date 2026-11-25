# tape_show.py - run from backend.   python tape_show.py TAPE_FOLDER [OUT.txt]
# Pass-by-pass view of one recorded session: transcript, events (word:status heard), what was committed.
# Writes UTF-8 (Arabic) to OUT.txt (default replay_out\tape_show.txt); open it in VS Code.
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay import find_tapes, read   # noqa: E402

AB = {"CORRECT": "ok", "MISTAKE_SUBSTITUTION": "SUB", "MISTAKE_SKIPPED": "DEL", "MISTAKE_INSERTION": "INS",
      "MISTAKE_UNSURE": "UNS"}


def show(tp):
    out = []
    for r in read(tp):
        k = r["k"]
        if k == "in":
            out.append("%7.3f IN   %s" % (r["t"], r["m"]))
        elif k == "pass":
            ev = " ".join("%s%d:%s[%s]" % ("*" if i < r["n"] else "", e[0], AB.get(e[1], e[1]), e[2] or "")
                          for i, e in enumerate(r["ev"]))
            flags = "".join(x for x, c in (("L", r.get("lookback") is not None), ("J", r.get("jumped")),
                                          ("S", r.get("short"))) if c)
            out.append("%7.3f PASS len %5.2fs %s cur %d->%d fed %s->%s n=%d %s asr %s ms | %s\n          %s" % (
                r["t"], r["len"] / 32000, "FINAL" if r["final"] else "part ", r.get("cur0", -1), r["cur"],
                r.get("fed0"), r["fed"], r["n"], flags, r.get("asr_ms"), r.get("text", "(no asr)"), ev))
        elif k in ("wav", "drop", "error"):
            out.append("%7.3f %s %s" % (r["t"], k.upper(), {x: y for x, y in r.items() if x not in ("k", "t")}))
        elif k == "out" and r["m"].get("event") in ("alignment_result", "recheck", "jump"):
            m = r["m"]
            out.append("%7.3f  -> %s %s %s" % (r["t"], m.get("event") if m.get("event") == "jump" else AB.get(m["status"]),
                                               m.get("matched_word_index", m.get("to_index")), m.get("spoken_transcription") or ""))
    return out


def main():
    tps = find_tapes([sys.argv[1]])
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "replay_out", "tape_show.txt")
    lines = []
    for tp in tps:
        lines += ["==== %s" % tp] + show(tp)
    open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("written: %s (%d lines)" % (dst, len(lines)))


if __name__ == "__main__":
    main()
