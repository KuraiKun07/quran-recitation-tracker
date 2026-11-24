# aligner_eval.py - run from backend.   python aligner_eval.py [--outputs model_outputs.jsonl]
# For every recorded clip: feed the model's FINAL transcript to the real alignment.Aligner against
# that ayah's reference words and count what the live tracker would show on a correct recitation.
# Not applied: the server's short-clip rule and multi-pass commit logic. Expect a clean clip when
# every reference word comes back CORRECT.
# Console is ASCII only; Arabic goes to aligner_eval_report.txt.
import argparse, collections, json, os, sqlite3

from alignment import Aligner, public
from trainutil import HERE, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="model_outputs.jsonl", help="from baseline.py (--tag ...)")
    args = ap.parse_args()
    rows = {r["file"]: r for r in load_manifest()}
    outs = [json.loads(l) for l in open(os.path.join(HERE, args.outputs), encoding="utf-8") if l.strip()]
    con = sqlite3.connect(os.path.join(HERE, "quran.db"))
    status = collections.Counter()
    by_kind = collections.defaultdict(lambda: [0, 0])      # kind -> [clean clips, clips]
    report = []
    for o in outs:
        r = rows.get(o["file"])
        if not r:
            continue
        words = [dict(ayah_number=a, word_position=p, text_uthmani=u, text_clean=c, page=pg, line=ln)
                 for a, p, u, c, pg, ln in con.execute(
                     "SELECT ayah, pos, text_uthmani, text_clean, page, line FROM words "
                     "WHERE surah=? AND ayah=? ORDER BY pos", (r["surah"], r["ayah"]))]
        ev = [public(e) for e in Aligner(words, 0, window=max(10, len(words))).feed(o["out"])]   # window covers the whole ayah: no artificial insertions
        st = collections.Counter(e.get("status") for e in ev)
        status.update(st)
        clean = st.get("CORRECT", 0) == len(words) and sum(st.values()) == st.get("CORRECT", 0)
        by_kind[r["kind"]][0] += clean
        by_kind[r["kind"]][1] += 1
        if not clean:
            report.append("%d:%d [%s] %d ref words | %s\n  heard: %s" % (
                r["surah"], r["ayah"], r["kind"], len(words), dict(st), o["out"]))
            for e in ev:
                if e.get("status") != "CORRECT":
                    report.append("    " + json.dumps(e, ensure_ascii=False))
    n = sum(v[1] for v in by_kind.values())
    c = sum(v[0] for v in by_kind.values())
    print("clips fully clean under the real Aligner: %d of %d (%.0f%%)" % (c, n, 100 * c / max(1, n)))
    print("by kind:", {k: "%d/%d" % (v[0], v[1]) for k, v in by_kind.items()})
    print("event statuses:", dict(status))
    with open(os.path.join(HERE, "aligner_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("Arabic details: backend\\aligner_eval_report.txt")


if __name__ == "__main__":
    main()
