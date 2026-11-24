# tape_early_reds.py - run from backend.   python tape_early_reds.py TAPES...
# Read-only analysis of recorded sessions: every non-green word committed on a PARTIAL pass (before the final,
# beam-searched pass of its utterance) is checked against what the final transcript of that utterance says
# about the same word (the final text aligned from the utterance's starting cursor, same Aligner settings).
#   AGREE   : final pass also non-green          -> waiting would not have helped
#   FIXABLE : final pass says CORRECT            -> a false red that waiting for the final pass would avoid
# Also counts non-green words committed on final passes. ASCII console; Arabic in early_reds.txt.
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay import find_tapes, read   # noqa: E402


def main():
    tapes = find_tapes(sys.argv[1:] or [os.path.join(HERE, "tapes")])
    import server                      # for load_surah / LENIENT / Aligner (no model loaded)
    from alignment import Aligner
    tot = dict(early=0, agree=0, fixable=0, unknown=0, final_red=0, early_ok=0)
    det = []
    for tp in tapes:
        recs = read(tp)
        surah, strict = None, False
        utt = []
        for r in recs:
            if r["k"] == "in" and r["m"].get("event") == "start":
                surah, strict = int(r["m"]["surah_id"]), bool(r["m"].get("strict", False))
            elif r["k"] == "pass":
                utt.append(r)
                if not r["final"]:
                    continue
                words = server.load_surah(surah)
                fin = utt[-1]
                for p in utt:
                    com = p["ev"][:p["n"]]
                    for gi, st, heard, ref, _spk in com:
                        if p is fin:
                            tot["final_red"] += st != "CORRECT" and ref is not None
                            continue
                        if st == "CORRECT":
                            tot["early_ok"] += 1
                            continue
                        if ref is None:          # extra word: no reference word to compare
                            continue
                        tot["early"] += 1
                        al = Aligner(words, utt[0]["cur0"], strict=strict, lenient=server.LENIENT)
                        fs = {e["matched_word_index"]: e["status"] for e in al.align(fin.get("text") or "")
                              if e["_ref"] is not None}
                        v = fs.get(gi)
                        kind = "unknown" if v is None else ("fixable" if v == "CORRECT" else "agree")
                        tot[kind] += 1
                        det.append("%s %-7s word %d %s | partial %s heard %s | final says %s | final text: %s" % (
                            os.path.basename(tp), kind.upper(), gi, words[gi]["text_uthmani"], st, heard, v, fin.get("text")))
                utt = []
    print("tapes %d | words committed green on partial passes %d" % (len(tapes), tot["early_ok"]))
    print("non-green committed on PARTIAL passes: %d  -> final pass agrees %d | final says CORRECT (fixable) %d | "
          "word not reached by final %d" % (tot["early"], tot["agree"], tot["fixable"], tot["unknown"]))
    print("non-green committed on FINAL passes  : %d" % tot["final_red"])
    open(os.path.join(HERE, "replay_out", "early_reds.txt"), "w", encoding="utf-8").write("\n".join(det) + "\n")
    print("details: replay_out\\early_reds.txt (UTF-8)")


if __name__ == "__main__":
    main()
