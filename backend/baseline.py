# baseline.py - run from backend:  python baseline.py   (add --beam 1 for greedy)
# Runs the CURRENT CT2 model over every recorded clip (same decode settings as server.py's
# final pass), scores it against text_clean after the aligner's normalisation, and shows what
# characters the model writes (evidence for the training-target format).
# Console is ASCII only; Arabic goes to baseline_report.txt (UTF-8, open in VS Code).
import argparse, collections, json, os, time, unicodedata

import torch  # must come before faster_whisper: loads torch's bundled CUDA DLLs (cublas64_12) for CTranslate2,
              # the same way server.py's GPU probe does

from trainutil import HERE, edit_distance, is_heldout, load_manifest, norm, read_wav

CT2 = os.path.join(HERE, "whisper-quran-ct2-fp16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--model", default=CT2, help="CT2 model dir (default: current production model)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _parity")
    args = ap.parse_args()

    rows = load_manifest()
    print("clips: %d (<= 29 s)" % len(rows))
    from faster_whisper import WhisperModel
    m = WhisperModel(args.model, device="cuda", compute_type="float16")

    res = []
    t0 = time.time()
    for r in rows:
        audio = read_wav(r["file"])
        segs, _ = m.transcribe(audio, language="ar", beam_size=args.beam, temperature=0.0,
                               without_timestamps=True, condition_on_previous_text=False,
                               vad_filter=False,
                               max_new_tokens=min(440, int(len(audio) / 16000 * 20) + 24),
                               no_speech_threshold=None, log_prob_threshold=None,
                               compression_ratio_threshold=None)
        out = " ".join(sg.text.strip() for sg in segs)
        ref, hyp = norm(r["clean"]).split(), norm(out).split()
        we = edit_distance(ref, hyp)
        ce = edit_distance(list(" ".join(ref)), list(" ".join(hyp)))
        res.append(dict(r, out=out, ref_n=len(ref), we=we, cn=len(" ".join(ref)), ce=ce,
                        held=is_heldout(r["surah"], r["ayah"])))
    print("decoded in %.1f s" % (time.time() - t0))

    with open(os.path.join(HERE, "model_outputs%s.jsonl" % args.tag), "w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps({"file": x["file"], "out": x["out"]}, ensure_ascii=False) + "\n")

    def summarize(label, sel):
        if not sel:
            print("  %-22s no clips" % label)
            return
        w, n = sum(x["we"] for x in sel), sum(x["ref_n"] for x in sel)
        c, cn = sum(x["ce"] for x in sel), sum(x["cn"] for x in sel)
        ex = sum(1 for x in sel if x["we"] == 0)
        print("  %-22s clips=%3d  WER=%5.1f%% (%d/%d)  CER=%5.1f%%  exact clips=%d%%"
              % (label, len(sel), 100 * w / n, w, n, 100 * c / cn, round(100 * ex / len(sel))))

    print("\nMODEL: %s" % args.model)
    print("BASELINE (normalised, vs text_clean), beam=%d" % args.beam)
    summarize("ALL", res)
    summarize("held-out ayahs", [x for x in res if x["held"]])
    summarize("train ayahs", [x for x in res if not x["held"]])
    for k in sorted({x["kind"] for x in res}):
        summarize("kind=" + k, [x for x in res if x["kind"] == k])

    # what the model writes: marks (diacritics, small signs) and non-Arabic characters
    marks, other = collections.Counter(), collections.Counter()
    with_marks = 0
    for x in res:
        has = False
        for ch in x["out"]:
            if unicodedata.category(ch) == "Mn":
                marks[ch] += 1
                has = True
            elif not ("\u0600" <= ch <= "\u06FF" or ch == " "):
                other[ch] += 1
        with_marks += has
    print("\nMODEL OUTPUT FORMAT: %d of %d outputs contain diacritics/marks" % (with_marks, len(res)))
    for ch, n in marks.most_common():
        print("  U+%04X %-38s %d" % (ord(ch), unicodedata.name(ch, "?"), n))
    if other:
        print("  non-Arabic characters:", ", ".join("U+%04X x%d" % (ord(c), n) for c, n in other.most_common(10)))

    with open(os.path.join(HERE, "baseline_report%s.txt" % args.tag), "w", encoding="utf-8") as f:
        for title, sel in (("HELD-OUT (worst first)", [x for x in res if x["held"]]),
                           ("TRAIN (worst 25)", [x for x in res if not x["held"]][:])):
            f.write("=" * 20 + " " + title + "\n")
            sel = sorted(sel, key=lambda x: -(x["we"] / max(1, x["ref_n"])))
            if title.startswith("TRAIN"):
                sel = sel[:25]
            for x in sel:
                f.write("%d:%d [%s] WER %d/%d  %s\n  uthmani: %s\n  clean  : %s\n  model  : %s\n  model_n: %s\n"
                        % (x["surah"], x["ayah"], x["kind"], x["we"], x["ref_n"], x["file"],
                           x["uthmani"], x["clean"], x["out"], norm(x["out"])))
    print("\nArabic details: backend\\baseline_report%s.txt ; raw outputs: backend\\model_outputs%s.jsonl" % (args.tag, args.tag))


if __name__ == "__main__":
    main()
