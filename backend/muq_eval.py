# muq_eval.py - closed-set scorer for the letter openings. Run from backend.
#   python muq_eval.py                                  # production model, held-out takes
#   python muq_eval.py --split all                      # every take
#   python muq_eval.py --model runs\X\ep3_ct2           # a trained model
# Per clip it measures, against the DB target (compact form, e.g. the normalised text of 2:1):
#   final   - does the whole-clip transcript equal the target (beam 4 = server's final pass, greedy)?
#   early   - did a PARTIAL pass (greedy, growing prefix, like the live passes) already equal the full
#             target before --early (default 75%) of the speech was over? = the page would turn green
#             while you are still reciting.
#   flip    - after first matching, did a later pass change away from the target?
#   never   - no pass ever matched.
# Console is ASCII only; Arabic goes to muq_eval_report.txt (UTF-8, open in VS Code).
import argparse, collections, os

import torch  # noqa: F401  (loads CUDA DLLs before faster_whisper)
import numpy as np

from trainutil import HERE, is_heldout_take, load_manifest, norm, read_wav, take_of

PROD = os.path.join(HERE, "whisper-quran-ct2-fp16")


def speech_bounds(a, frame=320):
    n = len(a) // frame
    if n == 0:
        return 0.0, len(a) / 16000
    rms = np.sqrt((a[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    act = np.where(rms > max(0.1 * float(rms.max()), 0.003))[0]
    if len(act) == 0:
        return 0.0, len(a) / 16000
    return act[0] * frame / 16000, (act[-1] + 1) * frame / 16000


def analyze_clip(decode, audio, target, step=0.3, first=0.7, early_frac=0.75):
    """decode(audio, beam) -> text. Pure logic, so it can be tested without a model."""
    dur = len(audio) / 16000
    _, end = speech_bounds(audio)
    f4, fg = norm(decode(audio, 4)), norm(decode(audio, 1))
    seq, t = [], first
    while t < dur - 0.05:
        seq.append((round(t, 2), norm(decode(audio[: int(t * 16000)], 1))))
        t += step
    seq.append((round(dur, 2), fg))
    first_match = next((t for t, o in seq if o == target), None)
    early = first_match is not None and first_match < early_frac * end
    flip = first_match is not None and (any(o != target for t, o in seq if t >= first_match) or f4 != target)
    never = first_match is None and f4 != target
    return {"f4": f4 == target, "fg": fg == target, "early": early, "flip": flip, "never": never,
            "first_match": first_match, "speech_end": end, "f4_text": f4, "seq": seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PROD)
    ap.add_argument("--split", choices=["heldout", "train", "all"], default="heldout")
    ap.add_argument("--early", type=float, default=0.75)
    args = ap.parse_args()

    rows = [r for r in load_manifest() if r["kind"] == "muq"]
    if args.split == "heldout":
        rows = [r for r in rows if is_heldout_take(r)]
    elif args.split == "train":
        rows = [r for r in rows if not is_heldout_take(r)]
    print("model: %s | split: %s | %d letter-opening clips" % (args.model, args.split, len(rows)))

    from faster_whisper import WhisperModel
    m = WhisperModel(args.model if os.path.isabs(args.model) else os.path.join(HERE, args.model),
                     device="cuda", compute_type="float16")

    def decode(audio, beam):
        segs, _ = m.transcribe(audio, language="ar", beam_size=beam, temperature=0.0, without_timestamps=True,
                               condition_on_previous_text=False, vad_filter=False,
                               max_new_tokens=min(440, int(len(audio) / 16000 * 20) + 24),
                               no_speech_threshold=None, log_prob_threshold=None, compression_ratio_threshold=None)
        return " ".join(sg.text.strip() for sg in segs)

    groups = collections.defaultdict(list)
    report = []
    for r in rows:
        target = norm(r["clean"])
        res = analyze_clip(decode, read_wav(r["file"]), target, early_frac=args.early)
        groups[target].append((r, res))
        report.append("%d:%d take %d  target=%s  first_match=%s speech_end=%.2f\n  final beam4: %s\n  prefixes: %s"
                      % (r["surah"], r["ayah"], take_of(r), target, res["first_match"], res["speech_end"],
                         res["f4_text"], " | ".join("%.1f:%s" % (t, o) for t, o in res["seq"])))

    def pct(x, n):
        return "%3d%%" % round(100 * x / max(1, n))

    print("\n%-9s %4s %7s %7s %7s %7s %7s" % ("example", "n", "final4", "finalG", "early", "flip", "never"))
    tot = collections.Counter()
    for tgt, lst in sorted(groups.items(), key=lambda kv: (kv[1][0][0]["surah"], kv[1][0][0]["ayah"])):
        n = len(lst)
        c = collections.Counter(k for _, res in lst for k in ("f4", "fg", "early", "flip", "never") if res[k])
        tot.update(c)
        tot["n"] += n
        ex = "%d:%d" % (lst[0][0]["surah"], lst[0][0]["ayah"])
        print("%-9s %4d %7s %7s %7s %7s %7s" % (ex, n, pct(c["f4"], n), pct(c["fg"], n), pct(c["early"], n),
                                              pct(c["flip"], n), pct(c["never"], n)))
    n = tot["n"]
    print("%-9s %4d %7s %7s %7s %7s %7s   <- ALL" % ("", n, pct(tot["f4"], n), pct(tot["fg"], n), pct(tot["early"], n),
                                                  pct(tot["flip"], n), pct(tot["never"], n)))
    with open(os.path.join(HERE, "muq_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\nArabic details: backend\\muq_eval_report.txt")


if __name__ == "__main__":
    main()
