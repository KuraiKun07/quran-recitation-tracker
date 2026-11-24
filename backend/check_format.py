# check_format.py - step 0 of the ASR fine-tune. Run from backend.
#
#   python check_format.py
#   python check_format.py --ayah 112:1 --out "<paste model output>"
#   python check_format.py --ayah 112:1 --wav debug_last.wav
#
# Question answered: can the model's tokenizer represent our DB targets
# (text_uthmani / text_clean) cheaply, i.e. are they in a style the model
# already knows?  Console output is ASCII only (your console garbles Arabic);
# all Arabic goes to check_format_report.txt (UTF-8, open in VS Code).
import argparse, collections, os, sqlite3, sys, unicodedata, wave

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "quran.db")
MODEL = "tarteel-ai/whisper-base-ar-quran"
CT2 = os.path.join(HERE, "whisper-quran-ct2-fp16")


def cp(ch):
    return "U+%04X %s" % (ord(ch), unicodedata.name(ch, "?"))


def load_ayahs():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT surah, ayah, pos, text_uthmani, text_clean "
                       "FROM words ORDER BY surah, ayah, pos").fetchall()
    con.close()
    acc = {}
    for s, a, p, u, c in rows:
        d = acc.setdefault((s, a), ([], []))
        d[0].append(u)
        d[1].append(c)
    return {k: (" ".join(v[0]), " ".join(v[1])) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ayah", help="S:A, e.g. 112:1")
    ap.add_argument("--out", help="a model output string to compare with that ayah")
    ap.add_argument("--wav", help="16 kHz mono WAV to transcribe with the current CT2 model")
    args = ap.parse_args()

    ayahs = load_ayahs()
    print("DB: %d ayahs" % len(ayahs))
    rep = open(os.path.join(HERE, "check_format_report.txt"), "w", encoding="utf-8")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    print("tokenizer class:", type(tok).__name__)
    try:
        tok.set_prefix_tokens(language="arabic", task="transcribe", predict_timestamps=False)
    except Exception as e:
        print("set_prefix_tokens failed (not fatal):", e)

    for name, idx in (("text_uthmani", 0), ("text_clean", 1)):
        counts, bad, nwords = [], [], 0
        for k, v in ayahs.items():
            t = v[idx]
            ids = tok(t, add_special_tokens=False).input_ids
            counts.append(len(ids))
            nwords += len(t.split())
            back = tok.decode(ids, skip_special_tokens=True).strip()
            if back != t:
                bad.append((k, t, back))
        over = sum(1 for c in counts if c > 440)
        print("\n[%s]" % name)
        print("  tokens/word: %.2f | longest ayah: %d tokens | ayahs over 440 tokens: %d"
              % (sum(counts) / nwords, max(counts), over))
        print("  round-trip mismatches: %d of %d" % (len(bad), len(ayahs)))
        rep.write("=== %s round-trip mismatches (first 30) ===\n" % name)
        for k, t, b in bad[:30]:
            rep.write("%s\n  want: %s\n  got : %s\n" % (k, t, b))

    # characters of text_uthmani that the tokenizer splits into 2+ tokens on their own
    freq = collections.Counter(ch for v in ayahs.values() for ch in v[0] if ch != " ")
    print("\n[text_uthmani characters] %d distinct" % len(freq))
    print("  chars costing 2+ tokens alone (rare to the model):")
    rare = 0
    for ch, n in freq.most_common():
        nt = len(tok(ch, add_special_tokens=False).input_ids)
        rep.write("%s\tcount=%d\ttokens_alone=%d\t%s\n" % (ch, n, nt, cp(ch)))
        if nt >= 2:
            rare += 1
            print("   %s  count=%d  tokens=%d" % (cp(ch), n, nt))
    if not rare:
        print("   none")

    if args.ayah and (args.out or args.wav):
        s, a = (int(x) for x in args.ayah.split(":"))
        u, c = ayahs[(s, a)]
        out = args.out
        if args.wav:
            import numpy as np
            from faster_whisper import WhisperModel
            with wave.open(args.wav, "rb") as w:
                assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16 kHz mono"
                pcm = w.readframes(w.getnframes())
            audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
            m = WhisperModel(CT2, device="cuda", compute_type="float16")
            segs, _ = m.transcribe(audio, language="ar", beam_size=4, without_timestamps=True,
                                   condition_on_previous_text=False, vad_filter=False)
            out = " ".join(sg.text.strip() for sg in segs)
        print("\n[compare %s]" % args.ayah)
        extra = sorted(set(out) - set(u) - {" "})
        missing = sorted(set(u) - set(out) - {" "})
        print("  chars in model output but NOT in text_uthmani:", [cp(x) for x in extra] or "none")
        print("  chars in text_uthmani but NOT in model output:", [cp(x) for x in missing] or "none")
        try:
            from alignment import normalize
            print("  normalized output == normalized text_clean:", normalize(out) == normalize(c))
        except Exception as e:
            print("  (normalize comparison skipped: %s)" % e)
        rep.write("\n=== compare %s ===\nmodel   : %s\nuthmani : %s\nclean   : %s\n" % (args.ayah, out, u, c))

    rep.close()
    print("\nArabic details: backend\\check_format_report.txt (open in VS Code)")


if __name__ == "__main__":
    main()
