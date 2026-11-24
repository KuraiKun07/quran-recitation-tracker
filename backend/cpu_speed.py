"""cpu_speed.py - time the CT2 model on CPU (int8 / float32) vs GPU fp16.

Run from backend:   python cpu_speed.py
Standalone: does not import server.py and changes nothing.
Console output is ASCII only; transcripts go to cpu_speed_report.txt (UTF-8).

Options:
  --model DIR        model folder (default: env CT2_DIR, else whisper-quran-ct2-muq)
  --lengths 2,5,10,20  target clip lengths in seconds (nearest clip is used)
  --configs LIST     device:compute:threads, comma separated
                     default cuda:float16:0,cpu:int8:4,cpu:int8:8,cpu:float32:8
  --reps N           repeats per measurement (default 3, median is reported)
  --wavs DIR         folder of 16 kHz mono PCM16 wavs (default train_data/wavs)
"""
import argparse
import glob
import os
import statistics
import sys
import time
import wave

import numpy as np

try:  # must come before faster_whisper on this machine (cublas dll)
    import torch  # noqa: F401
except Exception:
    pass

from faster_whisper import WhisperModel

STEP = float(os.environ.get("STEP_SECONDS", "0.2"))  # server re-transcribe step
FINAL_BEAMS = int(os.environ.get("FINAL_BEAMS", "4"))


def wav_info(path):
    with wave.open(path, "rb") as w:
        ok = (w.getnchannels() == 1 and w.getframerate() == 16000
              and w.getsampwidth() == 2)
        return ok, w.getnframes() / 16000.0


def read_wav(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(model, audio, beam):
    secs = len(audio) / 16000.0
    segs, _ = model.transcribe(
        audio,
        language="ar",
        temperature=0,
        without_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=False,
        beam_size=beam,
        max_new_tokens=int(min(440, secs * 20 + 24)),
    )
    return " ".join(s.text.strip() for s in segs)


def timed(model, audio, beam, reps):
    times, text = [], ""
    for _ in range(reps):
        t = time.perf_counter()
        text = transcribe(model, audio, beam)
        times.append(time.perf_counter() - t)
    return statistics.median(times), text


def pick_clips(wav_dir, lengths):
    files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    infos = []
    for f in files:
        try:
            ok, d = wav_info(f)
        except Exception:
            continue
        if ok and 0.5 <= d <= 29:
            infos.append((f, d))
    chosen, used = [], set()
    for target in lengths:
        best = None
        for f, d in infos:
            if f in used:
                continue
            if best is None or abs(d - target) < abs(best[1] - target):
                best = (f, d)
        if best:
            used.add(best[0])
            chosen.append(best)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CT2_DIR") or "whisper-quran-ct2-muq")
    ap.add_argument("--lengths", default="2,5,10,20")
    ap.add_argument("--configs", default="cuda:float16:0,cpu:int8:4,cpu:int8:8,cpu:float32:8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--wavs", default=os.path.join("train_data", "wavs"))
    a = ap.parse_args()

    lengths = [float(x) for x in a.lengths.split(",") if x.strip()]
    clips = pick_clips(a.wavs, lengths)
    if not clips:
        print("No usable wavs found in", a.wavs)
        sys.exit(1)
    audios = [(os.path.basename(f), d, read_wav(f)) for f, d in clips]

    print("model :", a.model)
    print("clips :", ", ".join("%s (%.1fs)" % (n, d) for n, d, _ in audios))
    print("greedy = partial pass (beam 1), final = beam %d, step = %.2fs" % (FINAL_BEAMS, STEP))
    print("")

    report = []
    ref_text = {}  # (clip name) -> greedy text from the first config that ran
    rows = []

    for cfg in a.configs.split(","):
        try:
            dev, comp, thr = cfg.split(":")
            thr = int(thr)
        except ValueError:
            print("bad config:", cfg)
            continue
        label = "%s/%s" % (dev, comp) + ("/%dt" % thr if dev == "cpu" else "")
        print("loading", label, "...")
        try:
            t0 = time.perf_counter()
            kw = dict(device=dev, compute_type=comp)
            if dev == "cpu":
                kw["cpu_threads"] = thr
            model = WhisperModel(a.model, **kw)
            load = time.perf_counter() - t0
            transcribe(model, audios[0][2], 1)  # warm-up, not timed
        except Exception as e:
            print("  FAILED:", str(e).encode("ascii", "replace").decode()[:200])
            continue
        print("  loaded in %.1fs" % load)

        for name, secs, audio in audios:
            g, gtxt = timed(model, audio, 1, a.reps)
            f, ftxt = timed(model, audio, FINAL_BEAMS, a.reps)
            if name not in ref_text:
                ref_text[name] = gtxt
                same = "ref"
            else:
                same = "same" if ref_text[name] == gtxt else "DIFF"
            rows.append((label, name, secs, g, f, same))
            report.append("[%s] %s (%.1fs) greedy=%.2fs final=%.2fs" % (label, name, secs, g, f))
            report.append("  greedy: " + gtxt)
            report.append("  final : " + ftxt)
        del model

    print("")
    print("%-18s %6s %9s %9s %7s %7s %s" % ("config", "clip_s", "greedy_s", "final_s", "RTF", "x_step", "text_vs_first"))
    for label, name, secs, g, f, same in rows:
        print("%-18s %6.1f %9.2f %9.2f %7.2f %7.1f %s" % (label, secs, g, f, g / secs, g / STEP, same))
    print("")
    print("RTF = greedy time / audio length (below 1 = faster than real time).")
    print("x_step = greedy time / server step; above 1 means passes pile up and the app lags.")

    with open("cpu_speed_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print("Transcripts written to cpu_speed_report.txt (open in VS Code).")


def rows_first_label(rows, name):
    for r in rows:
        if r[1] == name:
            return r[0]
    return None


if __name__ == "__main__":
    main()
