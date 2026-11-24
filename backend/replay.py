# replay.py - run from backend.  Deterministic replay of recorded sessions (tapes) through the REAL server code.
#   python replay.py NAME TAPES... [--asr] [--set KEY=VALUE ...] [--tape-env | --cur-env]
#     TAPES  = tape folders, or folders that contain tape folders (e.g. tapes  or  replay_out\sim_base_a\tapes)
#     --text (default): the recorded transcript of each pass is used (no model, CPU, seconds). Tests every change
#                       in alignment / commit / resync / lookback / jump logic.
#     --asr           : the model re-transcribes the same audio slices (GPU, uvicorn STOPPED). Tests ASR changes.
#     --set K=V       : override an env knob for this replay (the experiment), e.g. --set SPLIT_FOLD=0
#     env: by default each knob takes the value recorded in the FIRST tape (--tape-env); --cur-env uses this shell.
# How: the recorded utterance audio is fed into the unchanged /ws/recite worker exactly up to each recorded pass
# length, and the next slice is sent only after that pass is finished, so the pass schedule is identical.
# Output: replay_out\rp_NAME\  summary.txt (ASCII), details.txt (UTF-8, Arabic), <tape>\tape.jsonl (new tapes).
# Compare result vs the recorded tape: identical passes, first divergence, and every word whose paint changed.
import argparse, base64, glob, json, os, shutil, sys, threading, time, wave

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "replay_out")
NOT_KNOBS = {"TAPE", "TAPE_DIR", "DEBUG_TEXT", "DEBUG_ALIGN"}
CMP_FIELDS = ("len", "final", "n", "cur", "fed", "lookback", "jump", "jumped", "short", "ev")


def find_tapes(paths):
    out = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "tape.jsonl")):
            out.append(os.path.abspath(p))
        else:
            out += sorted(os.path.dirname(os.path.abspath(f)) for f in glob.glob(os.path.join(p, "**", "tape.jsonl"), recursive=True))
    return out


def read(path):
    return [json.loads(l) for l in open(os.path.join(path, "tape.jsonl"), encoding="utf-8") if l.strip()]


def plan(recs):
    """Tape -> list of steps: ("in", msg) or ("utt", wavfile_or_None, [passes]). Passes of one utterance are the
    passes up to the 'wav' record that closes it (or up to a drop/error, then wav is None)."""
    steps, cur = [], []
    for r in recs:
        k = r["k"]
        if k == "in":
            if cur:                                 # control message in the middle of an utterance: flush what we have
                steps.append(("utt", None, cur)); cur = []
            steps.append(("in", r["m"]))
        elif k == "pass":
            cur.append(r)
        elif k == "wav":
            steps.append(("utt", r["file"], cur)); cur = []
        elif k in ("drop", "error"):
            if cur:
                steps.append(("utt", None, cur, r.get("len") if k == "drop" else None)); cur = []
    if cur:
        steps.append(("utt", None, cur))
    return steps


def paints(recs):
    """Last paint per word index from the outbound alignment_result messages (what the page shows)."""
    p, ins = {}, 0
    for r in recs:
        if r["k"] == "out" and r["m"].get("event") == "alignment_result":
            m = r["m"]
            if m["status"] == "MISTAKE_INSERTION":
                ins += 1
            else:
                p[m["matched_word_index"]] = (m["status"], m.get("expected_word"), m.get("spoken_transcription"))
        elif r["k"] == "out" and r["m"].get("event") == "jump":
            p = {k: v for k, v in p.items() if k < r["m"]["to_index"]}   # page hides words from the jump target on
    return p, ins


def commits(recs):
    """word index -> (utterance number, audio bytes of the pass that FIRST committed it): when the page saw it."""
    out, u = {}, 0
    for r in recs:
        if r["k"] == "pass":
            for e in r["ev"][:r["n"]]:
                if e[1] != "MISTAKE_INSERTION" and e[0] not in out:
                    out[e[0]] = (u, r["len"])
        elif r["k"] in ("wav", "drop", "error"):
            u += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("tapes", nargs="+")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--cur-env", action="store_true")
    a = ap.parse_args()
    tapes = find_tapes(a.tapes)
    if not tapes:
        print("no tapes found"); return 1
    first = read(tapes[0])[0]
    if not a.cur_env:
        for k, v in first.get("knobs", {}).items():
            if k in NOT_KNOBS:
                continue
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for kv in a.set:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    od = os.path.join(OUT, "rp_" + a.name)
    os.makedirs(od, exist_ok=True)
    os.environ["TAPE"] = "1"
    if not a.asr:
        os.environ["DEBUG_TEXT"] = "1"            # skips model loading at startup (nothing else uses it in replay)
    sys.path.insert(0, HERE)
    import server
    from fastapi.testclient import TestClient

    done = threading.Event()
    orig_w = server.Tape.w
    def w(self, kind, **kv):
        orig_w(self, kind, **kv)
        if kind in ("pass", "drop", "error"):
            done.set()
    server.Tape.w = w
    fake = {"len": None, "text": "", "bad": 0}
    if not a.asr:
        def transcribe_ctx(pcm, final, prompt):
            if len(pcm) != fake["len"]:
                fake["bad"] += 1
            return fake["text"]
        server.transcribe_ctx = transcribe_ctx

    tot = dict(tapes=0, passes=0, same=0, diverged_tapes=0, stalls=0, words=0, changed=0, ins_old=0, ins_new=0)
    summ_lines, det, used = [], [], set()
    with TestClient(server.app) as client:
        for tp in tapes:
            recs = read(tp)
            s = recs[0]
            knob_diff = {k: (v, os.environ.get(k)) for k, v in s.get("knobs", {}).items()
                         if k not in NOT_KNOBS and os.environ.get(k) != v}
            server.STEP_BYTES, server.FIRST_PASS_BYTES = s["step_bytes"], s["first_pass_bytes"]
            server.FINAL_BEAMS = s["final_beams"]
            name = os.path.basename(tp)
            if name in used:                          # replaying replay output: inner time folders can repeat
                n_ = 2
                while "%s_%d" % (name, n_) in used:
                    n_ += 1
                name = "%s_%d" % (name, n_)
            used.add(name)
            server.TAPE_DIR = os.path.join(od, name)
            if os.path.isdir(server.TAPE_DIR):
                shutil.rmtree(server.TAPE_DIR)        # only ever inside replay_out\rp_NAME
            stalls = 0
            with client.websocket_connect("/ws/recite") as ws:
                def drain():
                    try:
                        while True:
                            ws.receive_json()
                    except Exception:
                        pass
                threading.Thread(target=drain, daemon=True).start()
                for step in plan(recs):
                    if step[0] == "in":
                        ws.send_json(step[1]); time.sleep(0.05)
                        continue
                    wavf, passes = step[1], step[2]
                    drop_len = step[3] if len(step) > 3 else None
                    if wavf:
                        with wave.open(os.path.join(tp, wavf), "rb") as wf:
                            pcm = wf.readframes(wf.getnframes())
                    else:                              # no audio kept (cough drop / error): synthetic audio
                        import numpy as np
                        n = max([p["len"] for p in passes] + [drop_len or 0])
                        pcm = bytearray((np.random.default_rng(0).standard_normal(n // 2) * 3000).astype(np.int16).tobytes())
                        if drop_len:                   # sound first, then quiet for the last pause -> the server drops it again
                            q = max(3200, drop_len - server.TAIL_BYTES - 3200)
                            pcm[q:] = bytes(len(pcm) - q)
                        pcm = bytes(pcm)
                    sent = 0
                    for p in passes:
                        done.clear()
                        fake["len"], fake["text"] = p["len"], p.get("text") or ""
                        if p["len"] > sent:
                            ws.send_json({"event": "audio_chunk",
                                          "pcm_data_base64": base64.b64encode(pcm[sent:p["len"]]).decode()})
                            sent = p["len"]
                        if not done.wait(60 if a.asr else 30):   # 30: FORCED_CHECK may load the model on first use
                            stalls += 1
                    if drop_len and drop_len > sent:   # deliver the rest: the worker sees the pause and drops it
                        done.clear()
                        ws.send_json({"event": "audio_chunk", "pcm_data_base64": base64.b64encode(pcm[sent:drop_len]).decode()})
                        done.wait(5)
                    elif wavf is None and passes and not passes[-1]["final"]:
                        done.clear()                    # end the synthetic utterance with silence
                        ws.send_json({"event": "audio_chunk", "pcm_data_base64": base64.b64encode(bytes(32000)).decode()})
                        done.wait(5)
                time.sleep(0.2)
            new = read(find_tapes([server.TAPE_DIR])[0])     # Tape() makes one <time> folder inside TAPE_DIR
            p0 = [r for r in recs if r["k"] == "pass"]
            p1 = [r for r in new if r["k"] == "pass"]
            same = 0
            first_div = None
            for i, (x, y) in enumerate(zip(p0, p1)):
                if all(x.get(f) == y.get(f) for f in CMP_FIELDS) and (a.asr or x.get("text") == y.get("text")):
                    same += 1
                elif first_div is None:
                    first_div = i
            if first_div is None and len(p0) != len(p1):
                first_div = min(len(p0), len(p1))
            w0, i0 = paints(recs)
            w1, i1 = paints(new)
            ch = sorted(k for k in set(w0) | set(w1) if (w0.get(k) or (None,))[0] != (w1.get(k) or (None,))[0])
            tot["tapes"] += 1; tot["passes"] += len(p0); tot["same"] += same; tot["stalls"] += stalls
            tot["diverged_tapes"] += first_div is not None; tot["words"] += len(set(w0) | set(w1))
            tot["changed"] += len(ch); tot["ins_old"] += i0; tot["ins_new"] += i1
            c0, c1 = commits(recs), commits(new)
            late = [(c1[k][1] - c0[k][1]) / 32000.0 for k in c0 if k in c1 and c1[k][0] == c0[k][0] and c1[k][1] > c0[k][1]]
            early = sum(1 for k in c0 if k in c1 and c1[k][0] == c0[k][0] and c1[k][1] < c0[k][1])
            tot.setdefault("late", []).extend(late); tot["early"] = tot.get("early", 0) + early
            line = "%-28s passes %3d/%3d same | first diff %s | words changed %d | extra words %d->%d%s%s%s" % (
                name, same, len(p0), first_div, len(ch), i0, i1,
                " | later %d (+%.2fs max)" % (len(late), max(late)) if late else "",
                " | STALLS %d" % stalls if stalls else "",
                " | knobs differ from tape: %s" % sorted(knob_diff) if knob_diff else "")
            print(line); summ_lines.append(line)
            if ch:
                det.append("== %s" % name)
                for k in ch:
                    o, n_ = w0.get(k), w1.get(k)
                    det.append("  word %d %s : %s (heard %s) -> %s (heard %s)" % (
                        k, (o or n_)[1], o[0] if o else "-", o[2] if o else "-", n_[0] if n_ else "-", n_[2] if n_ else "-"))
    sched = " | ASR schedule mismatches %d" % fake["bad"] if not a.asr else ""
    late = tot.get("late", [])
    sched += " | painted later %d (mean +%.2fs, max +%.2fs), earlier %d" % (
        len(late), sum(late) / len(late) if late else 0, max(late) if late else 0, tot.get("early", 0))
    total = ("TOTAL tapes %d | passes identical %d/%d | tapes diverged %d | words painted %d, changed %d | "
             "extra words %d->%d | stalls %d%s" % (tot["tapes"], tot["same"], tot["passes"], tot["diverged_tapes"],
                                                   tot["words"], tot["changed"], tot["ins_old"], tot["ins_new"], tot["stalls"], sched))
    print(total)
    print("settings: %s%s" % ("asr" if a.asr else "text", (" | --set " + " ".join(a.set)) if a.set else ""))
    open(os.path.join(od, "summary.txt"), "w", encoding="ascii", errors="replace").write("\n".join(summ_lines + [total]) + "\n")
    open(os.path.join(od, "details.txt"), "w", encoding="utf-8").write("\n".join(det) + "\n")
    print("files: replay_out\\rp_%s\\ (summary.txt, details.txt = Arabic)" % a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
