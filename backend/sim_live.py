# sim_live.py - run from backend, with uvicorn STOPPED (it loads the model on the GPU itself).
#   python sim_live.py NAME [--limit N] [--heldout] [--speed 1.0]
# Full-pipeline test with the REAL server code and model: each recorded single-ayah clip (train_data manifest,
# kind=ayah, <= 29 s) is streamed over the /ws/recite websocket in 64 ms chunks at real-time pace, like the
# browser, then 1.5 s of silence ends the utterance. Every painted word is scored against the ayah (all clips
# are correct recitations, so anything not green is a false alarm). Tapes go to replay_out\sim_NAME\tapes.
# Writes replay_out\sim_NAME\summary.txt (ASCII) and details.txt (UTF-8, Arabic).
# Timing-dependent: two runs can differ slightly (pass boundaries follow the wall clock). Compare with
#   python sim_live.py --compare A B
import argparse, base64, collections, json, os, sys, threading, time, wave

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "replay_out")
CHUNK = 1024                      # samples per message (64 ms), as the AudioWorklet sends
TAIL_SILENCE = 1.5                # seconds of zeros after the clip (pause that ends the utterance)


def load_clips(heldout_only, limit):
    sys.path.insert(0, HERE)
    from trainutil import DATA, is_heldout, load_manifest
    rows = [r for r in load_manifest(29.0) if r["kind"] == "ayah"]
    if heldout_only:
        rows = [r for r in rows if is_heldout(r["surah"], r["ayah"])]
    rows.sort(key=lambda r: r["file"])
    if limit:
        rows = rows[:limit]
    for r in rows:
        r["path"] = os.path.join(DATA, r["file"].replace("/", os.sep))
        r["heldout"] = is_heldout(r["surah"], r["ayah"])
    return rows


def pcm_of(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes())


def stream_clip(client, r, speed):
    msgs, done = [], threading.Event()
    with client.websocket_connect("/ws/recite") as ws:
        def reader():
            try:
                while True:
                    m = ws.receive_json()
                    msgs.append((time.time(), m))
                    if m.get("event") == "transcript" and m.get("final"):
                        done.set()
            except Exception:
                done.set()
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        ws.send_json({"event": "start", "surah_id": r["surah"], "starting_ayah": r["ayah"], "expected_word_index": 0})
        pcm = pcm_of(r["path"]) + bytes(int(TAIL_SILENCE * 16000) * 2)
        t0 = time.time()
        step = CHUNK * 2
        for k, i in enumerate(range(0, len(pcm), step)):
            ws.send_json({"event": "audio_chunk", "pcm_data_base64": base64.b64encode(pcm[i:i + step]).decode()})
            lag = t0 + (k + 1) * CHUNK / 16000 / speed - time.time()
            if lag > 0:
                time.sleep(lag)
        # wait for the final pass (a long clip may have been cut at 25 s: then wait for the second final too)
        deadline = time.time() + 20
        while time.time() < deadline:
            if done.wait(0.2):
                time.sleep(0.6)            # committed events follow the final transcript
                break
        audio_end = t0 + len(pcm) / 32000 / speed
    return msgs, t0, audio_end


def score(r, words, msgs):
    """words = this ayah's reference words (server.load_surah rows). Returns dict."""
    idx0 = next(i for i, w in enumerate(words) if w["ayah_number"] == r["ayah"])
    n = sum(1 for w in words if w["ayah_number"] == r["ayah"])
    status = {}
    extra = collections.Counter()
    jumps = 0
    for _t, m in msgs:
        if m.get("event") == "jump":
            jumps += 1
        if m.get("event") != "alignment_result":
            continue
        gi, st = m.get("matched_word_index"), m.get("status")
        if st == "MISTAKE_INSERTION":
            extra[st] += 1
            continue
        if idx0 <= gi < idx0 + n:
            status[gi - idx0] = st               # last paint wins, like the page (setWord overwrites; lookback repaints)
        else:
            extra["OUTSIDE"] += 1
    c = collections.Counter(status.values())
    missing = n - len(status)
    clean = c.get("CORRECT", 0) == n and not extra
    finals = [m for _t, m in msgs if m.get("event") == "transcript" and m.get("final")]
    return {"n": n, "status": status, "counts": dict(c), "missing": missing, "extra": dict(extra), "jumps": jumps,
            "clean": clean, "final_text": " | ".join(m["text"] for m in finals)}


def run(name, limit, heldout, speed):
    d = os.path.join(OUT, "sim_" + name)
    os.makedirs(d, exist_ok=True)
    os.environ["TAPE"] = "1"
    os.environ["TAPE_DIR"] = os.path.join(d, "tapes")
    sys.path.insert(0, HERE)
    import server                                       # after TAPE is set
    from fastapi.testclient import TestClient
    clips = load_clips(heldout, limit)
    print("clips: %d (heldout %d) | speed x%.1f" % (len(clips), sum(r["heldout"] for r in clips), speed))
    tot = collections.Counter()
    rows_out, det = [], []
    with TestClient(server.app) as client:
        for k, r in enumerate(clips, 1):
            words = server.load_surah(r["surah"])
            msgs, t0, audio_end = stream_clip(client, r, speed)
            s = score(r, words, msgs)
            # latency: seconds from end of audio (speech end, before the silence) to the last painted word
            paints = [t for t, m in msgs if m.get("event") == "alignment_result"]
            pcm = pcm_of(r["path"])                       # speech end = end of the last 100 ms frame above SILENCE_RMS
            loud = [i for i in range(0, len(pcm) - 3199, 3200) if server.rms(pcm[i:i + 3200]) >= server.SILENCE_RMS]
            speech_end = t0 + ((loud[-1] + 3200) if loud else len(pcm)) / 32000 / speed
            s["paint_lag"] = round(max(paints) - speech_end, 2) if paints else None
            key = "heldout" if r["heldout"] else "trained"
            tot[key + "_clips"] += 1
            tot[key + "_clean"] += s["clean"]
            for st_, v in s["counts"].items():
                tot["w_" + st_] += v
            tot["w_missing"] += s["missing"]
            tot["extra"] += sum(s["extra"].values())
            tot["jumps"] += s["jumps"]
            tot["words"] += s["n"]
            line = "%3d %3d:%-3d %-7s %-5s n=%2d %s miss=%d extra=%s jumps=%d lag=%s" % (
                k, r["surah"], r["ayah"], key, "CLEAN" if s["clean"] else "FLAG", s["n"], s["counts"], s["missing"],
                s["extra"], s["jumps"], s["paint_lag"])
            print(line)
            rows_out.append({"file": r["file"], "surah": r["surah"], "ayah": r["ayah"], "heldout": r["heldout"],
                             "clean": s["clean"], "counts": s["counts"], "missing": s["missing"], "extra": s["extra"],
                             "jumps": s["jumps"], "lag": s["paint_lag"],
                             "status": {str(k_): v for k_, v in sorted(s["status"].items())}})
            if not s["clean"]:
                det.append("%d:%d %s\n  ref  : %s\n  heard: %s\n  status: %s" % (
                    r["surah"], r["ayah"], r["file"], r["uthmani"], s["final_text"],
                    " ".join("%d=%s" % (k_, v[8:] if v.startswith("MISTAKE_") else "ok") for k_, v in sorted(s["status"].items()))))
    tapes = os.path.join(d, "tapes")
    passes = []
    for sub in sorted(os.listdir(tapes)) if os.path.isdir(tapes) else []:
        p = os.path.join(tapes, sub, "tape.jsonl")
        if os.path.exists(p):
            passes += [json.loads(l) for l in open(p, encoding="utf-8") if '"k": "pass"' in l]
    def pct(v, q):
        v = sorted(x for x in v if x is not None)
        return v[min(len(v) - 1, int(q * len(v)))] if v else None
    asr = [p["asr_ms"] for p in passes if p.get("asr_ms") is not None]
    aln = [p["align_ms"] for p in passes]
    lags = [x["lag"] for x in rows_out if x["lag"] is not None]
    summ = [
        "sim %s | clips %d | words %d" % (name, len(rows_out), tot["words"]),
        "clean clips: trained %d/%d | heldout %d/%d | all %d/%d" % (
            tot["trained_clean"], tot["trained_clips"], tot["heldout_clean"], tot["heldout_clips"],
            tot["trained_clean"] + tot["heldout_clean"], len(rows_out)),
        "word paints: CORRECT %d | SUB %d | UNSURE %d | SKIPPED %d | never painted %d | extra %d | jumps %d" % (
            tot["w_CORRECT"], tot["w_MISTAKE_SUBSTITUTION"], tot["w_MISTAKE_UNSURE"], tot["w_MISTAKE_SKIPPED"],
            tot["w_missing"], tot["extra"], tot["jumps"]),
        "passes %d | asr ms p50 %s p95 %s | align ms p50 %s p95 %s max %s" % (
            len(passes), pct(asr, .5), pct(asr, .95), pct(aln, .5), pct(aln, .95), max(aln) if aln else None),
        "last word painted after speech end (s): p50 %s p95 %s" % (pct(lags, .5), pct(lags, .95)),
    ]
    for l in summ:
        print(l)
    open(os.path.join(d, "summary.txt"), "w", encoding="ascii", errors="replace").write("\n".join(summ) + "\n")
    with open(os.path.join(d, "clips.jsonl"), "w", encoding="utf-8") as f:
        for x in rows_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    open(os.path.join(d, "details.txt"), "w", encoding="utf-8").write("\n".join(det) + "\n")
    print("files: replay_out\\sim_%s\\ (summary.txt, details.txt = Arabic, clips.jsonl, tapes\\)" % name)


def compare(a, b):
    def load(n):
        return {x["file"]: x for x in (json.loads(l) for l in open(os.path.join(OUT, "sim_" + n, "clips.jsonl"), encoding="utf-8"))}
    A, B = load(a), load(b)
    common = sorted(set(A) & set(B))
    better = [f for f in common if B[f]["clean"] and not A[f]["clean"]]
    worse = [f for f in common if A[f]["clean"] and not B[f]["clean"]]
    print("common clips %d | clean %s: %d -> %s: %d" % (len(common), a, sum(A[f]["clean"] for f in common), b,
                                                     sum(B[f]["clean"] for f in common)))
    for tag, lst in (("NOW CLEAN", better), ("NOW FLAGGED", worse)):
        for f in lst:
            print("  %-11s %d:%d %s" % (tag, A[f]["surah"], A[f]["ayah"], f))
    word_changes = 0
    for f in common:
        sa, sb = A[f]["status"], B[f]["status"]
        word_changes += sum(1 for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    print("words whose paint changed: %d" % word_changes)


def rescore(name, tapes_dir=None):
    """Re-score a sim (or a replay of one) from its tapes, with the page's last-paint rule. No model needed.
    tapes_dir defaults to replay_out\\sim_NAME\\tapes; the result is written as replay_out\\sim_NAME\\clips.jsonl."""
    sys.path.insert(0, HERE)
    from replay import find_tapes, read
    import server
    from trainutil import is_heldout
    tapes = find_tapes([tapes_dir or os.path.join(OUT, "sim_" + name, "tapes")])
    d = os.path.join(OUT, "sim_" + name)
    os.makedirs(d, exist_ok=True)
    rows_out = []
    for tp in tapes:
        recs = read(tp)
        st = next(r["m"] for r in recs if r["k"] == "in" and r["m"].get("event") == "start")
        wav = os.path.basename(tp)
        r = {"surah": int(st["surah_id"]), "ayah": int(st["starting_ayah"])}
        msgs = [(r_["t"], r_["m"]) for r_ in recs if r_["k"] == "out"]
        s = score(r, server.load_surah(r["surah"]), msgs)
        rows_out.append({"file": "%d:%d#%s" % (r["surah"], r["ayah"], wav), "surah": r["surah"], "ayah": r["ayah"],
                         "heldout": is_heldout(r["surah"], r["ayah"]), "clean": s["clean"], "counts": s["counts"],
                         "missing": s["missing"], "extra": s["extra"], "jumps": s["jumps"], "lag": None,
                         "status": {str(k_): v for k_, v in sorted(s["status"].items())}})
    # same clip order in every run -> key by clip position, not tape folder name
    for i, x in enumerate(rows_out):
        x["file"] = "%03d %d:%d" % (i, x["surah"], x["ayah"])
    with open(os.path.join(d, "clips.jsonl"), "w", encoding="utf-8") as f:
        for x in rows_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tr = [x for x in rows_out if not x["heldout"]]
    ho = [x for x in rows_out if x["heldout"]]
    print("rescored %s from %d tapes | clean: trained %d/%d | heldout %d/%d | all %d/%d" % (
        name, len(rows_out), sum(x["clean"] for x in tr), len(tr), sum(x["clean"] for x in ho), len(ho),
        sum(x["clean"] for x in rows_out), len(rows_out)))


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "--rescore":
        rescore(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
        sys.exit(0)
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    run(a.name, a.limit, a.heldout, a.speed)
