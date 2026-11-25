"""Quran Recitation Tracker server.  Run (inside backend/):  uvicorn server:app

Streaming strategy (Whisper is not a streaming model, so we emulate it):
  * keep one utterance buffer; every 0.2 s (GPU) / 0.4 s (CPU) re-transcribe the WHOLE buffer (partial pass, greedy)
  * correct words are painted at once; a mistake needs two agreeing passes and a correct word after it
  * a pause (silence) ends the utterance: final pass (beam search) commits the rest, the buffer is cleared
The defaults are the production settings. Useful env vars: CT2_DIR (speech model folder), SILENCE_RMS (default 350,
raise it if room noise keeps utterances from ending), HW_DEVICE=cpu, TAPE=1 (session recorder for replay.py/bench.py).
Every other knob exists for the benchmark (bench.py --set KEY=VALUE).
"""
import asyncio, base64, json, os, sqlite3, subprocess, sys, time, traceback, wave
from contextlib import asynccontextmanager
from functools import lru_cache
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from alignment import Aligner, normalize, public
from ayah_finder import AyahFinder
import cudaprep   # GPU detection without torch

APP_VERSION = "1.2.0-beta1"
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "quran.db")
DEBUG_TEXT = os.getenv("DEBUG_TEXT") == "1"
SILENCE_RMS = float(os.getenv("SILENCE_RMS", "350"))     # int16 scale; a 100 ms frame louder than this counts as speech
PAUSE_SECONDS = float(os.getenv("PAUSE_SECONDS", "0.9"))  # silence that ends a short utterance (phrases separated by shorter pauses stay together)
BPS = 32000                                               # bytes/second of 16 kHz PCM16 mono
STEP_BYTES, MIN_BYTES, MAX_BYTES, TAIL_BYTES = int(0.4 * BPS), int(0.5 * BPS), 25 * BPS, int(PAUSE_SECONDS * BPS)
SHORT_CLIP_BYTES = int(1.5 * BPS)   # Whisper is unreliable on clips shorter than this -> doubtful words are 'unsure', not 'wrong'
FIRST_PASS_BYTES, MIN_UTT_BYTES, PRE_ROLL = int(0.9 * BPS), int(0.7 * BPS), int(0.3 * BPS)  # avoid hallucinations on tiny clips

@asynccontextmanager
async def lifespan(app):
    if not DEBUG_TEXT:   # load the model at startup so the first recitation isn't slow
        loop = asyncio.get_running_loop()
        print("Loading ASR model…"); await loop.run_in_executor(None, get_asr)
        print("Warming up…"); await loop.run_in_executor(None, lambda: transcribe(bytes(BPS), True)); print("ASR ready.")
        if PHONE_CHECK: await loop.run_in_executor(None, phone_warmup)
    yield

print(f"*** Quran Recitation Tracker v{APP_VERSION} ***")

app = FastAPI(lifespan=lifespan)

# ---------- data ----------
@lru_cache(maxsize=16)
def load_surah(surah: int):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found. Run: python build_db.py (inside backend/)")
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT ayah AS ayah_number, pos AS word_position, text_uthmani, text_clean, page, line "
                       "FROM words WHERE surah=? ORDER BY ayah, pos", (surah,)).fetchall()
    return [dict(r) for r in rows]

@app.get("/surah/{surah}")
def surah_words(surah: int):
    try:
        words = load_surah(surah)
    except (FileNotFoundError, sqlite3.OperationalError) as e:
        raise HTTPException(503, f"Database problem: {e}. Delete backend/quran.db and run: python build_db.py")
    if not words:
        raise HTTPException(404, f"Surah {surah} is not in the database. Run: python build_db.py")
    return [{"uthmani": w["text_uthmani"], "ayah": w["ayah_number"], "page": w["page"], "line": w["line"]} for w in words]

@app.get("/api/surahs")
def api_surahs():
    try:
        con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
        rows = con.execute("SELECT surah, name_ar, name_en, ayahs, page FROM surahs ORDER BY surah").fetchall()
    except sqlite3.OperationalError:
        raise HTTPException(503, "Database is outdated or missing. Run: python build_db.py (inside backend/) to build all 114 surahs.")
    return [dict(r) for r in rows]

import mushaf; mushaf.register(app, DB_PATH)

@app.get("/api/version")
def api_version():
    try:
        con = sqlite3.connect(DB_PATH)
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.OperationalError:
        meta = {}
    return {"version": APP_VERSION, "db": meta}

# ---------- ASR ----------
def _model_dir():
    """CT2_DIR, else backend\\model (release package), else backend\\whisper-quran-ct2-muq (development)."""
    if os.getenv("CT2_DIR"): return os.getenv("CT2_DIR")
    for name in ("model", "whisper-quran-ct2-muq"):
        if os.path.isfile(os.path.join(HERE, name, "model.bin")): return os.path.join(HERE, name)
    return os.path.join(HERE, "model")
CT2_DIR = _model_dir()   # one fp16 CTranslate2 model: fp16 on the GPU, int8 on the processor
LENIENT = os.getenv("LENIENT", "1") != "0"   # tolerate dropped/garbled weak letters and 1-2 letter fragments

def ct2_gpu_works(path: str) -> bool:
    """CTranslate2 on Windows can hard-crash the process if CUDA libs are missing, so test it in a throwaway
    subprocess first. Result is cached in asr_probe.json (delete that file to re-test)."""
    cache = os.path.join(HERE, "asr_probe.json")
    try: stamp = os.path.getmtime(os.path.join(path, "model.bin"))
    except OSError: return False
    try:
        c = json.load(open(cache))
        if c.get("path") == path and c.get("stamp") == stamp: return bool(c["ok"])
    except Exception: pass
    code = ("import cudaprep; cudaprep.prep()\nfrom faster_whisper import WhisperModel\nimport numpy as np\n"
            f"m = WhisperModel(r'{path}', device='cuda', compute_type='{cudaprep.gpu_compute()}')\n"
            "list(m.transcribe(np.zeros(16000, dtype=np.float32), language='ar', beam_size=1)[0])")
    try: ok = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=180, cwd=HERE).returncode == 0
    except Exception: ok = False
    try: json.dump({"path": path, "stamp": stamp, "ok": ok}, open(cache, "w"))
    except Exception: pass
    return ok

FINAL_BEAMS = 1   # set in get_asr(): beam search for the final pass of each utterance (GPU only). Partial passes are greedy.

@lru_cache(maxsize=1)
def get_asr():
    """faster-whisper (CTranslate2): fp16 on a suitable NVIDIA GPU, else int8 on the processor."""
    global FINAL_BEAMS, STEP_BYTES, FIRST_PASS_BYTES
    if not os.path.isfile(os.path.join(CT2_DIR, "model.bin")):
        raise FileNotFoundError("speech model not found: %s (set CT2_DIR to the model folder)" % CT2_DIR)
    gpu = cudaprep.gpu_available()
    if gpu and not ct2_gpu_works(CT2_DIR):
        print("CTranslate2 GPU test failed -> using the processor (int8)")
        gpu = False
    from faster_whisper import WhisperModel
    print(f"ASR backend: faster-whisper/CTranslate2 | {CT2_DIR} | device: {'GPU fp16' if gpu else 'CPU int8'}")
    FINAL_BEAMS = int(os.getenv("FINAL_BEAMS", "4" if gpu else "1"))
    if gpu:                                       # the GPU is fast enough for denser passes
        STEP_BYTES = int(float(os.getenv("STEP_SECONDS", "0.2")) * BPS)
        FIRST_PASS_BYTES = int(0.7 * BPS)
    return WhisperModel(CT2_DIR, device="cuda" if gpu else "cpu",
                        compute_type=cudaprep.gpu_compute() if gpu else "int8", cpu_threads=cudaprep.cpu_threads())

def transcribe(pcm: bytes, final: bool = False, prompt: str = "") -> str:
    audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    # Cap decoding length by audio duration: a runaway/repeating decode can take ages and freezes tracking.
    max_tokens = min(400 if prompt else 440, int(len(pcm) / BPS * 20) + 24)
    segs, _ = get_asr().transcribe(audio, language="ar", beam_size=FINAL_BEAMS if final else 1, temperature=0.0,
                                   without_timestamps=True, condition_on_previous_text=False, vad_filter=False,
                                   max_new_tokens=max_tokens, no_speech_threshold=None, log_prob_threshold=None,
                                   compression_ratio_threshold=None, initial_prompt=(prompt or None))
    return " ".join(sg.text.strip() for sg in segs)

def rms(b: bytes) -> float:
    if len(b) < 2: return 0.0
    a = np.frombuffer(b[: len(b) // 2 * 2], np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a)))

def peak_rms(b: bytes, frame: int = 3200) -> float:        # loudest 100 ms frame
    return max((rms(b[i:i + frame]) for i in range(0, len(b), frame)), default=0.0)

def start_index(words, ayah, word_idx):
    first = next((i for i, w in enumerate(words) if w["ayah_number"] == ayah), 0)
    return first + word_idx

# --- spoken basmala and the letter openings (muqattaat) ---
from basmala import strip_basmala
from muqattaat import is_muq
MUQ_HOLD = os.environ.get("MUQ_HOLD", "1") != "0"             # MUQ_HOLD=0 switches the hold off
BASMALA_ON = os.environ.get("BASMALA", "1") != "0"           # BASMALA=0 switches the feature off
BASMALA_EVENT = os.environ.get("BASMALA_EVENT", "0") == "1"  # BASMALA_EVENT=1 also sends {"event":"basmala","count":n} (the client must ignore unknown events)

# ---------- websocket ----------
# --- find by voice ---
@lru_cache(maxsize=1)
def get_finder():
    return AyahFinder(DB_PATH)

def find_payload(text: str, final: bool) -> dict:
    r = get_finder().find(text, topn=5, partial=not final)
    cands = []
    if r["n_words"] >= (1 if final else 2):
        con = sqlite3.connect(DB_PATH)
        try:
            for x in r["results"]:
                rows = con.execute("SELECT text_uthmani, page FROM words WHERE surah=? AND ayah=? ORDER BY pos",
                                   (x["surah"], x["ayah"])).fetchall()
                cands.append({"surah": x["surah"], "ayah": x["ayah"],
                              "start": 0 if x["start"] <= 1 else x["start"],   # 1 = first word probably dropped by the ASR
                              "score": round(x["score"], 2), "conf": round(x["conf"], 2),
                              "page": rows[0][1] if rows else None, "nwords": len(rows),
                              "preview": " ".join(w for w, _p in rows[:8])})
        finally:
            con.close()
    return {"event": "find_result", "final": final, "transcript": text, "n_words": r["n_words"],
            "confident": bool(r["confident"] and cands), "margin": round(r["margin"], 2),
            "basmala_stripped": r["basmala_stripped"], "candidates": cands}

# --- jump guard ---
JUMP_MIN_WORDS = int(os.getenv("JUMP_MIN_WORDS", "3"))
JUMP_FED = int(os.getenv("JUMP_FED", "2"))   # no jump once this many words of the utterance are consumed (0 disables)
def jump_allowed(jp, n_words, st):
    """Reject a restart/jump that leaves the practised range or rests on too little speech."""
    why = None
    lo = st.get('range_lo')
    if n_words < JUMP_MIN_WORDS: why = 'only %d words' % n_words
    elif lo is not None and jp < lo: why = 'before range start %d' % lo
    elif JUMP_FED > 0 and st.get('fed', 0) >= JUMP_FED: why = 'mid-utterance, %d words already consumed' % st.get('fed', 0)
    if why and st.get('last_rej') != (jp, why):
        st['last_rej'] = (jp, why); print('[jump to %d rejected: %s]' % (jp, why))
    return why is None

DEBUG_ALIGN = os.getenv("DEBUG_ALIGN") == "1"   # verbose alignment log

# --- tape: session flight recorder (TAPE=1, default off). Local files: tapes\<time>\tape.jsonl + utt_NNNN.wav ---
TAPE = os.getenv("TAPE") == "1"
TAPE_DIR = os.getenv("TAPE_DIR") or os.path.join(HERE, "tapes")
def _tape_knobs():
    """Every env knob read by server.py / alignment.py, with its current value (None = unset = code default)."""
    import re as _re_tp
    names = set()
    for fn in ("server.py", "alignment.py"):
        try:
            src = open(os.path.join(HERE, fn), encoding="utf-8").read()
            names |= set(_re_tp.findall(r'(?:getenv|environ\.get)\(\s*"([A-Z][A-Z0-9_]+)"', src))
        except OSError:
            pass
    return {k: os.environ.get(k) for k in sorted(names)}
class Tape:
    def __init__(self):
        self.f, self.n = None, 0
        if not TAPE: return
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S"); self.dir = os.path.join(TAPE_DIR, stamp); k = 1
            while os.path.exists(self.dir):
                self.dir = os.path.join(TAPE_DIR, "%s_%d" % (stamp, k)); k += 1
            os.makedirs(self.dir)
            self.f = open(os.path.join(self.dir, "tape.jsonl"), "w", encoding="utf-8")
            self.t0 = time.time()
            self.w("session", version=APP_VERSION, knobs=_tape_knobs(), final_beams=FINAL_BEAMS, step_bytes=STEP_BYTES,
                   first_pass_bytes=FIRST_PASS_BYTES, lenient=LENIENT, silence_rms=SILENCE_RMS)
            print("[tape: %s]" % ascii(self.dir))
        except Exception as ex:
            print("[tape off: %s: %s]" % (type(ex).__name__, ex)); self.f = None
    def w(self, kind, **kv):
        if self.f is None: return
        try:
            self.f.write(json.dumps(dict(k=kind, t=round(time.time() - self.t0, 3), **kv), ensure_ascii=False) + "\n")
            self.f.flush()
        except Exception:
            pass
    def wav(self, pcm):
        if self.f is None: return
        self.n += 1; name = "utt_%04d.wav" % self.n
        try:
            with wave.open(os.path.join(self.dir, name), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(pcm)
            self.w("wav", file=name, len=len(pcm))
        except Exception:
            pass
    def close(self):
        if self.f is None: return
        try:
            self.w("end"); self.f.close()
        except Exception:
            pass
        self.f = None

# --- ASR context prompt: the words recited just before this utterance are given to the model as a prompt ---
ASR_PROMPT = os.getenv("ASR_PROMPT", "ctx")                # ctx (default) or 0 = off
ASR_PROMPT_CTX = int(os.getenv("ASR_PROMPT_CTX", "3"))     # number of preceding words
ASR_PROMPT_DIAC = os.getenv("ASR_PROMPT_DIAC", "1") != "0"   # 0 = strip short vowels too
ASR_PROMPT_MIN_BYTES = int(float(os.getenv("ASR_PROMPT_MIN_S", "1.5")) * BPS)   # no prompt on very short buffers
_PR_DROP = {chr(c) for c in range(0x06D6, 0x06EE)} | {"\u0640", "\u0670", "\u0653", "\u0654", "\u0655"}
_PR_VOWELS = {chr(c) for c in range(0x064B, 0x0653)}
def _pr_style(w):
    """Uthmani text -> the plain diacritized style the model writes."""
    w = w.replace("\u0671", "\u0627")
    return "".join(c for c in w if c not in _PR_DROP and (ASR_PROMPT_DIAC or c not in _PR_VOWELS))
def asr_prompt(st, n):
    """Context for the current utterance, fixed at its first pass. n = bytes in the buffer."""
    if ASR_PROMPT != "ctx" or st.get("find"): return ""
    al = st.get("aligner")
    if al is None: return ""
    if st.get("done", 0) == 0 or n < st.get("pr_n", 0) or "pr_cur" not in st:
        st["pr_cur"] = al.cursor
    st["pr_n"] = n
    if n < ASR_PROMPT_MIN_BYTES: return ""
    cur = st["pr_cur"]
    ws = al.words[max(0, cur - ASR_PROMPT_CTX):cur]
    return " ".join(_pr_style(w["text_uthmani"]) for w in ws).strip()
import re as _re_pr
_PR_DIAC = _re_pr.compile("[\u064B-\u0652\u0670\u0640]")
def _pr_norm(tok):
    s = _PR_DIAC.sub("", tok)
    for a in "\u0625\u0623\u0622\u0671": s = s.replace(a, "\u0627")
    if len(s) > 3 and s[0] == "\u0648": s = s[1:]
    return s
def _pr_rep(t):
    """First repeating loop as (start, n, run) or None. n = period (1-3 words).
    One word: 3 in a row. 2-3 words: about 2.5 repetitions (run >= 2n-1), so a legit
    exact double like 102:3-4 is not flagged."""
    w = [_pr_norm(x) for x in t.split()]
    for i in range(len(w)):
        for n in (1, 2, 3):
            run = 0
            while i + run + n < len(w) and w[i + run] == w[i + run + n]: run += 1
            if run >= (2 if n == 1 else 2 * n - 1): return (i, n, run)
    return None
def _pr_collapse(t):
    w = t.split()
    for _ in range(3):
        r = _pr_rep(" ".join(w))
        if r is None: break
        i, n, run = r
        w = w[:i + n] + w[i + run + n:]
    return " ".join(w)
def transcribe_ctx(pcm, final, prompt):
    """Transcribe with the prompt; on a repeat loop (prompt echo) retry without it, and collapse a loop that remains."""
    if not prompt: return transcribe(pcm, final)
    try:
        t = transcribe(pcm, final, prompt)
        if _pr_rep(t) is not None:
            print("[prompt echo, retried without]")
            t2 = transcribe(pcm, final)
            if _pr_rep(t2) is not None:
                print("[repeat loop collapsed]")
                t2 = _pr_collapse(t2)
            return t2
        return t
    except Exception as e:
        print("[prompt failed, retried without: %s]" % type(e).__name__)
        return transcribe(pcm, final)

# --- forced decoding helpers (used by the letter check) ---
# The ASR model scores a given reading of the audio (CTranslate2 align = forced decoding).
import re as _re_fc, math as _math_fc
_FC_KEEP = "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652"
def _fc_style(u):
    """text_uthmani -> the diacritized style the ASR writes (same rules as trainutil.to_model_style)."""
    out = []
    for ch in u:
        if ch == "\u0671":
            ch = "\u0627"
        if ch == "\u0649":
            j = len(out) - 1
            while j >= 0 and out[j] in _FC_KEEP:
                if out[j] == "\u0650":
                    ch = "\u064A"
                j -= 1
        if ch == " " or ("\u0621" <= ch <= "\u064A" and ch != "\u0640") or ch in _FC_KEEP:
            out.append(ch)
    s = _re_fc.sub(" +", " ", "".join(out)).strip()
    def reorder(mt):
        return "\u0651" + mt.group().replace("\u0651", "")
    return _re_fc.sub("[\u064B-\u0652]{2,}", lambda mt: reorder(mt) if "\u0651" in mt.group() else mt.group(), s)
_FC = {}
def _fc_setup():
    if _FC: return _FC
    try:
        m = get_asr()
        if getattr(m.model, "device", "cuda") == "cpu": raise RuntimeError("GPU only")
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.audio import pad_or_trim
        tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language="ar")
        _FC.update(m=m, tok=tok, pad=pad_or_trim, prefix=list(tok.sot_sequence) + [tok.no_timestamps])
    except Exception as ex:
        print("[forced check disabled: %s: %s]" % (type(ex).__name__, ex))
        _FC.update(off=True)
    return _FC
# --- letter check ---
# A word about to be committed GREEN is re-scored by forced decoding with one letter swapped for a commonly
# confused one; if a swapped reading beats the expected word by more than LETTER_T -> UNSURE (orange).
LETTER_CHECK = os.getenv("LETTER_CHECK", "1") != "0"
LETTER_T = float(os.getenv("LETTER_T", "6"))
LETTER_MAXV = int(os.getenv("LETTER_MAXV", "10"))
LETTER_SKIP_EXACT = os.getenv("LETTER_SKIP_EXACT", "1") != "0"   # skip words the ASR wrote letter-exact
_LC_CONF = {
    "ح": "ه", "ه": "ح",
    "ع": "أ", "أ": "ع", "إ": "ع", "ء": "ع",
    "ق": "ك", "ك": "ق",
    "ص": "س", "س": "صث",
    "ض": "دظ", "د": "ضذ", "ظ": "زضذ",
    "ط": "ت", "ت": "ط",
    "ث": "ست", "ذ": "زد", "ز": "ذ",
    "غ": "خ", "خ": "غح",
}
def _lc_variants(s):
    out = []
    for i, ch in enumerate(s):
        for alt in _LC_CONF.get(ch, ""):
            out.append(s[:i] + alt + s[i + 1:])
    return out[:LETTER_MAXV]
def letter_check(events, n, al, st, snap):
    """Returns [[word index, best swap score], ...] for the checked words; flips a word to UNSURE above LETTER_T."""
    out = []
    try:
        raw, fed = st.get("raw") or [], st.get("fed", 0)
        if not raw or n <= 0 or st.get("done") != len(snap) or len(snap) > int(29.5 * 32000): return out
        rest_words = raw[fed:]
        pos, p = {}, 0
        for i, e in enumerate(events):                      # event -> index of its token inside rest_words
            sp = e.get("spoken_transcription")
            if sp is None: continue
            q = p
            while q < len(rest_words) and rest_words[q] != sp: q += 1
            if q < len(rest_words): pos[i] = q; p = q + 1
        cand = []
        for i in range(min(n, len(events))):
            e = events[i]
            if e["status"] != "CORRECT" or e.get("_ref") is None or i not in pos: continue
            if LETTER_SKIP_EXACT and normalize(e.get("spoken_transcription") or "") == normalize(al.words[e["matched_word_index"]]["text_clean"]):
                continue                                # the ASR wrote it letter-exact, nothing to overrule
            exp = _fc_style(al.words[e["matched_word_index"]]["text_uthmani"])
            vs = _lc_variants(exp)
            if vs: cand.append((i, exp, vs))
        if not cand: return out
        fc = _fc_setup()
        if fc.get("off"): return out
        m, tok = fc["m"], fc["tok"]
        audio = np.frombuffer(snap, np.int16).astype(np.float32) / 32768.0
        fe = m.feature_extractor
        enc = m.encode(fc["pad"](fe(audio)[:, :fe.nb_max_frames]))
        nf = min(int(len(audio) / fe.hop_length), fe.nb_max_frames)
        def score(r_):
            return sum(_math_fc.log(max(x, 1e-9)) for x in list(r_.text_token_probs))
        def lps(seqs):                                      # all readings in batches (same scores, ~40% less time)
            ids = [tok.encode(" " + " ".join(w_)) for w_ in seqs]
            res = [None] * len(seqs)
            ok = [j for j, x in enumerate(ids) if len(x) + len(fc["prefix"]) <= 440]
            for b0 in range(0, len(ok), 16):
                grp = ok[b0:b0 + 16]
                try:
                    import ctranslate2 as _ct2
                    host = np.array(enc.to_device(_ct2.Device.cpu))
                    encb = _ct2.StorageView.from_array(np.repeat(host, len(grp), 0)).to_device(_ct2.Device.cuda)
                    rr = m.model.align(encb, fc["prefix"], [ids[j] for j in grp], [nf] * len(grp))
                    for j, r_ in zip(grp, rr): res[j] = score(r_)
                except Exception:
                    for j in grp: res[j] = score(m.model.align(enc, fc["prefix"], [ids[j]], nf)[0])
            return res
        seqs, idx = [], []
        for i, exp, vs in cand:
            k = fed + pos[i]
            for v in [exp] + vs:
                t2 = list(raw); t2[k] = v; seqs.append(t2)
            idx.append((i, len(seqs) - len(vs) - 1, len(vs)))
        allp = lps(seqs)
        for i, s0, nv in idx:
            le = allp[s0]
            if le is None: continue
            lv = [x for x in allp[s0 + 1:s0 + 1 + nv] if x is not None]
            if not lv: continue
            best = max(lv) - le
            gi = events[i]["matched_word_index"]
            out.append([gi, round(best, 2)])
            if best > LETTER_T:
                events[i]["status"] = "MISTAKE_UNSURE"; events[i]["unmask_next"] = True
                print("[letter check: word %d -> unsure (score %.2f)]" % (gi, best))
    except Exception as ex:
        print("[letter check failed: %s: %s]" % (type(ex).__name__, ex))
    return out

# --- phone check (phonecheck.py) ---
# At the end of every utterance its audio is heard again by a phoneme model (no language model, so it does not
# autocorrect). A word committed GREEN in this utterance in which a commonly confused letter was clearly heard instead
# (ح/ه, ص/س, ق/ك ...) is repainted UNSURE (orange) with a "recheck" event (it does not move the reading position).
PHONE_CHECK = os.getenv("PHONE_CHECK", "1") != "0"
PHONE_T = float(os.getenv("PHONE_T", "0"))         # min probability of the heard letter (0: the model's own default)
PHONE_MAX_WORDS = 60
def _phone_model():
    import phonecheck
    return phonecheck.get(cudaprep.gpu_available() and ct2_gpu_works(CT2_DIR))
def _phone_replaces_lc():
    """the GPU phone check (Muaalem) catches everything the letter check does (bench: identical card with LETTER_CHECK=0)"""
    if not PHONE_CHECK: return False
    m = sys.modules.get("phonecheck")
    m = m._M.get("m") if m else None
    return m is not None and m.name == "muaalem"
def phone_warmup():
    try:
        m = _phone_model()
        if m is not None:
            import phonecheck
            phonecheck.check(m, phonecheck.phon_words(1, 1, ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَـٰنِ", "ٱلرَّحِيمِ"]), np.zeros(32000, np.float32), PHONE_T or m.thr)
            print("Phone check ready (%s)." % m.name)
    except Exception as ex:
        print("[phone check warm-up failed: %s: %s]" % (type(ex).__name__, ex))
def phone_check(al, st, snap):
    """Returns [[word index, heard word], ...] for the words to repaint UNSURE."""
    out = []
    try:
        utt = st.get("utt") or {}
        if not utt or st.get("utt_jump") or st.get("surah") is None: return out
        lo, hi = min(utt), max(utt)
        if hi - lo + 1 > PHONE_MAX_WORDS: return out
        m = _phone_model()
        if m is None: return out
        import phonecheck
        W = al.words
        if getattr(al, "_ayah_idx", None) is None:
            al._ayah_idx = {}
            for i, w in enumerate(W): al._ayah_idx.setdefault(w["ayah_number"], []).append(i)
        cache, words = {}, []
        for i in range(lo, hi + 1):
            a = W[i]["ayah_number"]
            if a not in cache: cache[a] = phonecheck.phon_words(st["surah"], a, [W[j]["text_uthmani"] for j in al._ayah_idx[a]])
            words.append(cache[a][W[i]["word_position"] - 1])
        audio = np.frombuffer(snap, np.int16).astype(np.float32) / 32768.0
        flags, _ = phonecheck.check(m, words, audio, PHONE_T or m.thr)
        for k, sw in sorted(flags.items()):
            gi = lo + k
            if utt.get(gi) != "CORRECT": continue
            out.append([gi, phonecheck.heard_word(W[gi]["text_uthmani"], sw)])
            print("[phone check: word %d -> unsure (%s)]" % (gi, " ".join("%04x>%04x" % (ord(a), ord(b)) for a, b in sw)))
    except Exception as ex:
        print("[phone check failed: %s: %s]" % (type(ex).__name__, ex))
    return out

# --- look-back: revisit 1-2 words committed as mistakes ---
LOOKBACK = os.getenv("LOOKBACK", "1") != "0"
LOOKBACK_MARGIN = float(os.getenv("LOOKBACK_MARGIN", "0.6"))
LOOKBACK_MIN_OK = int(os.getenv("LOOKBACK_MIN_OK", "2"))
def lookback(al, st, rest):
    """Start of an utterance only (nothing consumed yet). Returns a cursor 1-2 words back, or None.
    Only words committed non-CORRECT can be revisited, and the spoken words must align better from
    there (more matches, cost lower by LOOKBACK_MARGIN)."""
    if not LOOKBACK or st.get("fed", 0) != 0 or al.cursor <= 0: return None
    sp = al._spoken(rest)
    if len(sp) < 2: return None
    lb = st.get("lb", {})
    f_cost, f_ok = al._score(sp, al.cursor)
    best = None
    for k in (1, 2):
        c = al.cursor - k
        if c < st.get("range_lo", 0) or c < 0: continue
        if any(lb.get(i, "CORRECT") == "CORRECT" for i in range(c, al.cursor)): continue
        cost, ok = al._score(sp, c)
        if ok > f_ok and ok >= LOOKBACK_MIN_OK and cost + LOOKBACK_MARGIN < f_cost and (best is None or cost < best[0]): best = (cost, c)
    return best[1] if best else None

# --- fed resync: keep the count of consumed tokens right when a new pass re-splits the words ---
from alignment import similarity as _sim
RESYNC_FED = os.getenv("RESYNC_FED", "1") != "0"
RESYNC_MARGIN = float(os.getenv("RESYNC_MARGIN", "0.6"))
RESYNC_LETTERS = os.getenv("RESYNC_LETTERS", "1") != "0"
LETTER_MARGIN = float(os.getenv("LETTER_MARGIN", "0.08"))
RESYNC_SAFE = os.getenv("RESYNC_SAFE", "1") != "0"
# last hold: the last word of an ayah, heard inexactly, waits for one agreeing pass
LAST_HOLD = os.getenv("LAST_HOLD", "1") != "0"
LAST_HOLD_SIM = float(os.getenv("LAST_HOLD_SIM", "1.0"))
LAST_HOLD_END = os.getenv("LAST_HOLD_END", "1") == "1"   # only the last word of an ayah
def _ayah_end(words, i):
    return i + 1 >= len(words) or words[i + 1]["ayah_number"] != words[i]["ayah_number"]
def _lev(a, b):
    i = 0                                   # equal affixes cost nothing: drop them first (63 ms -> 0.2 ms)
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]: i += 1
    a, b = a[i:], b[i:]
    j = 0
    n = min(len(a), len(b))
    while j < n and a[-1 - j] == b[-1 - j]: j += 1
    if j: a, b = a[:-j], b[:-j]
    p = list(range(len(b) + 1))
    for x in range(1, len(a) + 1):
        q = [x] + [0] * len(b)
        for y in range(1, len(b) + 1):
            q[y] = min(p[y] + 1, q[y - 1] + 1, p[y - 1] + (a[x - 1] != b[y - 1]))
        p = q
    return p[-1]
def _next_key(st):
    try:
        al = st["aligner"]; return al._key(al.words[al.cursor])
    except Exception: return None
def _letters_j(old, new, f, nxt=None):
    """Best number of leading tokens of `new` that cover the consumed text `old` (f tokens), judged on
    joined letters. Returns j (== f when keeping it is as good, within LETTER_MARGIN), or None when
    nothing fits well enough (caller falls back to the token alignment)."""
    S = "".join(old)
    if not S: return None
    n = len(new)
    cost = {}
    for j in range(max(0, f - 3), min(n, f + 4) + 1):
        cost[j] = _lev(S, "".join(new[:j])) / max(len(S), 4.0)
    if not cost: return None
    b = min(cost, key=lambda j: (cost[j] + 0.01 * abs(j - f), j))
    if cost[b] > 0.5: return None
    if b > f and nxt and _sim(new[b - 1], nxt) >= 0.6: return f   # last token is the next expected word, not consumed text
    if RESYNC_SAFE and b > f and nxt and len(new[b - 1]) >= 2 and _sim(new[b - 1], nxt[:len(new[b - 1])]) >= 0.6:
        return f                            # last token = the start of the next expected word
    if b != f and f in cost and cost[f] - cost[b] < LETTER_MARGIN: return f
    return b

def resync_fed(st):
    """Re-align the tokens already consumed (st['done_toks']) with the prefix of the new transcript; fix st['fed']."""
    if not RESYNC_FED: return
    f = st.get("fed", 0)
    if f <= 0:
        st["done_toks"] = []; return
    old = st.get("done_toks") or []
    new = [normalize(t) for t in st["raw"]]
    if RESYNC_SAFE and len(new) < f and len("".join(new)) < len("".join(st.get("done_toks") or [])):
        return                                # truncated pass (fewer letters): change nothing
    if len(old) != f:                       # e.g. basmala tokens were consumed: trust fed, just record it
        st["done_toks"] = new[:f]; return
    m, n = len(old), len(new)
    if RESYNC_LETTERS:
        jl = _letters_j(old, new, f, _next_key(st))
        if jl is not None:
            if jl != f: print("[fed resync (letters): %d -> %d]" % (f, jl)); st["fed"] = jl
            st["done_toks"] = new[:st["fed"]]; return
    INF = 1e9
    D = [[INF] * (n + 1) for _ in range(m + 1)]
    D[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            d = D[i][j]
            if d >= INF: continue
            if i < m and d + 1.0 < D[i + 1][j]: D[i + 1][j] = d + 1.0
            if j < n and d + 1.0 < D[i][j + 1]: D[i][j + 1] = d + 1.0
            if i < m and j < n:
                c = min(2.0, 2.0 * (1.0 - _sim(old[i], new[j])))
                if d + c < D[i + 1][j + 1]: D[i + 1][j + 1] = d + c
    hi = min(n, m + 3)
    lo = min(max(0, m - 3), hi)
    j = min(range(lo, hi + 1), key=lambda x: (D[m][x], abs(x - m)))
    if D[m][j] <= 0.6 * m + 1.0 and j != f:
        if f <= n and D[m][j] + RESYNC_MARGIN > D[m][f]:   # keep fed: garbled token only
            print("[fed resync held: %d (not %d)]" % (f, j))
            st["done_toks"] = new[:f]; return
        print("[fed resync: %d -> %d]" % (f, j))
        st["fed"] = j
    st["done_toks"] = new[:st["fed"]]

# --- merge fix: a word the ASR swallows into the previous one ---
MERGE_FIX = os.getenv("MERGE_FIX", "1") != "0"
MERGE_PAIRS = {("\u0645\u0646", "\u0630\u0627")}     # (previous word, word the ASR swallows): man + dha
def merge_fix(events, al):
    for e in events:
        if e["status"] != "MISTAKE_SKIPPED": continue
        gi = e["matched_word_index"]
        if gi < 1: continue
        pair = (normalize(al.words[gi - 1]["text_uthmani"]), normalize(al.words[gi]["text_uthmani"]))
        if pair not in MERGE_PAIRS: continue
        prev_ok = gi - 1 < al.cursor or any(x["matched_word_index"] == gi - 1 and x["status"] in ("CORRECT", "MISTAKE_UNSURE") for x in events)
        if prev_ok:
            e["status"] = "CORRECT"; e["unmask_next"] = False

# --- wasl fix: connected speech drops the article's alif (hamzat al-wasl) ---
WASL_FIX = os.getenv("WASL_FIX", "1") != "0"
_AL = "ال"
def _wasl_join(ka, kb):
    """Spoken form of A + B when B starts with the article: A's final long vowel and B's alif are silent."""
    a = ka[:-1] if len(ka) > 1 and ka[-1] in "اوي" else ka
    return a + kb[1:]
def wasl_fix(events, al, st):
    fed = st.get("fed", 0)
    seen = False
    for e in events:
        if e["status"] in ("MISTAKE_SUBSTITUTION", "MISTAKE_UNSURE") and e["_spk"] and e.get("_ref") is not None:
            k = normalize(al.words[e["matched_word_index"]]["text_clean"])
            h = normalize(e["spoken_transcription"] or "")
            if (fed > 0 or seen) and k.startswith(_AL) and len(k) > 3 and h == k[2:]:
                e["status"] = "CORRECT"; e["unmask_next"] = False
                print("[wasl fix: word %d, article not written]" % e["matched_word_index"])
        if e["_spk"]: seen = True

# --- span fix: a word boundary the ASR put in the wrong place ---
SPAN_FIX = os.getenv("SPAN_FIX", "1") != "0"
SPAN_MIN = float(os.getenv("SPAN_MIN", "0.8"))
SPAN_GAIN = float(os.getenv("SPAN_GAIN", "0.1"))
def span_fix(events, al, st):
    lb, raw, fed = st.get("lb", {}), st.get("raw", []), st.get("fed", 0)
    okp = ("CORRECT", "MISTAKE_UNSURE")
    for idx, e in enumerate(events):                      # rule 1: fused token swallowed a word
        if e["status"] != "MISTAKE_SKIPPED": continue
        gi = e["matched_word_index"]
        if gi < 1: continue
        prev = next((x for x in events[:idx] if x["matched_word_index"] == gi - 1 and x["_spk"] and x["status"] in okp), None)
        if prev is not None: heard = normalize(prev["spoken_transcription"] or "")
        elif idx == 0 and gi == al.cursor and 0 < fed <= len(raw) and lb.get(gi - 1, "CORRECT") in okp: heard = normalize(raw[fed - 1])
        else: continue
        if not heard: continue
        ka, kb = al._key(al.words[gi - 1]), al._key(al.words[gi])
        s2, s1 = al._sim(heard, ka + kb), al._sim(heard, ka)
        if s2 >= SPAN_MIN and s2 >= s1 + SPAN_GAIN:
            e["status"] = "CORRECT"; e["unmask_next"] = False
            print("[span fix: word %d joined with %d (%.2f vs %.2f)]" % (gi, gi - 1, s2, s1))
        elif DEBUG_ALIGN: print("[span no: word %d s2=%.2f s1=%.2f]" % (gi, s2, s1))
    sub = ("MISTAKE_SUBSTITUTION", "MISTAKE_UNSURE")
    i = 0
    while i + 1 < len(events):                            # rule 2: boundary slipped between two words
        a, b = events[i], events[i + 1]
        if (a["status"] in sub and b["status"] in sub and a["_spk"] and b["_spk"]
                and b["matched_word_index"] == a["matched_word_index"] + 1):
            ha, hb = normalize(a["spoken_transcription"] or ""), normalize(b["spoken_transcription"] or "")
            ka, kb = al._key(al.words[a["matched_word_index"]]), al._key(al.words[b["matched_word_index"]])
            if len(ha) >= 2 and len(hb) >= 2:
                s = al._sim(ha + hb, ka + kb)
                if WASL_FIX and kb.startswith(_AL): s = max(s, al._sim(ha + hb, _wasl_join(ka, kb)))
                if s >= SPAN_MIN:
                    for x in (a, b): x["status"] = "CORRECT"; x["unmask_next"] = False
                    print("[span fix: words %d-%d joined (%.2f)]" % (a["matched_word_index"], b["matched_word_index"], s))
                    i += 1
        i += 1

@app.websocket("/ws/recite")
async def recite(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    buf = bytearray()
    st = {}
    def reset():
        st.update(raw=[], fed=0, done=0, speech=False, prev_sigs=[], beamed=False, jump_cand=None, jump_none=0, utt={}, utt_jump=False)
    reset(); st["aligner"] = None
    tape = Tape(); _send = ws.send_json
    async def _tsend(data, *a, **k):
        tape.w("out", m=data); return await _send(data, *a, **k)
    if tape.f is not None: ws.send_json = _tsend

    async def feed(words):                       # feed in slices so the alignment window never truncates
        for i in range(0, len(words), 8):
            for ev in st["aligner"].feed(" ".join(words[i:i + 8])):
                await ws.send_json(ev)

    async def worker():
        while True:
            await asyncio.sleep(0.03)
            al = st["aligner"]
            if (al is None and not st.get("find")) or len(buf) < MIN_BYTES: continue
            try:
                if not st["speech"]:             # wait for speech; trim leading silence (Whisper hallucinates on it)
                    region = bytes(buf)
                    first = next((i for i in range(0, len(region) - 3199, 3200) if rms(region[i:i + 3200]) >= SILENCE_RMS), None)
                    if first is None:
                        if len(buf) > 2 * MIN_BYTES: del buf[:-TAIL_BYTES]
                        continue
                    del buf[:max(0, first - PRE_ROLL)]; st.update(speech=True, done=0)
                new = len(buf) - st["done"]
                # the longer the utterance, the shorter the pause that ends it -> bounded decoding cost and faster flushes
                tail = TAIL_BYTES if len(buf) < 3 * BPS else int((0.35 if len(buf) < 6 * BPS else 0.25) * BPS)
                final = peak_rms(bytes(buf[-tail:])) < SILENCE_RMS or len(buf) > MAX_BYTES
                if final and len(buf) - tail < int(0.5 * BPS):    # a click/cough, not recitation (pre-roll 0.3 s + <0.2 s of sound)
                    tape.w("drop", len=len(buf)); buf.clear(); reset(); continue
                if not final and (len(buf) < FIRST_PASS_BYTES or new < STEP_BYTES): continue

                snap = bytes(buf)
                _tp = {"len": len(snap), "final": final, "asr_ms": None}
                if new > 0 or (final and not st["beamed"]):
                    t0 = time.time()
                    text = await loop.run_in_executor(None, transcribe_ctx, snap, final, asr_prompt(st, len(snap)))
                    took = time.time() - t0
                    print(f"[{len(snap)/BPS:4.1f}s audio | {took:4.1f}s to transcribe | final={final}] {text}")
                    st.update(raw=text.split(), done=len(snap), beamed=final)
                    _tp.update(asr_ms=round(took * 1000), text=text)
                    resync_fed(st)
                    await ws.send_json({"event": "transcript", "text": text, "final": final, "rms": round(peak_rms(snap)), "took": round(took, 1)})
                    if st.get("find"):                       # find-by-voice mode: the aligner is bypassed
                        try:
                            await ws.send_json(await loop.run_in_executor(None, find_payload, text, final))
                        except Exception:
                            traceback.print_exc()
                        if final: del buf[:len(snap)]; reset()
                        continue
                _ta = time.perf_counter(); _tp.update(cur0=al.cursor, fed0=st["fed"])
                rest_words = st["raw"][st["fed"]:]
                if BASMALA_ON and st.get("bas_ok"):                      # swallow a spoken basmala at the very start of a surah
                    if al.cursor != 0: st["bas_n"], st["bas_done"] = 0, False   # cursor left the start: re-arm for a later restart
                    elif not st["bas_done"] and rest_words:
                        k, st["bas_n"], st["bas_done"], rest_words = strip_basmala(rest_words, st["bas_n"], final)
                        if k:
                            st["fed"] += k
                            if BASMALA_EVENT: await ws.send_json({"event": "basmala", "count": st["bas_n"], "done": st["bas_done"]})
                rest = " ".join(rest_words)
                jumped = False
                lb_to = lookback(al, st, rest)
                if lb_to is not None:
                    print("[lookback: cursor %d -> %d]" % (al.cursor, lb_to))
                    al.cursor = lb_to; st.update(prev_sigs=[], jump_cand=None, jump_none=0)
                jp = al.suggest_jump(rest)               # restart / rewind / far jump? (needs two agreeing passes, or the final pass)
                if jp is not None and not jump_allowed(jp, len(rest_words), st): jp = None
                if final and jp is not None and jp != st["jump_cand"]: jp = None; st["jump_cand"] = None   # the final pass needs an agreeing earlier pass
                if jp is not None and (final or jp == st["jump_cand"]):
                    print("[jump accepted: cursor %d -> %d, final=%s, fed=%d, rest=%s, fwd=%s, cand=%s]" % (al.cursor, jp, final, st["fed"], ascii(" ".join(rest_words)), al._score(al._spoken(rest), al.cursor), al._score(al._spoken(rest), jp))); al.cursor = jp; st.update(prev_sigs=[], jump_cand=None, jump_none=0)
                    await ws.send_json({"event": "jump", "to_index": jp}); jumped = True; st["utt_jump"] = True
                elif jp is not None:
                    st["jump_cand"], st["jump_none"] = jp, 0
                else:                                    # live hypotheses flicker: forget a suspected restart only after 3 misses
                    st["jump_none"] += 1
                    if st["jump_none"] >= 3: st["jump_cand"] = None
                events = al.align(rest)
                if MERGE_FIX: merge_fix(events, al)
                if SPAN_FIX: span_fix(events, al, st)
                if WASL_FIX: wasl_fix(events, al, st)
                speech_bytes = len(snap) - (tail if final else 0)
                short = speech_bytes < SHORT_CLIP_BYTES
                if short:                                         # short clips are unreliable: doubt, don't accuse
                    for e in events:
                        if e["status"] in ("MISTAKE_SUBSTITUTION", "MISTAKE_SKIPPED"): e["status"] = "MISTAKE_UNSURE"
                sigs = [(e["status"], e["matched_word_index"], normalize(e["spoken_transcription"] or "")) for e in events]
                if final:
                    n = len(events)
                    if n and all(e["status"] != "CORRECT" for e in events) and speech_bytes < int(0.8 * BPS):
                        n = 0                                     # very short utterance matching nothing = noise/hallucination
                else:
                    # Correct words are committed immediately. A mistake is committed only when (a) two passes agree on it,
                    # (b) it is not the last word heard, and (c) a correct word follows it - otherwise early partial guesses
                    # (e.g. 1-2 words) would be wrongly locked in as "extra words".
                    n = 0
                    last_ok = max((i for i, e in enumerate(events) if e["status"] == "CORRECT"), default=-1)
                    for i, e in enumerate(events):
                        stable = i < len(events) - 1 and i < len(st["prev_sigs"]) and st["prev_sigs"][i] == sigs[i]
                        same = i < len(st["prev_sigs"]) and st["prev_sigs"][i] == sigs[i]
                        if MUQ_HOLD and e["status"] == "CORRECT" and not same and is_muq(st.get("surah"), al.words[e["matched_word_index"]]): break   # a letters opening needs 2 agreeing passes
                        if (LAST_HOLD and e["status"] == "CORRECT" and i == len(events) - 1 and e.get("_ref") is not None
                                and (not LAST_HOLD_END or _ayah_end(al.words, e["matched_word_index"]))
                                and not (i < len(st["prev_sigs"]) and st["prev_sigs"][i][:2] == sigs[i][:2])
                                and _sim(sigs[i][2], normalize(al.words[e["matched_word_index"]]["text_clean"])) < LAST_HOLD_SIM):
                            break                      # last word heard, inexact: wait for one agreeing pass
                        if e["status"] != "CORRECT" and not (stable and i < last_ok and not short): break   # no accusations from short clips until final
                        n = i + 1
                if not jumped and (jp is not None or st["jump_cand"] is not None): n = 0   # restart suspected, not confirmed yet: commit nothing against the old position
                _tal = round((time.perf_counter() - _ta) * 1000, 1)
                if LETTER_CHECK and n and not jumped and not _phone_replaces_lc():
                    _tl = time.perf_counter()
                    _tp["lc"] = await loop.run_in_executor(None, letter_check, events, n, al, st, snap)
                    _tp["lc_ms"] = round((time.perf_counter() - _tl) * 1000, 1)
                if n:
                    st["fed"] += al.commit(events[:n])
                    lbd = st.setdefault("lb", {})
                    for e in events[:n]:
                        if e["_ref"] is not None: lbd[e["matched_word_index"]] = e["status"]
                        if e["status"] != "MISTAKE_INSERTION": st["utt"][e["matched_word_index"]] = e["status"]
                    for k_ in sorted(lbd)[:-8]: del lbd[k_]
                    st["done_toks"] = [normalize(t) for t in st["raw"][:st["fed"]]]
                    for e in events[:n]:
                        # LENIENT: a 1-2 letter extra token is almost always an ASR word-ending fragment, not a real extra word
                        if e["status"] == "MISTAKE_INSERTION" and (short or (LENIENT and len(normalize(e["spoken_transcription"] or "")) <= 2)): continue
                        await ws.send_json(public(e))
                st["prev_sigs"] = sigs[n:]
                if final and PHONE_CHECK and st["utt"]:
                    _tq = time.perf_counter()
                    _tp["pc"] = await loop.run_in_executor(None, phone_check, al, st, snap)
                    _tp["pc_ms"] = round((time.perf_counter() - _tq) * 1000, 1)
                    for gi, heard in _tp["pc"]:
                        st.setdefault("lb", {})[gi] = "MISTAKE_UNSURE"
                        await ws.send_json({"event": "recheck", "matched_word_index": gi, "status": "MISTAKE_UNSURE",
                                            "expected_word": al.words[gi]["text_uthmani"], "spoken_transcription": heard})
                tape.w("pass", align_ms=_tal, lookback=lb_to, jump=jp, jumped=jumped, short=short, n=n, cur=al.cursor,
                       fed=st["fed"], ev=[[e["matched_word_index"], e["status"], e["spoken_transcription"], e["_ref"], e["_spk"]]
                                          for e in events], **_tp)
                if DEBUG_ALIGN: print("[dbg] final=%s raw=%d fed=%d rest=%d n=%d/%d cur=%d | %s" % (final, len(st["raw"]), st["fed"], len(rest_words), n, len(events), al.cursor, " ".join("%d:%s%s" % (e["matched_word_index"], {"CORRECT": "ok", "MISTAKE_SUBSTITUTION": "SUB", "MISTAKE_SKIPPED": "DEL", "MISTAKE_INSERTION": "INS", "MISTAKE_UNSURE": "UNS"}.get(e["status"], "?"), "*" if k < n else "") for k, e in enumerate(events))))
                if final:
                    tape.wav(snap)
                    del buf[:len(snap)]; reset()
            except Exception as e:               # never let one bad chunk kill the connection
                traceback.print_exc(); buf.clear(); reset()
                tape.w("error", msg=repr(e))
                try: await ws.send_json({"event": "error", "message": str(e)})
                except Exception: return

    task = asyncio.create_task(worker())
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("event") != "audio_chunk": tape.w("in", m=msg)
            if msg["event"] == "start":          # also sent after every reconnect
                words = load_surah(msg["surah_id"])
                st["aligner"] = Aligner(words, start_index(words, msg["starting_ayah"], msg["expected_word_index"]),
                                        strict=msg.get("strict", False), lenient=LENIENT)
                buf.clear(); reset(); st["find"] = False
                _c = st["aligner"].cursor   # lowest start seen for this surah = range start
                st["range_lo"] = min(_c, st["range_lo"]) if st.get("range_sid") == msg["surah_id"] and st.get("range_lo") is not None else _c
                st["range_sid"] = msg["surah_id"]
                try: st["bas_ok"] = int(msg["surah_id"]) not in (1, 9)      # surah 1 has the basmala as ayah 1, surah 9 has none
                except Exception: st["bas_ok"] = False
                try: st["surah"] = int(msg["surah_id"])
                except Exception: st["surah"] = None
                st["bas_n"], st["bas_done"] = 0, False
            elif msg["event"] == "find_start":
                st["find"] = True; buf.clear(); reset()
                await loop.run_in_executor(None, get_finder)         # builds the index once (~0.3 s)
                await ws.send_json({"event": "find_ready"})
            elif msg["event"] == "find_cancel":
                st["find"] = False; buf.clear(); reset()
            elif msg["event"] == "debug_text" and st["aligner"] and DEBUG_TEXT:
                await feed(msg["text"].split())
            elif msg["event"] == "audio_chunk" and (st["aligner"] or st.get("find")):
                buf += base64.b64decode(msg["pcm_data_base64"])
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        tape.close()

# Serve the web client at http://localhost:8000/  (keep at the end of the file)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "client-web"), html=True), name="web")