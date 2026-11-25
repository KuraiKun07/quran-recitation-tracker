# phonecheck.py - letter-level check of what was recited, with a phoneme model. A phoneme model has no language model:
# it writes the sounds that were said, so it cannot "autocorrect" a wrong letter the way the Whisper transcript does.
#   GPU : Muaalem phoneme head (obadx/muaalem-model-v3_2, MIT), ONNX float16, onnxruntime CUDA
#   CPU : HisnLabs tajweed-v2 (Apache-2.0, distilled from Muaalem), ONNX fp32, onnxruntime CPU
#   rules: third_party/imam.py (HisnLabs, Apache-2.0) aligns the heard phonetic script with the expected one
#          (quran-transcript phonetizer) and lists the clear mistakes (lahn jali) of every word
#   filter: only a consonant swapped for a commonly confused one (the letter check's pairs), heard with
#          probability >= threshold. Everything else the model disagrees on is ignored (mostly model noise).
# Files: backend\phone-model\ (built by phone_convert.py). Everything here fails soft: no model -> no check.
import json, os, re, sys, threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONS = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
PAIRS = {frozenset(p) for p in ["حه", "عء", "قك", "صس", "سث", "ضد", "ضظ", "دذ", "ظز", "ظذ", "طت", "ثت", "ذز", "غخ", "خح"]}
_WAQF = re.compile("[ۖ-۞۩]")


def _ort():
    d = os.getenv("PHONE_ORT_DIR")                  # dev: onnxruntime-gpu kept outside the app's Python
    if d and d not in sys.path:
        sys.path.insert(0, d)
    import onnxruntime
    return onnxruntime


def _runs(ids, top, label):
    """CTC best path: (script, mean probability of each symbol over its frames); blank = 0"""
    txt, probs, t = [], [], 0
    while t < len(ids):
        u = t
        while u < len(ids) and ids[u] == ids[t]:
            u += 1
        if ids[t] != 0:
            txt.append(label(int(ids[t]))); probs.append(float(top[t:u].mean()))
        t = u
    return "".join(txt), probs


class Muaalem:
    name = "muaalem"
    thr = 0.7                                        # bench: +5 false oranges, 23 -> 38 of 46 mistakes caught

    def __init__(self, d):
        ort = _ort()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s = ort.InferenceSession(os.path.join(d, "muaalem-phonemes-fp16.onnx"), so,
                                      providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})])
        if "CUDAExecutionProvider" not in self.s.get_providers():
            raise RuntimeError("onnxruntime CUDA provider not available")
        self.labels = json.load(open(os.path.join(d, "phone_config.json"), encoding="utf-8"))["labels"]
        z = np.load(os.path.join(d, "phone_fbank.npz"))
        self.mel, self.win = z["mel"].astype(np.float64), z["window"].astype(np.float64)

    def features(self, x):
        """SeamlessM4T / Kaldi fbank: the same numbers as transformers' SeamlessM4TFeatureExtractor, numpy only."""
        w = np.asarray(x, np.float64) * 32768.0
        n = 1 + (len(w) - 400) // 160
        if n < 4:
            return None
        fr = w[np.arange(400)[None, :] + 160 * np.arange(n)[:, None]]
        fr = fr - fr.mean(1, keepdims=True)
        fr[:, 1:] = fr[:, 1:] - 0.97 * fr[:, :-1]
        fr[:, 0] *= 1 - 0.97
        fr *= self.win
        pw = np.abs(np.fft.rfft(fr, n=512).astype(np.complex64), dtype=np.float64) ** 2
        f = np.log(np.maximum(1.192092955078125e-07, pw @ self.mel)).astype(np.float32)
        f = (f - f.mean(0, keepdims=True)) / np.sqrt(f.var(0, ddof=1, keepdims=True) + 1e-7)
        if len(f) % 2:                               # transformers pads to an even frame count (value 1)
            f = np.concatenate([f, np.ones((1, f.shape[1]), np.float32)])
        return f.reshape(len(f) // 2, 160)

    def hear(self, x):
        f = self.features(x)
        if f is None:
            return "", []
        lp = self.s.run(None, {"features": f[None]})[0][0]
        ids = lp.argmax(-1)
        return _runs(ids, np.exp(lp[np.arange(len(ids)), ids]), lambda i: self.labels[i])


class TajweedV2:
    name = "tajweed-v2"
    thr = 0.85                                       # bench (CPU, no letter check): +10 false oranges, 10 -> 32 caught
    LEAD = int(16000 * 0.4)                          # its training clips start and end with silence

    def __init__(self, d):
        ort = _ort()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(os.getenv("PHONE_THREADS", "2"))
        self.s = ort.InferenceSession(os.path.join(d, "tajweed-v2-fp32.onnx"), so, providers=["CPUExecutionProvider"])
        self.labels = json.load(open(os.path.join(d, "tajweed-v2.json"), encoding="utf-8"))["phonemes"]
        from faster_whisper.feature_extractor import FeatureExtractor
        self.fe = FeatureExtractor(feature_size=80)

    def hear(self, x):
        audio = np.concatenate([np.zeros(self.LEAD, np.float32), np.asarray(x, np.float32), np.zeros(self.LEAD, np.float32)])
        mel = self.fe(audio, padding=0)
        mel = mel[:, : min(3000, mel.shape[1] // 2 * 2)]
        lp = self.s.run(None, {"mel": mel[None].astype(np.float32), "riwaya": np.array([0], np.int64)})[0][0]
        ids = lp.argmax(-1)
        return _runs(ids, np.exp(lp[np.arange(len(ids)), ids]), lambda i: self.labels[i - 1])


_M = {}
_LOCK = threading.Lock()
def get(gpu):
    """the phoneme model for this PC (Muaalem on a CUDA GPU, else tajweed-v2), or None. Loaded once."""
    with _LOCK:
        if "m" in _M:
            return _M["m"]
        d = os.getenv("PHONE_DIR") or os.path.join(HERE, "phone-model")
        want = os.getenv("PHONE_MODEL", "auto")           # auto | muaalem | tajweed-v2
        m, why = None, []
        for cls in (Muaalem, TajweedV2):
            if want not in ("auto", cls.name) or (cls is Muaalem and want == "auto" and not gpu):
                continue
            try:
                m = cls(d); break
            except Exception as ex:
                why.append("%s: %s: %s" % (cls.name, type(ex).__name__, str(ex)[:160]))
        print("[phone check: %s%s]" % (m.name if m else "off", (" | " + " | ".join(why)) if why else ""))
        _M["m"] = m
        return m


_TEXT = {}
def phon_words(surah, ayah, fallback):
    """the ayah's words in the phonetizer's own Uthmani spelling (same word count as quran.db), else ours"""
    if (surah, ayah) not in _TEXT:
        try:
            from quran_transcript import Aya
            _TEXT[(surah, ayah)] = Aya(surah, ayah).get().uthmani.split(" ")
        except Exception:
            _TEXT[(surah, ayah)] = None
    w = _TEXT[(surah, ayah)]
    return w if w and len(w) == len(fallback) else [_WAQF.sub("", t).strip() or t for t in fallback]


def swaps(note):
    """consonant swaps of a 'letter' note: [(expected, heard), ...] or None"""
    if note.get("rule") != "letter":
        return None
    e, g = note.get("expected") or "", note.get("heard") or ""
    if not e or len(e) != len(g):
        return None
    out = [(a, b) for a, b in zip(e, g) if a != b]
    if not out or any(a not in CONS or b not in CONS for a, b in out):
        return None
    return out


def check(model, words, audio, thr):
    """words: the Uthmani words of this utterance (phonetizer spelling); audio: float32 16 kHz.
    Returns ({word position: [(expected letter, heard letter), ...]}, heard script)."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from third_party import imam
    heard, probs = model.hear(audio)
    if not heard or not words:
        return {}, heard
    ref = imam.recited(words, heard)
    res = imam.listen(ref, heard, heard_probs=probs, min_prob=0.3, min_prob_jali=thr, langs=("en",))
    out = {}
    for w in res:
        for n in w["notes"]:
            if n.get("severity") != "jali":
                continue
            for s in swaps(n) or []:
                if frozenset(s) in PAIRS:
                    out.setdefault(w["index"], []).append(s)
    return out, heard


def heard_word(word, sw):
    """the expected word with the swapped letter(s): what was heard"""
    for a, b in sw:
        for e_ in ((a,) if a != "ء" else ("ء", "أ", "إ", "ؤ", "ئ")):
            if e_ in word:
                word = word.replace(e_, b, 1); break
    return word
