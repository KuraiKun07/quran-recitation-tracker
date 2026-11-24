"""v1.3: spoken basmala at the start of a surah.

strip_basmala() is a pure function used by server.py's worker. It consumes the leading spoken words that match the
basmala (بسم الله الرحمن الرحيم) so the aligner never sees them. Tolerant per word (the ASR garbles/truncates the
last word, e.g. رحيم for الرحيم). On non-final passes the last heard word is never consumed (it is still being
decoded), and while the basmala is half-heard nothing is handed to the aligner (held back).
"""
from alignment import normalize, similarity

BASMALA = ["بسم", "الله", "الرحمن", "الرحيم"]
MIN_SIM = 0.6

def _no_al(w: str) -> str:
    return w[2:] if w.startswith("ال") and len(w) > 3 else w

def word_matches(heard: str, target: str) -> bool:
    h = normalize(heard)
    if not h:
        return False
    return max(similarity(h, target), similarity(_no_al(h), _no_al(target))) >= MIN_SIM

def strip_basmala(words, n, final):
    """words: spoken words not yet fed; n: basmala words already consumed (0-4); final: last pass of the utterance.
    Returns (consumed, new_n, done, remaining): `consumed` spoken words swallowed (advance st['fed'] by it),
    `done` = basmala phase over, `remaining` = words to hand to the aligner this pass."""
    i, held = 0, False
    while i < len(words) and n < len(BASMALA):
        if not normalize(words[i]):                 # no letters at all: swallow
            i += 1; continue
        if not word_matches(words[i], BASMALA[n]):
            break
        if not final and i == len(words) - 1:       # last word of a live partial: not trustworthy yet
            held = True; break
        i += 1; n += 1
    if n >= len(BASMALA):
        return i, n, True, list(words[i:])
    if held:
        return i, n, False, []
    if i >= len(words):                             # everything swallowed, basmala not complete (e.g. paused mid-way)
        return i, n, False, []
    # words[i] does not match the next basmala word
    left = len(words) - i
    if n == 0:
        # no basmala heard (yet). A lone unmatched word on a partial may still turn into بسم: pass it through, stay armed.
        return i, n, (final or left >= 2), list(words[i:])
    if final or left >= 2:                          # basmala started then something else: it is over
        return i, n, True, list(words[i:])
    return i, n, False, []                          # half-heard basmala, last word garbled on a partial: wait
