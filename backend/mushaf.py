"""Mushaf layout API (v1.1 part 1).  Put next to server.py, with divisions.json.

  GET /api/page/{1..604}  -> {"page","cap","rows":[{"t":"header","s"} | {"t":"basmala","s"} | {"t":"text","w":[...]}]}
        each word: {"i": index in its surah (same index as /surah/{n}), "s": surah, "a": ayah, "w": text, "e": 1 if last word of ayah}
  GET /api/units          -> {"1/8":[{n,juz,part,parts,from:[s,a],to:[s,a],page_from,page_to}], "1/4":[...], "1/2":[...], "1":[...]}
Check your database:  python mushaf.py --check
"""
import json, os, sqlite3, sys
from fastapi import HTTPException

HERE = os.path.dirname(os.path.abspath(__file__))
NO_BASMALA = (1, 9)                      # Al-Fatihah (basmala is ayah 1) and At-Tawbah (none)
SIZES = {"1/8": 1, "1/4": 2, "1/2": 4, "1": 8}   # in rub' al-hizb; a juz has 8
cap_of = lambda page: 8 if page in (1, 2) else 15

def build(rows, ayah_counts, rub):
    """rows: (surah, ayah, pos, text, page, line) ordered by surah, ayah, pos.  Returns (pages, units)."""
    lines, counter, first_page = {}, {}, {}          # lines[page][line] -> words
    for k, (s, a, pos, txt, pg, ln) in enumerate(rows):
        i = counter[s] = counter.get(s, -1) + 1
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        end = 1 if (nxt is None or (nxt[0], nxt[1]) != (s, a)) else 0
        lines.setdefault(pg, {}).setdefault(ln, []).append({"i": i, "s": s, "a": a, "w": txt, "e": end})
        first_page.setdefault((s, a), pg)
    n_pages = max(lines)
    items = {}
    for pg in range(1, n_pages + 1):
        out = []
        for ln in sorted(lines.get(pg, {})):
            ws = lines[pg][ln]
            if ws[0]["i"] == 0:                      # first word of a surah -> header (+ basmala) goes before this line
                s = ws[0]["s"]
                out.append({"t": "header", "s": s})
                if s not in NO_BASMALA: out.append({"t": "basmala", "s": s})
            out.append({"t": "text", "w": ws})
        items[pg] = out
    for pg in range(2, n_pages + 1):                 # a page that is over-full pushes its first header to the end of the previous page
        over = len(items[pg]) - cap_of(pg)
        if over > 0:
            if over != 1 or items[pg][0]["t"] != "header":
                raise ValueError(f"page {pg}: cannot place {over} extra line(s)")
            items[pg - 1].append(items[pg].pop(0))
    bad = [pg for pg in items if len(items[pg]) != cap_of(pg)]
    if bad: raise ValueError(f"pages without exactly {cap_of(3)} lines: {bad[:10]}")
    prev = lambda s, a: (s, a - 1) if a > 1 else (s - 1, ayah_counts[s - 1])
    units = {}
    for name, k in SIZES.items():
        lst = []
        for m in range(0, len(rub), k):
            frm = tuple(rub[m]); to = prev(*rub[m + k]) if m + k < len(rub) else (114, ayah_counts[114])
            lst.append({"n": m // k + 1, "juz": m // 8 + 1, "part": (m % 8) // k + 1, "parts": 8 // k, "from": list(frm), "to": list(to),
                        "page_from": first_page[frm], "page_to": first_page[to]})
        units[name] = lst
    return items, units

_cache = {}
def _state(db_path):
    if "v" not in _cache:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute("SELECT surah, ayah, pos, text_uthmani, page, line FROM words ORDER BY surah, ayah, pos").fetchall()
            ayahs = dict(con.execute("SELECT surah, ayahs FROM surahs").fetchall())
        finally: con.close()
        rub = json.load(open(os.path.join(HERE, "divisions.json"), encoding="utf-8"))["rub"]
        _cache["v"] = build(rows, ayahs, rub)
    return _cache["v"]

def register(app, db_path):
    def get():
        try: return _state(db_path)
        except (OSError, sqlite3.Error, ValueError, KeyError) as e:
            raise HTTPException(503, f"Mushaf layout unavailable: {e}. Check divisions.json and re-run build_db.py.")
    @app.get("/api/page/{page}")
    def api_page(page: int):
        items, _ = get()
        if page not in items: raise HTTPException(404, f"Page {page} does not exist (1-{max(items)}).")
        return {"page": page, "cap": cap_of(page), "rows": items[page]}
    @app.get("/api/units")
    def api_units(): return get()[1]

if __name__ == "__main__" and "--check" in sys.argv:
    db = next((a for a in sys.argv[1:] if not a.startswith("--")), os.path.join(HERE, "quran.db"))
    items, units = _state(db)
    kinds = [r["t"] for p in items.values() for r in p]
    words = [w for p in items.values() for r in p if r["t"] == "text" for w in r["w"]]
    print(f"pages {len(items)} | all full (15 lines, 8 on pages 1-2) | headers {kinds.count('header')} | basmalas {kinds.count('basmala')} | words {len(words)}")
    print({k: len(v) for k, v in units.items()}, "units; first/last of 1/8:", units["1/8"][0]["from"], units["1/8"][-1]["to"])
    print("OK")
