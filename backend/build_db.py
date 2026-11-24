"""Builds backend/quran.db: all 114 surahs, 6,236 ayahs, 77,429 words, 604 Madani-mushaf pages with line positions.

  python build_db.py                 download the 604 page files (cached in quran_source/) and build
  python build_db.py --source DIR    build from page-001.json ... page-604.json already on disk

Data: github.com/zonetecde/mushaf-layout (Madani Mushaf, Hafs 'an 'Asim). Check that repository's licence and the
Quran-text/font licences before redistributing the app or the database to others.
"""
import json, os, re, sqlite3, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from alignment import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "quran.db")
SRC_DIR = os.path.join(HERE, "quran_source")
URL = "https://raw.githubusercontent.com/zonetecde/mushaf-layout/main/mushaf/page-{:03d}.json"
EXPECTED = {"surahs": 114, "ayahs": 6236, "words": 77429}
SCHEMA_VERSION = 2
AYAH_NUM = re.compile(r"\s*[0-9٠-٩۰-۹]+\s*$")      # the dataset glues the ayah number to the last word of each ayah

NAMES_AR = ("الفاتحة|البقرة|آل عمران|النساء|المائدة|الأنعام|الأعراف|الأنفال|التوبة|يونس|هود|يوسف|الرعد|إبراهيم|الحجر|النحل|"
 "الإسراء|الكهف|مريم|طه|الأنبياء|الحج|المؤمنون|النور|الفرقان|الشعراء|النمل|القصص|العنكبوت|الروم|لقمان|السجدة|الأحزاب|سبأ|"
 "فاطر|يس|الصافات|ص|الزمر|غافر|فصلت|الشورى|الزخرف|الدخان|الجاثية|الأحقاف|محمد|الفتح|الحجرات|ق|الذاريات|الطور|النجم|القمر|"
 "الرحمن|الواقعة|الحديد|المجادلة|الحشر|الممتحنة|الصف|الجمعة|المنافقون|التغابن|الطلاق|التحريم|الملك|القلم|الحاقة|المعارج|نوح|"
 "الجن|المزمل|المدثر|القيامة|الإنسان|المرسلات|النبأ|النازعات|عبس|التكوير|الانفطار|المطففين|الانشقاق|البروج|الطارق|الأعلى|"
 "الغاشية|الفجر|البلد|الشمس|الليل|الضحى|الشرح|التين|العلق|القدر|البينة|الزلزلة|العاديات|القارعة|التكاثر|العصر|الهمزة|الفيل|"
 "قريش|الماعون|الكوثر|الكافرون|النصر|المسد|الإخلاص|الفلق|الناس").split("|")
NAMES_EN = ("Al-Fatihah|Al-Baqarah|Ali 'Imran|An-Nisa|Al-Ma'idah|Al-An'am|Al-A'raf|Al-Anfal|At-Tawbah|Yunus|Hud|Yusuf|Ar-Ra'd|"
 "Ibrahim|Al-Hijr|An-Nahl|Al-Isra|Al-Kahf|Maryam|Ta-Ha|Al-Anbiya|Al-Hajj|Al-Mu'minun|An-Nur|Al-Furqan|Ash-Shu'ara|An-Naml|"
 "Al-Qasas|Al-'Ankabut|Ar-Rum|Luqman|As-Sajdah|Al-Ahzab|Saba|Fatir|Ya-Sin|As-Saffat|Sad|Az-Zumar|Ghafir|Fussilat|Ash-Shura|"
 "Az-Zukhruf|Ad-Dukhan|Al-Jathiyah|Al-Ahqaf|Muhammad|Al-Fath|Al-Hujurat|Qaf|Adh-Dhariyat|At-Tur|An-Najm|Al-Qamar|Ar-Rahman|"
 "Al-Waqi'ah|Al-Hadid|Al-Mujadilah|Al-Hashr|Al-Mumtahanah|As-Saff|Al-Jumu'ah|Al-Munafiqun|At-Taghabun|At-Talaq|At-Tahrim|"
 "Al-Mulk|Al-Qalam|Al-Haqqah|Al-Ma'arij|Nuh|Al-Jinn|Al-Muzzammil|Al-Muddaththir|Al-Qiyamah|Al-Insan|Al-Mursalat|An-Naba|"
 "An-Nazi'at|'Abasa|At-Takwir|Al-Infitar|Al-Mutaffifin|Al-Inshiqaq|Al-Buruj|At-Tariq|Al-A'la|Al-Ghashiyah|Al-Fajr|Al-Balad|"
 "Ash-Shams|Al-Layl|Ad-Duha|Ash-Sharh|At-Tin|Al-'Alaq|Al-Qadr|Al-Bayyinah|Az-Zalzalah|Al-'Adiyat|Al-Qari'ah|At-Takathur|"
 "Al-'Asr|Al-Humazah|Al-Fil|Quraysh|Al-Ma'un|Al-Kawthar|Al-Kafirun|An-Nasr|Al-Masad|Al-Ikhlas|Al-Falaq|An-Nas").split("|")
assert len(NAMES_AR) == len(NAMES_EN) == 114, (len(NAMES_AR), len(NAMES_EN))

def fetch_page(n, dest):
    path = os.path.join(dest, f"page-{n:03d}.json")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        return path
    err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(n), headers={"User-Agent": "quran-tracker"})
            data = urllib.request.urlopen(req, timeout=40).read()
            json.loads(data)                                  # must be valid JSON
            with open(path, "wb") as f: f.write(data)
            return path
        except Exception as e:
            err = e; time.sleep(1 + attempt)
    raise RuntimeError(f"could not download page {n}: {err}")

def download_all(dest):
    os.makedirs(dest, exist_ok=True)
    print("Downloading 604 page files (cached in quran_source/, ~15 MB)…")
    with ThreadPoolExecutor(12) as ex:
        for k, _ in enumerate(ex.map(lambda n: fetch_page(n, dest), range(1, 605)), 1):
            if k % 100 == 0: print(f"  {k}/604")

def parse(src):
    words, layout = {}, []
    for n in range(1, 605):
        d = json.load(open(os.path.join(src, f"page-{n:03d}.json"), encoding="utf-8"))
        head = None
        for l in d["lines"]:
            if l["type"] == "surah-header":
                head = int(l["surah"]); layout.append((n, l["line"], "header", head))
            elif l["type"] == "basmala":
                layout.append((n, l["line"], "basmala", head))
            else:
                for w in l.get("words", []):
                    s, a, i = map(int, w["location"].split(":"))
                    text = AYAH_NUM.sub("", w["word"]).strip()   # keep waqf marks, drop the ayah number
                    clean = normalize(text)
                    if not clean:
                        raise ValueError(f"word {w['location']} has no letters after cleaning: {w['word']!r}")
                    words[(s, a, i)] = (text, clean, n, l["line"])
    return words, layout

def validate(words):
    ayahs = {(s, a) for s, a, _ in words}
    got = {"surahs": len({s for s, _ in ayahs}), "ayahs": len(ayahs), "words": len(words)}
    if got != EXPECTED:
        raise ValueError(f"incomplete data: got {got}, expected {EXPECTED}. Delete quran_source/ and run again.")

def build(words, layout, path=DEFAULT_DB):
    tmp = path + ".tmp"
    if os.path.exists(tmp): os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.executescript("""
      CREATE TABLE words(surah INT, ayah INT, pos INT, text_uthmani TEXT, text_clean TEXT,
                         start_ms INT, end_ms INT, page INT, line INT);
      CREATE TABLE surahs(surah INT PRIMARY KEY, name_ar TEXT, name_en TEXT, ayahs INT, page INT);
      CREATE TABLE layout(page INT, line INT, type TEXT, surah INT);
      CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);""")
    keys = sorted(words)
    con.executemany("INSERT INTO words VALUES(?,?,?,?,?,NULL,NULL,?,?)",
                    [(s, a, i, *words[(s, a, i)][:2], *words[(s, a, i)][2:]) for s, a, i in keys])
    info = {}
    for s, a, i in keys:
        r = info.setdefault(s, {"ayahs": 0, "page": words[(s, a, i)][2]})
        r["ayahs"] = max(r["ayahs"], a)
    con.executemany("INSERT INTO surahs VALUES(?,?,?,?,?)",
                    [(s, NAMES_AR[s - 1], NAMES_EN[s - 1], r["ayahs"], r["page"]) for s, r in sorted(info.items())])
    con.executemany("INSERT INTO layout VALUES(?,?,?,?)", layout)
    con.executemany("INSERT INTO meta VALUES(?,?)", [
        ("schema_version", str(SCHEMA_VERSION)), ("source", "zonetecde/mushaf-layout"),
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("counts", json.dumps(EXPECTED))])
    con.execute("CREATE INDEX idx_words ON words(surah, ayah, pos)")
    con.execute("CREATE INDEX idx_page ON words(page, line)")
    con.commit(); con.close()
    os.replace(tmp, path)                                     # atomic: never leaves a half-built database

if __name__ == "__main__":
    src = sys.argv[sys.argv.index("--source") + 1] if "--source" in sys.argv else SRC_DIR
    if "--source" not in sys.argv: download_all(src)
    words, layout = parse(src); validate(words); build(words, layout)
    print(f"quran.db built: {EXPECTED['surahs']} surahs, {EXPECTED['ayahs']} ayahs, {EXPECTED['words']} words, 604 pages.")
