"""make_package.py - build the shareable zips. ASCII only, read-only on your project (writes only to --out).

Run on your PC, from backend:
    python kit\\fetch_offline.py                       (once: downloads Python, VC++ runtime and wheels to kit\\offline_cache)
    python kit\\make_package.py --model whisper-quran-ct2-muq --edition cpu gpu
Editions:  cpu     offline, processor only, works on any Windows 10/11 x64 PC without internet
           gpu     offline, also carries the NVIDIA GPU components (about 1.3 GB more)
           online  small; downloads Python and the components from the internet on first start
Result: dist\\<edition>\\QuranTracker\\  and  dist\\QuranTracker-<version>-<edition>.zip  (CRLF for .bat/.ps1/.txt files).
The version comes from APP_VERSION in server.py. Only a whitelist is copied: no backups, tools, logs or caches."""
import argparse
import datetime
import hashlib
import os
import re
import shutil
import sys
import zipfile

BACKEND_FILES = ["server.py", "alignment.py", "ayah_finder.py", "basmala.py", "muqattaat.py",
                 "mushaf.py", "cudaprep.py", "quran.db", "divisions.json", "phonecheck.py",
                 "third_party/__init__.py", "third_party/imam.py", "third_party/LICENSE-tajweed.txt"]
# pronunciation check models (phone_convert.py -> backend\phone-model): the gpu edition gets Muaalem + the CPU model
PHONE_CPU = ["tajweed-v2-fp32.onnx", "tajweed-v2.json"]
PHONE_GPU = PHONE_CPU + ["muaalem-phonemes-fp16.onnx", "phone_config.json", "phone_fbank.npz"]
MODEL_NEEDED = ["config.json", "model.bin", "vocabulary.json", "tokenizer.json"]
KIT = {"Start.bat": "", "READ_ME_FIRST.txt": "", "run.ps1": "setup", "lib.ps1": "setup",
       "requirements.lock.txt": "setup", "requirements-cuda.txt": "setup", "requirements-ortgpu.txt": "setup",
       "THIRD_PARTY_NOTICES.txt": ""}
TEXT_EXT = (".bat", ".ps1", ".txt")
STORED_EXT = (".whl", ".zip", ".exe", ".onnx")   # already compressed (or nearly: model weights)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if dst.lower().endswith(TEXT_EXT):
        data = open(src, "rb").read()
        try:
            data.decode("ascii")
        except UnicodeDecodeError:
            raise SystemExit("not pure ASCII (breaks Windows PowerShell 5 / cmd): " + src)
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        open(dst, "wb").write(data)
    else:
        shutil.copyfile(src, dst)


def offline_files(cache, edition, problems):
    """published path -> source for setup\\offline (empty for the online edition)."""
    out = {}
    if edition == "online":
        return out
    if not os.path.isdir(cache):
        problems.append("offline cache not found: %s (run kit\\fetch_offline.py first)" % cache)
        return out
    for f in os.listdir(cache):
        if re.match(r"python-3\.12\.\d+-embed-amd64\.zip$", f) or f == "vc_redist.x64.exe":
            out[os.path.join("setup", "offline", f)] = os.path.join(cache, f)
    dirs = ["wheels"] + (["wheels_cuda"] if edition == "gpu" else [])
    for d in dirs:
        p = os.path.join(cache, d)
        whl = [f for f in os.listdir(p) if f.endswith(".whl")] if os.path.isdir(p) else []
        if not whl:
            problems.append("no wheels in %s (run kit\\fetch_offline.py)" % p)
        for f in whl:
            out[os.path.join("setup", "offline", "wheels", f)] = os.path.join(p, f)
    if not any("embed-amd64" in k for k in out):
        problems.append("python embed zip missing in " + cache)
    if not any(k.endswith("vc_redist.x64.exe") for k in out):
        problems.append("vc_redist.x64.exe missing in " + cache)
    if not any(os.path.basename(k).startswith("pip-") for k in out):
        problems.append("pip wheel missing in %s\\wheels" % cache)
    return out


def build(a, edition, backend, web, version):
    problems = []
    srcs = {}
    for f in BACKEND_FILES:
        p = os.path.join(backend, f)
        (srcs.__setitem__(os.path.join("app", "backend", f), p) if os.path.isfile(p) else problems.append("missing backend file: " + p))
    idx = os.path.join(web, "index.html")
    (srcs.__setitem__(os.path.join("app", "client-web", "index.html"), idx) if os.path.isfile(idx) else problems.append("missing: " + idx))
    model = os.path.abspath(a.model) if a.model else None
    if not model or not os.path.isdir(model):
        problems.append("model folder not found (use --model DIR or set CT2_DIR): %s" % a.model)
    else:
        for f in MODEL_NEEDED:
            p = os.path.join(model, f)
            (srcs.__setitem__(os.path.join("app", "backend", "model", f), p) if os.path.isfile(p) else problems.append("model file missing: " + p))
    phone = os.path.abspath(a.phone or os.path.join(backend, "phone-model"))
    for f in (PHONE_GPU if edition == "gpu" else PHONE_CPU):
        p = os.path.join(phone, f)
        (srcs.__setitem__(os.path.join("app", "backend", "phone-model", f), p) if os.path.isfile(p)
         else problems.append("phone model file missing (run phone_convert.py): " + p))
    for name, sub in KIT.items():
        p = os.path.join(a.kit, name)
        (srcs.__setitem__(os.path.join(sub, name) if sub else name, p) if os.path.isfile(p) else problems.append("kit file not found: " + p))
    srcs.update(offline_files(os.path.join(a.kit, "offline_cache"), edition, problems))
    for f in BACKEND_FILES:
        if f.endswith(".py") and os.path.isfile(os.path.join(backend, f)):
            try:
                compile(open(os.path.join(backend, f), "rb").read(), f, "exec")
            except SyntaxError as e:
                problems.append("syntax error in %s: %s" % (f, e))
    if problems:
        print("NOT BUILT (%s). Problems:" % edition)
        for p in problems:
            print("  -", p)
        return 1

    out = os.path.abspath(a.out)
    pkg = os.path.join(out, edition, "QuranTracker")
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    for rel, src in srcs.items():
        copy_file(src, os.path.join(pkg, rel))
    info = ["Quran Tracker package", "version: " + version, "edition: " + edition,
            "built: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "server.py sha256: " + sha256(os.path.join(backend, "server.py")),
            "alignment.py sha256: " + sha256(os.path.join(backend, "alignment.py")),
            "model.bin sha256: " + sha256(os.path.join(model, "model.bin"))]
    open(os.path.join(pkg, "app", "VERSION.txt"), "wb").write(("\r\n".join(info) + "\r\n").encode("ascii"))

    zpath = os.path.join(out, "QuranTracker-%s-%s.zip" % (version, edition))
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for d, _, files in os.walk(pkg):
            for f in sorted(files):
                full = os.path.join(d, f)
                arc = os.path.join("QuranTracker", os.path.relpath(full, pkg)).replace(os.sep, "/")
                z.write(full, arc, compress_type=zipfile.ZIP_STORED if f.lower().endswith(STORED_EXT) else zipfile.ZIP_DEFLATED)
    n = sum(len(fs) for _, _, fs in os.walk(pkg))
    print("built %-6s %3d files -> %s (%.0f MB)" % (edition, n, zpath, os.path.getsize(zpath) / 1e6))
    return 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="backend folder (default: the folder above kit)")
    ap.add_argument("--web", default=None, help="client-web folder (default: <backend>\\..\\client-web)")
    ap.add_argument("--model", default=os.environ.get("CT2_DIR"), help="CT2 model folder (default: env CT2_DIR)")
    ap.add_argument("--kit", default=here, help="folder with the kit files (default: this script's folder)")
    ap.add_argument("--phone", default=None, help="phone model folder (default: <backend>/phone-model)")
    ap.add_argument("--edition", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu", "online"])
    ap.add_argument("--out", default="dist")
    ap.add_argument("--version", default=None, help="default: APP_VERSION in server.py")
    a = ap.parse_args()
    backend = os.path.abspath(a.backend or os.path.dirname(here))
    web = os.path.abspath(a.web or os.path.join(backend, "..", "client-web"))
    m = re.search(r'^APP_VERSION = "([0-9][0-9A-Za-z.\-]*)"', open(os.path.join(backend, "server.py"), encoding="utf-8").read(), re.M)
    if not m and not a.version:
        sys.exit("APP_VERSION not found in server.py (or pass --version)")
    version = a.version or m.group(1)
    rc = 0
    for ed in a.edition:
        rc |= build(a, ed, backend, web, version)
    sys.exit(rc)


if __name__ == "__main__":
    main()
