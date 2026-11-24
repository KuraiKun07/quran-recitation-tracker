"""fetch_offline.py - download once what the offline packages carry, into backend\\kit\\offline_cache\\ (ASCII only).

Run on your PC (internet needed once), with Python 3.12 64-bit (the same version the package installs):
    python kit\\fetch_offline.py
Result:  offline_cache\\python-3.12.10-embed-amd64.zip   private Python for the package
         offline_cache\\vc_redist.x64.exe                Microsoft Visual C++ runtime (installed only if missing)
         offline_cache\\wheels\\                          app components (requirements.lock.txt + pip)
         offline_cache\\wheels_cuda\\                     NVIDIA GPU components (requirements-cuda.txt), GPU edition only
Already downloaded files are kept; delete offline_cache to start over. make_package.py --edition cpu|gpu uses it."""
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "offline_cache")
PY_VER = "3.12.10"
FILES = {
    "python-%s-embed-amd64.zip" % PY_VER: "https://www.python.org/ftp/python/%s/python-%s-embed-amd64.zip" % (PY_VER, PY_VER),
    "vc_redist.x64.exe": "https://aka.ms/vs/17/release/vc_redist.x64.exe",
}
PIP_TARGET = ["--only-binary=:all:", "--platform", "win_amd64", "--python-version", "3.12", "--implementation", "cp"]


def pip_download(args, dest):
    os.makedirs(dest, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "-d", dest] + PIP_TARGET + args
    print("pip download", " ".join(args), "->", os.path.relpath(dest, HERE))
    if subprocess.call(cmd) != 0:
        raise SystemExit("pip download failed: " + " ".join(args))


def main():
    if sys.version_info[:2] != (3, 12) or sys.maxsize < 2 ** 32:
        print("note: running on Python %d.%d; wheels are still selected for Windows x64 / CPython 3.12" % sys.version_info[:2])
    os.makedirs(CACHE, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(CACHE, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print("have", name)
            continue
        print("download", name)
        urllib.request.urlretrieve(url, dest + ".part")
        os.replace(dest + ".part", dest)
    pip_download(["-r", os.path.join(HERE, "requirements.lock.txt")], os.path.join(CACHE, "wheels"))
    pip_download(["pip"], os.path.join(CACHE, "wheels"))
    pip_download(["--no-deps", "-r", os.path.join(HERE, "requirements-cuda.txt")], os.path.join(CACHE, "wheels_cuda"))
    for d in ("wheels", "wheels_cuda"):
        p = os.path.join(CACHE, d)
        n = [f for f in os.listdir(p) if f.endswith(".whl")]
        print("%-12s %3d wheels, %5.0f MB" % (d, len(n), sum(os.path.getsize(os.path.join(p, f)) for f in n) / 1e6))
    print("done:", CACHE)


if __name__ == "__main__":
    main()
