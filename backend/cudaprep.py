"""cudaprep.py - hardware helpers that need no torch (ASCII only).

prep()          import torch if it is installed (dev machine: it provides the cublas dll and must come
                before ctranslate2 / faster_whisper), then register the DLL folders of the
                nvidia-*-cu12 wheels (packaged install, no torch).
gpu_available() True if CTranslate2 sees a CUDA GPU. Env HW_DEVICE=cpu forces False (test knob).
gpu_compute()   best GPU compute type this card supports (env GPU_COMPUTE overrides).
cpu_threads()   env CPU_THREADS, else the logical core count (previous behaviour)."""
import glob
import os
import sys

_done = False


def prep():
    global _done
    if _done:
        return
    _done = True
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    if os.name != "nt":
        return
    for p in list(sys.path):
        for d in glob.glob(os.path.join(p, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def gpu_available():
    if os.getenv("HW_DEVICE", "").lower() == "cpu":
        return False
    prep()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def gpu_compute():
    forced = os.getenv("GPU_COMPUTE")
    if forced:
        return forced
    prep()
    try:
        import ctranslate2
        have = set(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return "float16"
    for c in ("float16", "int8_float32", "float32"):
        if c in have:
            return c
    return "float16"


def cpu_threads():
    try:
        n = int(os.getenv("CPU_THREADS", "0"))
    except ValueError:
        n = 0
    return n if n > 0 else (os.cpu_count() or 4)
