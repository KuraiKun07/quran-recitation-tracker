# ct2conv.py - convert a HF Whisper checkpoint to CTranslate2 fp16. Run from backend.
#   python ct2conv.py tarteel-ai/whisper-base-ar-quran whisper-ct2-parity
#   python ct2conv.py <hf_dir> <out_dir>
# Used first as a PARITY CHECK: convert the unmodified checkpoint, then compare with the
# production folder (whisper-quran-ct2-fp16) and run baseline.py on it.
import filecmp, os, sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # use the local HF cache, no network
import torch  # noqa: F401  (loads torch's CUDA DLLs; harmless here)

HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.join(HERE, "whisper-quran-ct2-fp16")


def convert(src, out, quantization="float16"):
    from ctranslate2.converters import TransformersConverter
    TransformersConverter(src).convert(out, quantization=quantization, force=True)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    if not os.path.isabs(out):
        out = os.path.join(HERE, out)
    convert(src, out)
    print("converted ->", out)
    for n in sorted(os.listdir(out)):
        print("  %-18s %12d bytes" % (n, os.path.getsize(os.path.join(out, n))))
    if os.path.isdir(PROD) and os.path.abspath(out) != os.path.abspath(PROD):
        print("\nvs production folder:")
        for n in sorted(os.listdir(PROD)):
            a, b = os.path.join(PROD, n), os.path.join(out, n)
            if not os.path.exists(b):
                print("  %-18s MISSING in new folder" % n)
            else:
                print("  %-18s size %d vs %d | identical bytes: %s"
                      % (n, os.path.getsize(a), os.path.getsize(b), filecmp.cmp(a, b, shallow=False)))


if __name__ == "__main__":
    main()
