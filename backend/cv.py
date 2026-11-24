# cv.py - 5-fold cross-validation over ALL recorded clips. Run from backend.
#   python cv.py --target B --lr 3e-6 --epochs 3
#   python cv.py --target B --lr 3e-6 --epochs 3 --freeze-encoder
#   python cv.py --target B --lr 3e-6 --epochs 3 --summary-only     # just re-print the table
# Each fold holds out ~1/5 of the AYAHS (never trained on), so the totals cover every clip
# (about 582 words) instead of the 82-word fixed split. Finished folds are skipped on re-run.
import argparse, json, os, subprocess, sys

from trainutil import HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="B")
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    base = args.tag or ("cv_" + args.target + ("_fe" if args.freeze_encoder else "") + "_lr%g" % args.lr)

    def rpath(i):
        return os.path.join(HERE, "runs", "%s_f%d" % (base, i), "results.json")

    def done(i):
        try:
            return len(json.load(open(rpath(i)))) == args.epochs + 1
        except Exception:
            return False

    if not args.summary_only:
        for i in range(args.folds):
            if done(i):
                print("fold %d: already done" % i)
                continue
            cmd = [sys.executable, "train.py", "--target", args.target, "--lr", str(args.lr),
                   "--epochs", str(args.epochs), "--folds", str(args.folds), "--fold", str(i), "--tag", base]
            if args.freeze_encoder:
                cmd.append("--freeze-encoder")
            print("--- fold %d/%d" % (i + 1, args.folds))
            p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            tail = []
            for line in p.stdout:
                tail = (tail + [line.rstrip()])[-15:]
                if line.startswith(("epoch ", "clips:")):
                    print("  " + line.rstrip())
            if p.wait() != 0:
                print("fold %d FAILED; last output:" % i)
                print("\n".join(tail))
                sys.exit(1)

    res = {}
    for i in range(args.folds):
        if done(i):
            res[i] = json.load(open(rpath(i)))
    if not res:
        print("no finished folds for", base)
        return
    print("\n=== %s | folds finished: %d of %d" % (base, len(res), args.folds))
    E = args.epochs
    for e in range(E + 1):
        b4 = sum(res[i][e]["beam4_err"] for i in res)
        gr = sum(res[i][e]["greedy_err"] for i in res)
        w = sum(res[i][e]["words"] for i in res)
        line = "epoch %d: beam4 %3d/%d (%.1f%%) | greedy %3d/%d (%.1f%%)" % (e, b4, w, 100 * b4 / w, gr, w, 100 * gr / w)
        if e > 0:
            d = [(res[i][e]["beam4_err"] + res[i][e]["greedy_err"]) - (res[i][0]["beam4_err"] + res[i][0]["greedy_err"])
                 for i in res]
            line += " | vs epoch 0, folds better/same/worse: %d/%d/%d" % (
                sum(x < 0 for x in d), sum(x == 0 for x in d), sum(x > 0 for x in d))
        else:
            line += "   <- production model"
        print(line)


if __name__ == "__main__":
    main()
