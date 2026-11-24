# eval_snapshot.py - run from backend.   python eval_snapshot.py NAME
# Runs aligner_eval.py and mistake_eval.py on every model_outputs*_ep4/base file and stores console output and
# report files in replay_out\snap_NAME\. Compare two snapshots with:  python eval_snapshot.py --compare A B
# Purpose: prove a change is neutral (identical) or list exactly what moved. ASCII console only.
import filecmp, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "replay_out")
FILES = ["model_outputs.jsonl", "model_outputs_new_ep4.jsonl", "model_outputs_own3_ep4.jsonl",
         "model_outputs_voice2_ep4.jsonl"]


def run(cmd, log):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", env=env)
    with open(log, "w", encoding="utf-8") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        print("FAILED:", " ".join(cmd), "(see", os.path.basename(log) + ")")
        sys.exit(1)


def snapshot(name):
    d = os.path.join(OUT, "snap_" + name)
    os.makedirs(d, exist_ok=True)
    for fn in FILES:
        tag = fn[:-6]
        run(["aligner_eval.py", "--outputs", fn], os.path.join(d, "ae_%s.console.txt" % tag))
        shutil.copy(os.path.join(HERE, "aligner_eval_report.txt"), os.path.join(d, "ae_%s.report.txt" % tag))
        run(["mistake_eval.py", "--outputs", fn], os.path.join(d, "me_%s.console.txt" % tag))
        shutil.copy(os.path.join(HERE, "mistake_eval_report.txt"), os.path.join(d, "me_%s.report.txt" % tag))
        print("done:", tag)
    print("snapshot stored in replay_out\\snap_%s" % name)


def compare(a, b):
    da, db = os.path.join(OUT, "snap_" + a), os.path.join(OUT, "snap_" + b)
    names = sorted(set(os.listdir(da)) | set(os.listdir(db)))
    diff = [n for n in names if not (os.path.exists(os.path.join(da, n)) and os.path.exists(os.path.join(db, n))
                                     and filecmp.cmp(os.path.join(da, n), os.path.join(db, n), shallow=False))]
    print("%d files compared, %d differ" % (len(names), len(diff)))
    for n in diff:
        print("  DIFFERS:", n)
    print("RESULT:", "IDENTICAL" if not diff else "CHANGED (inspect with: python mydiff.py old new out.txt)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        snapshot(sys.argv[1])
    else:
        print(__doc__ if __doc__ else "usage: python eval_snapshot.py NAME | --compare A B")
