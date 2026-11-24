# error_words.py - run from backend AFTER baseline.py (uses model_outputs.jsonl).
# Shows WHAT the baseline errors are: substitutions / deletions / insertions, near-misses (one
# letter off) vs completely different words, and which reference words fail most.
# Console is ASCII only; the Arabic list goes to error_words_report.txt (UTF-8, VS Code).
import collections, json, os

from trainutil import HERE, edit_distance, load_manifest, norm


def ops(ref, hyp):
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                out.append(("S", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            out.append(("I", "", hyp[j - 1]))
            j -= 1
    return out[::-1]


def main():
    rows = {r["file"]: r for r in load_manifest()}
    outs = [json.loads(l) for l in open(os.path.join(HERE, "model_outputs.jsonl"), encoding="utf-8") if l.strip()]
    cnt = collections.Counter()
    near = 0
    by_word = collections.Counter()
    by_kind = collections.Counter()
    bad_clips = 0
    total_words = 0
    lines = []
    for o in outs:
        r = rows.get(o["file"])
        if not r:
            continue
        ref, hyp = norm(r["clean"]).split(), norm(o["out"]).split()
        total_words += len(ref)
        ev = ops(ref, hyp)
        if ev:
            bad_clips += 1
        for k, a, b in ev:
            cnt[k] += 1
            by_kind[r["kind"]] += 1
            if k == "S" and edit_distance(a, b) <= 1:
                near += 1
            if a:
                by_word[a] += 1
            lines.append("%d:%d [%s] %s  ref=%s  heard=%s" % (r["surah"], r["ayah"], r["kind"], k, a, b))
    tot = sum(cnt.values())
    print("words: %d | errors: %d (substitutions %d, deletions %d, insertions %d)"
          % (total_words, tot, cnt["S"], cnt["D"], cnt["I"]))
    print("clips with at least one error: %d of %d" % (bad_clips, len(outs)))
    print("substitutions that are only ONE letter off: %d of %d" % (near, cnt["S"]))
    print("errors by clip kind:", dict(by_kind))
    print("reference words that failed more than once: %d distinct" % sum(1 for w, n in by_word.items() if n > 1))
    with open(os.path.join(HERE, "error_words_report.txt"), "w", encoding="utf-8") as f:
        f.write("TOP FAILING REFERENCE WORDS\n")
        for w, n in by_word.most_common(25):
            f.write("%3d  %s\n" % (n, w))
        f.write("\nALL ERRORS\n" + "\n".join(lines) + "\n")
    print("\nArabic details: backend\\error_words_report.txt")


if __name__ == "__main__":
    main()
