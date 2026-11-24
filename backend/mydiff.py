import sys, difflib, io
a, b, out = sys.argv[1:4]
A = io.open(a, encoding="utf-8-sig", newline="").read().splitlines()
B = io.open(b, encoding="utf-8-sig", newline="").read().splitlines()
d = list(difflib.unified_diff(A, B, a, b, n=3, lineterm=""))
io.open(out, "w", encoding="utf-8").write("\n".join(d) + "\n")
print(len(d), "diff lines;", sum(1 for x in d if x.startswith("+") and not x.startswith("+++")), "added,", sum(1 for x in d if x.startswith("-") and not x.startswith("---")), "removed")
