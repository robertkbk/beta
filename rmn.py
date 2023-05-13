from pathlib import Path
from os import listdir

d = Path("dataset")
n = Path("dataset-s")

n.mkdir(exist_ok=True)

for f in listdir(d):
    with open(d / f, "r") as r:
        ls = [ls for l in r.readlines() if (ls := l.strip())]

        if len(ls) < 3:
            print(f)
            continue

        with open(n / f"S-{f}", "w") as w:
            w.writelines(f"{l}\n" for l in ls)
