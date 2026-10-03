#!/usr/bin/env python3
"""results/K7_fss.csv: finite-size-scaling table of claims K7.1 and K7.2.

Rows (column ``table``):
  grid      every (kappa, L, y) of the 44 chains (first 100 trajectories cut, replicas combined), K7.1 / K7.2;
  pooled    the pooled P_c replicas at L = 6 and 8 (K7.2).
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.analysis import t3a  # noqa: E402
from masspairing.data import RESULTS  # noqa: E402

COLS = ["table", "kappa", "L", "y", "n", "tau_int_m2", "replicas", "scored", "flags"]
VALS = ["mabs", "R4", "xiL", "Spi", "sig2", "O4", "Ve_deficit"]


def fmt(x, d=6):
    return f"{x:.{d}g}"


def main():
    pilot, new = t3a.load_records("pilot"), t3a.load_records("sharpened")
    P = t3a.analyse_pilot(pilot)
    S = t3a.analyse_sharpened(pilot, new, P)

    def row(table, s):
        out = dict(table=table, kappa=s["kappa"], L=s["L"], y=s["y"], n=s["n"], tau_int_m2=fmt(s["tau"], 4))
        out |= dict(replicas=s.get("replicas", 1), scored=s["scored"], flags=",".join(s["flags"]))
        for k in VALS[:-1]:
            out[k], out["e_" + k] = fmt(s[k]), fmt(s["e_" + k], 2)
        out["Ve_deficit"], out["e_Ve_deficit"] = fmt(2 / 3 - s["Ve"]), fmt(s["e_Ve"], 2)
        return out

    rows = [row("grid", s) for _, s in sorted(S["comb"].items())]
    rows += [row("pooled", S["verdict"]["pooled"][L]) for L in ("L6", "L8")]
    cols = COLS + [c for k in VALS for c in (k, "e_" + k)]
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "K7_fss.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"results/K7_fss.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
