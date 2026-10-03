#!/usr/bin/env python3
"""results/K7_discrimination.csv: expected Delta chi^2 of walking vs power law for xi(y) (claim K7.3)."""

import csv
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.analysis.fss import discrimination_table  # noqa: E402
from masspairing.data import RESULTS  # noqa: E402


def main():
    rows = discrimination_table()
    for r in rows:
        r["y_c"] = "free" if r.pop("free_shift") else "fixed"
        r["R"] = f"{r['R']:.4g}"
        for k in ("dchi2_WP", "dchi2_PW"):
            r[k] = "" if math.isnan(r[k]) else f"{r[k]:.4g}"
    RESULTS.mkdir(exist_ok=True)
    cols = ["grid", "Lmax", "eps", "y_c", "n_use", "R", "dchi2_WP", "dchi2_PW"]
    with open(RESULTS / "K7_discrimination.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"results/K7_discrimination.csv: {len(rows)} rows (dchi2_WP: truth walking, fit power law; empty = no fit)")


if __name__ == "__main__":
    main()
