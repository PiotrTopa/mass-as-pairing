#!/usr/bin/env python3
"""results/K6_alpha.csv: corner-block loop readouts of the 39 stored configurations (claim K6.2).

One row per configuration: archive file, L, y, kappa, phase label, scored / near-critical flags, f_M, alpha, N_odd, the
reading of the pre-registered rule, quantised, det half turns. Exact per configuration (no errors).
"""

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import RESULTS, derived

COLS = ["file", "L", "y", "kappa", "phase", "scored", "near_critical", "f_M", "alpha", "N_odd", "reading", "quantised"]
COLS += ["det_half_turns"]


def main():
    recs = json.loads(derived("n1inst", "K6_winding_stored.json").read_text())["records"]
    recs.sort(key=lambda r: (r["L"], r["kappa"], r["y"], r["file"]))
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "K6_alpha.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in recs:
            w.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in COLS])
    print(dst)


if __name__ == "__main__":
    main()
