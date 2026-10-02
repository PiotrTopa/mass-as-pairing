#!/usr/bin/env python3
"""results/K4_takeover.csv: elementary and composite response to the nodal source on flavour 4 (claim K4.1).

One row per chain: L, y, h, run, measurements after the cut, mean Pfaffian sign, phi_T/h, phi_heavy/h, phi_light/h with
their errors, and |phi_T|/phi_heavy.
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.analysis.n1inst import takeover_row
from masspairing.data import RESULTS, derived, load_chain

COLS = ["L", "y", "h", "run", "n", "sign", "phiT_h", "ephiT_h", "heavy_h", "eheavy_h", "light_h", "elight_h", "ratio"]


def main():
    rows = []
    for f in sorted(derived("n1inst", "takeover").glob("*.npz")):
        r = takeover_row(*load_chain(f))
        r["run"] = f.stem.rsplit("_", 1)[1]
        rows.append(r)
    rows.sort(key=lambda r: (r["y"], r["L"], r["h"], r["run"]))
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "K4_takeover.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], (int, str)) else f"{r[k]:.6g}" for k in COLS])
    print(dst)


if __name__ == "__main__":
    main()
