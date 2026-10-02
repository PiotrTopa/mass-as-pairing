#!/usr/bin/env python3
"""results/K3_dimer.csv: the bond-crystal (columnar dimer) and eps-channel susceptibilities V<D_par^2>, V<D_perp^2>,
V<|phi_stag|^2> and the link-field dimer length xi/L per point and volume: the +edge points of K3.6 and the six
U(1)_eps-exact points of K3.5 (errors: blocked jackknife of each series)."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from masspairing.analysis import k3
from masspairing.data import RESULTS


def main():
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "K3_dimer.csv"
    cols = [
        "claim",
        "y",
        "g10",
        "g6",
        "L",
        "V_D_par",
        "err",
        "V_D_perp",
        "err",
        "V_phi_stag",
        "err",
        "xi_dimer_over_L",
        "err",
    ]
    sets = [
        ("K3.6", (y, g, g), 1, lambda L, y=y, g=g: [(k3.c073_rel(L, y, g), 0)])
        for y, g in ((2.41, 0.05), (3.0, 0.1), (2.41, 0.1))
    ]
    sets += [("K3.5", p, 0, lambda L, p=p: k3.x_spec(p, L)) for p in k3.X_POINTS]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for claim, p, pair, spec in sets:
            for L in (4, 6, 8):
                r = k3.ensemble_of(spec(L))
                row = [claim, *p, L]
                for k in ("dimer_sq_par", "dimer_sq_perp", "phi_stag_sq"):
                    row += [f"{r['V'] * r[k]['mean']:.5g}", f"{r['V'] * r[k]['err']:.2g}"]
                x, e = k3.value(r, f"xi2_s_dimer_{pair}L")
                w.writerow(row + [f"{x:.4f}", f"{e:.4f}"])
    print(path)


if __name__ == "__main__":
    main()
