#!/usr/bin/env python3
"""results/K3_chi10_scaling.csv: the complete chi10 at p = 0 (and its largest corner), xi10/L and the ratio to the free
theory per point and volume -- K3.1 (five link-field points), K3.2 (epsilon model at P_c), K3.5 (U(1)_eps-exact
family) -- with the finite-size exponents on (4,6) and (6,8) on the L = 6 and L = 8 rows."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from masspairing.analysis import k3
from masspairing.data import RESULTS

COLS = [
    "claim",
    "model",
    "y",
    "g10",
    "g6",
    "L",
    "chi10",
    "chi10_err",
    "corner_max",
    "chi10_over_free",
    "xi10_over_L",
    "xi10_over_L_err",
    "exponent",
    "exponent_err",
    "ntraj",
]


def rows():
    sets = [("K3.1", "wedge", p, k3.p1_spec) for p in k3.P1_POINTS]
    sets += [("K3.5", "wedge", p, k3.x_spec) for p in k3.X_POINTS]
    out = []
    for claim, model, p, spec in sets:
        R = {L: k3.ensemble_of(spec(p, L)) for L in (4, 6, 8)}
        for L in (4, 6, 8):
            r = R[L]
            e = (
                (None, None)
                if L == 4
                else k3.exponent(k3.value(R[L - 2], "chi10_corner_max"), k3.value(r, "chi10_corner_max"), L - 2, L)
            )
            out.append(row(claim, model, p, L, r, e))
    for L in (4, 6):
        r = k3.ensemble_of([(f"{k3.XI}/F8_P3_eps/{k3.chain_name(L, 2.41, 0, 0)}", 0)])
        out.append(row("K3.2", "epsilon", (2.41, 0.0, 0.0), L, r, (None, None), key="xi10"))
    return out


def row(claim, model, p, L, r, e, key="xi10_cmax"):
    chi, err = k3.value(r, "chi10")
    xi, xe = k3.value(r, key + "L")
    f = lambda v, n: "" if v is None else f"{v:.{n}f}"
    return [
        claim,
        model,
        p[0],
        p[1],
        p[2],
        L,
        f"{chi:.5f}",
        f"{err:.5f}",
        r["chi10_corner_max"]["label"],
        f"{chi / k3.FREE_CHI10[L][0]:.4f}",
        f"{xi:.5f}",
        f"{xe:.5f}",
        f(e[0], 4),
        f(e[1], 4),
        r["ntraj"],
    ]


def main():
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "K3_chi10_scaling.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows())
    print(path)


if __name__ == "__main__":
    main()
