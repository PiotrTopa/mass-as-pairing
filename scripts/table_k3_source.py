#!/usr/bin/env python3
"""results/K3_source_ssb.csv: the pairing-source test of K3.4 per point and volume -- the weighted slope
S = <Phi_src>/h over h = 0.005, 0.01, 0.02, the h -> 0 intercept m0, rho = S(0.005)/S(0.02), the ratio to the free
response (L = 4, 6, 8: 3.51456, 3.33051, 3.19412) and the exponent d ln S/d ln V on (6,8) with the verdict."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from masspairing.analysis import k3
from masspairing.data import RESULTS

HS = (0.005, 0.01, 0.02)
LS = (4, 6, 8)
FREE = {4: 3.51456, 6: 3.33051, 8: 3.19412}


def main():
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "K3_source_ssb.csv"
    cols = [
        "y",
        "g10",
        "g6",
        "L",
        "S",
        "S_err",
        "m0",
        "m0_err",
        "rho",
        "rho_err",
        "S_over_free",
        "e68",
        "e68_err",
        "verdict",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for g6 in (0.05, 0.0):
            M = {}
            for L in LS:
                for h in HS:
                    r = k3.ensemble_of([(k3.p2_rel(L, g6, h), 0)])
                    M[(L, h)] = (r["phi_src"]["mean"], r["phi_src"]["err"])
            o = k3.source_point(M, HS, LS)
            for L in LS:
                S, eS, _ = o["S"][L]
                last = L == 8
                w.writerow(
                    [
                        2.41,
                        0.05,
                        g6,
                        L,
                        f"{S:.4f}",
                        f"{eS:.4f}",
                        f"{o['m0'][L][0]:+.5f}",
                        f"{o['m0'][L][1]:.5f}",
                        f"{o['rho'][L][0]:.3f}",
                        f"{o['rho'][L][1]:.3f}",
                        f"{S / FREE[L]:.3f}",
                        f"{o['e68'][0]:+.4f}" if last else "",
                        f"{o['e68'][1]:.4f}" if last else "",
                        o["verdict"] if last else "",
                    ]
                )
    print(path)


if __name__ == "__main__":
    main()
