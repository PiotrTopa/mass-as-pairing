#!/usr/bin/env python3
"""results/K3_linear_response.csv: R(L) = [chi10(wedge, g) - chi10(completion, g)]/g at P_c and the eps-channel
control R_eps(L) (V<|phi_stag|^2>), L = 4, 6 (exact-propagator chains) and 8 (dense measurements of the o and r
streams), with the ratios R(8)/R(6), R(6)/R(4) and the verdict of K3.3."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from masspairing.analysis import k3
from masspairing.data import RESULTS

NCOPY = {("compl", 0.01): 1268, ("compl", 0.02): 1071, ("wedge", 0.01): 1500, ("wedge", 0.02): 1500}


def dense8(m, g):
    """Dense 8^4 chi10 and V<|phi_stag|^2> of one chain (o stream from the copy point, r stream 50 later)."""
    chis, ctrl = [], []
    for rep, k0 in (("", NCOPY[(m, g)]), ("_rA", NCOPY[(m, g)] + 50)):
        ((s, kk),), metas = k3.streams([(k3.p3_rel(m, g, 8, rep), k0)])
        c = k3.cut(s, kk)
        chis.append(np.asarray(c["ts_chi10"], float))
        ctrl.append(8**4 * np.asarray(c["ts_phi_stag_sq"], float))
    return k3.combine_streams(chis)[:2], k3.combine_streams(ctrl)[:2]


def responses(g):
    """{L: (R, err, R_eps, err)}."""
    diff = lambda a, b: ((a["mean"] - b["mean"]) / g, np.hypot(a["err"], b["err"]) / g)
    R = {}
    for L in (4, 6):
        w = k3.ensemble_of([(k3.p3_rel("wedge", g, L), 450 if (L, g) == (6, 0.01) else 0)])
        c = k3.ensemble_of([(k3.p3_rel("compl", g, L), 0)])
        R[L] = diff(w["chi10"], c["chi10"]) + diff(w["V_phi_stag_sq"], c["V_phi_stag_sq"])
    (wc, we), (wv, wve) = dense8("wedge", g)
    (cc, ce), (cv, cve) = dense8("compl", g)
    R[8] = ((wc - cc) / g, np.hypot(we, ce) / g, (wv - cv) / g, np.hypot(wve, cve) / g)
    return R


def main():
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "K3_linear_response.csv"
    cols = ["g", "L", "R", "R_err", "R_eps", "R_eps_err", "ratio_to_previous_L", "ratio_err", "verdict"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for g in (0.01, 0.02):
            R = responses(g)
            v = k3.response_verdict(R[8][:2], R[6][:2])
            for L, prev in ((4, None), (6, 4), (8, 6)):
                q = k3.ratio(R[L][:2], R[prev][:2]) if prev else ("", "")
                fmt = lambda x, n: x if x == "" else f"{x:.{n}f}"
                w.writerow(
                    [
                        g,
                        L,
                        f"{R[L][0]:.4f}",
                        f"{R[L][1]:.4f}",
                        f"{R[L][2]:.2f}",
                        f"{R[L][3]:.2f}",
                        fmt(q[0], 4),
                        fmt(q[1], 4),
                        v if L == 8 else "",
                    ]
                )
    print(path)


if __name__ == "__main__":
    main()
