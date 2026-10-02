#!/usr/bin/env python3
"""Calibration of the Yukawa axis (claim I.3):

results/I3_calibration.csv   every stored point: L, kappa, y, N, acceptance, max tau_int, |Sigma_stag|,
                             chi_Sigma,stag, xi_2,stag
results/I3_calibration.json  y_1(L = 8) with its 16-84 % range, the y_2(L = 8) bracket, the misclassified points of
                             the three rescalings
"""

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from masspairing.analysis.calibration import ORDERED, misclassified, point_summary, y1_parabola  # noqa: E402
from masspairing.data import RESULTS, derived, load_chain  # noqa: E402


def main():
    R = {}
    for p in sorted(derived("k7", "calib").glob("*.npz")):
        s, meta = load_chain(p)
        r = point_summary(s, meta)
        r["source"] = meta["source"]
        R[(r["L"], round(r["kappa"], 4), round(r["y"], 4))] = r
    RESULTS.mkdir(exist_ok=True)
    cols = ["L", "kappa", "y", "ntraj", "acceptance", "tau_int_max", "Sigma_stag_abs", "e_Sigma_stag_abs"]
    cols += ["chi_Sigma_stag", "e_chi_Sigma_stag", "xi2_stag", "e_xi2_stag", "source"]
    with open(RESULTS / "I3_calibration.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for (L, k, y), r in sorted(R.items()):
            vals = [
                x
                for q in ("Sigma_stag_abs", "chi_Sigma_stag", "xi2_stag")
                for x in (f"{r[q]['mean']:.6g}", f"{r[q]['err']:.2g}")
            ]
            w.writerow([L, k, y, r["ntraj"], f"{r['acceptance']:.4g}", f"{r['tau_max']:.4g}"] + vals + [r["source"]])

    k8 = sorted(y for (L, k, y) in R if L == 8 and k == 0.2)
    low = [y for y in k8 if y <= 3.2]
    chi = np.array([R[(8, 0.2, y)]["chi_Sigma_stag"]["mean"] for y in low])
    err = np.array([R[(8, 0.2, y)]["chi_Sigma_stag"]["err"] for y in low])
    i = int(np.argmax(chi))
    y1 = y1_parabola(low[i - 1 : i + 2], chi[i - 1 : i + 2], err[i - 1 : i + 2])
    S = {y: R[(8, 0.2, y)]["Sigma_stag_abs"]["mean"] for y in k8}
    hi = [y for y in k8 if y > 3.0]
    pts = [(y, S[y] >= ORDERED) for y in k8]
    out = dict(
        y1_L8=dict(median=y1[0], p16=y1[1], p84=y1[2], reference=float(np.sqrt(2) * 1.706)),
        y2_L8_bracket=dict(
            ordered=max(y for y in hi if S[y] >= ORDERED),
            disordered=min(y for y in hi if S[y] < ORDERED),
            reference=float(np.sqrt(2) * 2.69),
        ),
        misclassified_L8={
            name: misclassified(pts, s) for name, s in (("sqrt2", np.sqrt(2)), ("1", 1.0), ("1/sqrt2", 1 / np.sqrt(2)))
        },
        P_c=[round(float(np.sqrt(2) * 1.706), 3), -0.01],
    )
    (RESULTS / "I3_calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"results/I3_calibration.csv ({len(R)} points), results/I3_calibration.json")


if __name__ == "__main__":
    main()
