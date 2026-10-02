#!/usr/bin/env python3
"""figures/i3_calibration.{pdf,png}: L = 8, kappa = 0.2 scan (claim I.3). Left: |Sigma_stag| against y with the
ordered threshold 0.15 and the published window rescaled by sqrt(2) (dashed). Right: chi_Sigma,stag against y with
sqrt(2) x 1.706 (dashed)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from masspairing.analysis.calibration import ORDERED, Y1_REF, Y2_REF, point_summary  # noqa: E402
from masspairing.data import derived, load_chain  # noqa: E402
from masspairing.plotting import NEUTRAL, SERIES, figure, save  # noqa: E402


def main():
    pts = []
    for p in sorted(derived("k7", "calib").glob("calib__L8_k0.2__*.npz")):
        r = point_summary(*load_chain(p))
        pts.append(r)
    pts.sort(key=lambda r: r["y"])
    y = np.array([r["y"] for r in pts])
    fig, ax = figure(ncols=2)
    for j, (key, label) in enumerate(
        (
            ("Sigma_stag_abs", r"$\langle|\Sigma_\mathrm{stag}|\rangle$"),
            ("chi_Sigma_stag", r"$\chi_{\Sigma,\mathrm{stag}}$"),
        )
    ):
        a = ax[0, j]
        a.errorbar(y, [r[key]["mean"] for r in pts], yerr=[r[key]["err"] for r in pts], fmt="o", color=SERIES[0])
        for edge in (Y1_REF, Y2_REF) if j == 0 else (Y1_REF,):
            a.axvline(np.sqrt(2) * edge, color=NEUTRAL, ls="--", lw=0.8)
        if j == 0:
            a.axhline(ORDERED, color=NEUTRAL, ls=":", lw=0.8)
        a.set_xlabel("y  (L = 8, κ = 0.2)")
        a.set_ylabel(label)
    save(fig, "i3_calibration")


if __name__ == "__main__":
    main()
