#!/usr/bin/env python3
"""figures/k4_takeover.{pdf,png}: response per unit h to the nodal source on flavour 4 against y (claim K4.1).

Left: 4^4 at h = 0.1 (second run). Right: 6^4 at h = 0.3. Elementary amplitude phi_heavy/h and composite amplitude
|phi_T|/h of Psi^4 = chi^1 chi^2 chi^3; logarithmic axis.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from masspairing.analysis.n1inst import takeover_row
from masspairing.data import derived, load_chain
from masspairing.plotting import MARKERS, SERIES, figure, save


def main():
    rows = {}
    for f in sorted(derived("n1inst", "takeover").glob("*.npz")):
        r = takeover_row(*load_chain(f))
        rows[f.stem] = r
    panels = [
        ("4⁴, h = 0.1", [f"L4_y{y}_h0.1_b" for y in ("2", "2.41", "3")]),
        ("6⁴, h = 0.3", [f"L6_y{y}_h0.3_b" for y in ("2", "2.41", "3")]),
    ]
    fig, ax = figure(ncols=2, width=3.7)
    for j, (title, names) in enumerate(panels):
        a = ax[0, j]
        sel = [rows[n] for n in names]
        y = np.array([r["y"] for r in sel])
        a.errorbar(
            y,
            [r["heavy_h"] for r in sel],
            yerr=[r["eheavy_h"] for r in sel],
            fmt=MARKERS[0] + "-",
            color=SERIES[0],
            label="elementary χ⁴",
        )
        a.errorbar(
            y,
            [abs(r["phiT_h"]) for r in sel],
            yerr=[r["ephiT_h"] for r in sel],
            fmt=MARKERS[1] + "-",
            color=SERIES[1],
            label="composite Ψ⁴ = χ¹χ²χ³",
        )
        a.set_yscale("log")
        a.set_xticks([2.0, 2.41, 3.0], ["2.0\nSYM", "2.41\n$P_c$", "3.0\nSMG"])
        a.set_xlabel("y")
        a.set_ylabel("amplitude / h")
        a.set_title(title)
        if j == 0:
            a.legend()
    save(fig, "k4_takeover")


if __name__ == "__main__":
    main()
