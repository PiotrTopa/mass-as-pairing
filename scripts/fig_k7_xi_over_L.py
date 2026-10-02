#!/usr/bin/env python3
"""figures/k7_xi_over_L.{pdf,png}: xi_2,stag/L and S(pi) against y on the kappa = -0.01 line at L = 6 and 8
(claims K7.2, K7.3; first 100 trajectories cut, replicas combined; open markers: not scored). The dashed line marks
P_c, y = 2.41."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.analysis import t3a  # noqa: E402
from masspairing.plotting import MARKERS, NEUTRAL, SERIES, figure, save  # noqa: E402


def main():
    pilot, new = t3a.load_records("pilot"), t3a.load_records("sharpened")
    S = t3a.analyse_sharpened(pilot, new, t3a.analyse_pilot(pilot))
    fig, ax = figure(ncols=2)
    for j, (key, label) in enumerate(
        (("xiL", r"$\xi_{2,\mathrm{stag}}/L$"), ("Spi", r"$S(\pi) = V\langle m^2\rangle$"))
    ):
        a = ax[0, j]
        a.axvline(t3a.YC, color=NEUTRAL, ls="--", lw=0.8)
        for i, L in enumerate((6, 8)):
            pts = sorted((y, s) for (k, LL, y), s in S["comb"].items() if k == -0.01 and LL == L)
            for scored in (True, False):
                sel = [(y, s) for y, s in pts if s["scored"] == scored]
                if not sel:
                    continue
                a.errorbar(
                    [y for y, _ in sel],
                    [s[key] for _, s in sel],
                    yerr=[s["e_" + key] for _, s in sel],
                    fmt=MARKERS[i],
                    color=SERIES[i],
                    mfc=SERIES[i] if scored else "white",
                    label=f"L = {L}" if scored else None,
                )
        a.set_xlabel("y  (κ = −0.01)")
        a.set_ylabel(label)
    ax[0, 0].legend(frameon=False)
    save(fig, "k7_xi_over_L")


if __name__ == "__main__":
    main()
