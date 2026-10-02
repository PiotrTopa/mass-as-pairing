#!/usr/bin/env python3
"""Figures of the N1 stage claims (reads data/derived only):

figures/K8_light_pair_gap_vs_h   K8.1/K8.2/K5.8/M.1: the light pair-channel midpoint cosh mass vs h at P_c (y = 2.41)
                                 and in SMG (y = 3.0) on 8^4, 6^4, 6^3x12 (stored rows; at P_c also the new h = 1.5, 3)
figures/K5_E8_no_plateau         K5.7 (E8): cosh effective mass of the light correlator vs t on 6^3x12 -- no plateau
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from masspairing.analysis import stage1b as B
from masspairing.plotting import MARKERS, SERIES, figure, save

LATS = (("L8", "8⁴", 0.0), ("L6", "6⁴", 0.02), ("L6x12", "6³×12", -0.02))


def gap_vs_h(rows):
    fig, ax = figure(ncols=2, width=3.6, height=2.8)
    for k, (y, title) in enumerate(((2.41, "P$_c$ (y = 2.41)"), (3.0, "SMG (y = 3.0)"))):
        a = ax[0, k]
        for i, (lat, label, dx) in enumerate(LATS):
            pts = []
            for h in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                r = B.R(rows, lat, y, h, "stored") or (B.R(rows, lat, y, h, "new") if y == 2.41 else None)
                if r is not None:
                    pts.append((h + dx, *r["m"]))
            hs, m, e = zip(*pts, strict=True)
            a.errorbar(hs, m, yerr=e, color=SERIES[i], marker=MARKERS[i], label=label, linestyle="-")
        a.set_title(title)
        a.set_xlabel("h (Majorana mass on the heavy doublet)")
        a.set_ylabel("light pair-channel cosh mass (midpoint)" if k == 0 else "cosh mass (midpoint)")
        a.legend(loc="lower right" if y == 2.41 else "center right")
    save(fig, "K8_light_pair_gap_vs_h")


def no_plateau(rows):
    fig, ax = figure(width=3.6, height=2.8)
    a = ax[0, 0]
    for i, (y, h, label) in enumerate(
        ((3.0, 0.0, "y = 3.0, h = 0"), (2.41, 0.0, "y = 2.41, h = 0"), (2.41, 2.0, "y = 2.41, h = 2"))
    ):
        r = B.R(rows, "L6x12", y, h, "stored")
        pts = [(t + 0.05 * (i - 1), *r["mcosh_t"][str(t)]) for t in range(1, r["Lt"] // 2)]
        ts, m, e = zip(*pts, strict=True)
        a.errorbar(ts, m, yerr=e, color=SERIES[i], marker=MARKERS[i], label=label, linestyle="-")
    a.set_title("6³×12: light pair channel, no plateau")
    a.set_xlabel("t")
    a.set_ylabel("cosh effective mass")
    a.legend(loc="upper right")
    save(fig, "K5_E8_no_plateau")


def main():
    rows, _ = B.analyse()
    gap_vs_h(rows)
    no_plateau(rows)


if __name__ == "__main__":
    main()
