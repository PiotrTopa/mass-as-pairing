#!/usr/bin/env python3
"""figures/k3_chi10_response.pdf: (a) the complete chi10 at P_c relative to the free theory vs L at three link-field
points (K3.1, K3.2); (b) the response R(L) of chi10 to the pure plaquette term at g = 0.01, 0.02 (K3.3).
Reads results/K3_chi10_scaling.csv and results/K3_linear_response.csv."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from masspairing.data import RESULTS
from masspairing.plotting import MARKERS, NEUTRAL, SERIES, figure, save

POINTS = [("2.41", "0.1", "0.1"), ("2.41", "0.05", "0.0"), ("2.41", "0.02", "0.0")]


def read(name):
    with open(RESULTS / name) as fh:
        return list(csv.DictReader(fh))


def main():
    scal = read("K3_chi10_scaling.csv")
    resp = read("K3_linear_response.csv")
    fig, ax = figure(ncols=2)
    a = ax[0, 0]
    for i, (y, g, g6) in enumerate(POINTS):
        rows = [r for r in scal if r["claim"] == "K3.1" and (r["y"], r["g10"], r["g6"]) == (y, g, g6)]
        L = [int(r["L"]) for r in rows]
        free = [float(r["chi10"]) / float(r["chi10_over_free"]) for r in rows]
        val = [float(r["chi10_over_free"]) for r in rows]
        err = [float(r["chi10_err"]) / f for r, f in zip(rows, free, strict=True)]
        a.errorbar(
            [x + 0.08 * (i - 1) for x in L],
            val,
            yerr=err,
            color=SERIES[i],
            marker=MARKERS[i],
            ls="-",
            label=f"({y}, {g}, {g6})",
        )
    a.axhline(1.0, color=NEUTRAL, lw=1.0, ls="--")
    a.text(4.0, 0.93, "free theory", color=NEUTRAL, fontsize=8, va="top")
    a.set_xlabel("$L$")
    a.set_ylabel(r"$\chi_{10}(p=0)\,/\,\chi_{10}^{\rm free}$")
    a.set_xticks([4, 6, 8])
    a.set_ylim(0, 1.1)
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), title="$(y, g_{10}, g_6)$")
    a.set_title("(a) complete 10-channel correlator at $P_c$")
    b = ax[0, 1]
    for i, g in enumerate(("0.01", "0.02")):
        rows = [r for r in resp if r["g"] == g]
        b.errorbar(
            [int(r["L"]) for r in rows],
            [float(r["R"]) for r in rows],
            yerr=[float(r["R_err"]) for r in rows],
            color=SERIES[i],
            marker=MARKERS[i],
            ls="-",
            label=f"$g = {g}$",
        )
    b.set_xlabel("$L$")
    b.set_ylabel(r"$R(L)$")
    b.set_xticks([4, 6, 8])
    b.set_ylim(0, 20)
    b.legend(loc="lower right")
    b.set_title(r"(b) $R = [\chi_{10}^{\rm wedge} - \chi_{10}^{\rm compl}]/g$ at $P_c$")
    print(save(fig, "k3_chi10_response"))


if __name__ == "__main__":
    main()
