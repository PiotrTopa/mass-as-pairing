#!/usr/bin/env python3
"""Figures of the N1 stage claims (reads data/derived only):

figures/K8_light_pair_gap_vs_h   K8.1/K8.2/K5.8/M.1: the light pair-channel midpoint cosh mass vs h at P_c (y = 2.41)
                                 and in SMG (y = 3.0) on 8^4, 6^4, 6^3x12 (stored rows; at P_c also the new h = 1.5, 3)
figures/K5_E8_no_plateau         K5.7 (E8): cosh effective mass of the light correlator vs t on 6^3x12 -- no plateau
figures/K5_h2_replicas           K5.8: e - e_free at y = 3.0, h = 2 on the stage-1c replicas, their pool, stage 1b
                                 and the pool with stage 1b, against the SSB threshold 0.5 of clause (b1)
figures/K8_onset_6x6             K8.3: the light pair-channel cosh mass m(t = 2) vs h at P_c on 6^4, h = 0 ... 3, with
                                 the stage-1c points and the F1 fit over h <= 1
figures/K8_light_vs_free         K8.1/K8.2/K8.4: at P_c the light pair-channel cosh mass vs h on three lattices with
                                 the free values (8^4, 6^4), and the staggered structure factor S(pi) vs h against its
                                 SMG value
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from masspairing.analysis import direction as D
from masspairing.analysis import stage1b as B
from masspairing.analysis import stage1c as C1c
from masspairing.plotting import MARKERS, NEUTRAL, SERIES, figure, save

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


def h2_replicas(res1b, res1c):
    S = res1c["A_scores"]
    pts = [
        ("replica A", S["per_replica"]["rA"]["rows"]["2.0"]["e_minus_free"]),
        ("replica B", S["per_replica"]["rB"]["rows"]["2.0"]["e_minus_free"]),
        ("A ∪ B", S["pooled"]["rows"]["2.0"]["e_minus_free"]),
        ("stage 1b", res1b["verdict_y3_new"]["rows"]["2.0"]["e_minus_free"]),
        ("A ∪ B ∪ 1b", S["pooled_with_1b"]["rows"]["2.0"]["e_minus_free"]),
    ]
    fig, ax = figure(width=3.6, height=2.8)
    a = ax[0, 0]
    x = np.arange(len(pts))
    a.errorbar(x, [p[1][0] for p in pts], yerr=[p[1][1] for p in pts], color=SERIES[0], marker=MARKERS[0], ls="none")
    a.axhline(0.5, color=NEUTRAL, ls="--", lw=1)
    a.axhline(0.0, color=NEUTRAL, ls=":", lw=1)
    a.text(-0.45, 0.51, "SSB threshold (b1)", ha="left", va="bottom", fontsize=7, color=NEUTRAL)
    a.text(-0.45, 0.01, "free scaling", ha="left", va="bottom", fontsize=7, color=NEUTRAL)
    a.set_xticks(x, [p[0] for p in pts], rotation=20)
    a.set_xlim(-0.5, len(pts) - 0.5)
    a.set_ylabel("e − e$_{free}$ of χ$_L$ on (6⁴, 8⁴)")
    a.set_title("SMG (y = 3.0), h = 2: light pair susceptibility")
    save(fig, "K5_h2_replicas")


def onset_6x6(res1c):
    Bres = res1c["B"]
    groups = {"stage 1, 1b": [], "stage 1c": [], "stage 1 (h = 0.5, not in F1)": []}
    for pt in Bres["curve"]:
        o = pt["origin"]
        g = (
            "stage 1c"
            if o.startswith("stage 1c")
            else ("stage 1 (h = 0.5, not in F1)" if "beside" in o else "stage 1, 1b")
        )
        groups[g].append((pt["h"] + (0.03 if "beside" in o else 0.0), *pt["m"]))
    fig, ax = figure(width=3.6, height=2.8)
    a = ax[0, 0]
    F = Bres["F1"]
    m0, c, p = F["m"]["m0"], F["c"], F["p"][0]
    hh = np.linspace(0, 1, 101)
    a.plot(hh, np.sqrt(m0**2 + c**2 * hh ** (2 * p)), color=NEUTRAL, lw=1, ls="--", label=f"F1, h ≤ 1: p = {p:.2f}")
    for i, (g, pts) in enumerate(groups.items()):
        pts.sort()
        hs, m, e = zip(*pts, strict=True)
        a.errorbar(hs, m, yerr=e, color=SERIES[i], marker=MARKERS[i], ls="none", label=g)
    a.set_xlabel("h (Majorana mass on the heavy doublet)")
    a.set_ylabel("light pair-channel cosh mass m(t = 2)")
    a.set_title("P$_c$ on 6⁴: onset and saturation")
    a.legend(loc="lower right")
    save(fig, "K8_onset_6x6")


def light_vs_free(rows1b):
    fc = D.free_compare()
    pc, _ = D.pc_rows(fc)
    fig, ax = figure(ncols=2, width=3.9, height=2.8)
    a, b = ax[0, 0], ax[0, 1]
    for i, (lat, label, dx) in enumerate(LATS):
        hs = D.HS
        m = [pc[(lat, h)]["m"] for h in hs]
        a.errorbar(
            [h + dx for h in hs],
            [x[0] for x in m],
            yerr=[x[1] for x in m],
            color=SERIES[i],
            marker=MARKERS[i],
            label=label,
        )
        if lat != "L6x12":
            a.plot(hs, [pc[(lat, h)]["m_free"] for h in hs], color=SERIES[i], ls="--", lw=1)
        S = [pc[(lat, h)]["S"] for h in hs]
        b.errorbar(
            [h + dx for h in hs],
            [x[0] for x in S],
            yerr=[x[1] for x in S],
            color=SERIES[i],
            marker=MARKERS[i],
            label=label,
        )
        smg = B.R(rows1b, lat, 3.0, 0.0, "stored")["Spi"][0]
        b.axhline(smg, color=SERIES[i], ls="--", lw=1)
    a.set_title("P$_c$: light pair channel (dashed: free)")
    a.set_xlabel("h (Majorana mass on the heavy doublet)")
    a.set_ylabel("light pair-channel cosh mass (midpoint)")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8))
    b.set_yscale("log")
    b.set_yticks([2, 3, 5, 10, 20], ["2", "3", "5", "10", "20"])
    b.minorticks_off()
    b.set_title("P$_c$: σ channel (dashed: SMG, y = 3.0, h = 0)")
    b.set_xlabel("h (Majorana mass on the heavy doublet)")
    b.set_ylabel("staggered structure factor S(π)")
    b.legend(loc="upper right")
    save(fig, "K8_light_vs_free")


def main():
    rows, res1b = B.analyse()
    gap_vs_h(rows)
    no_plateau(rows)
    res1c = C1c.analyse()[2]
    h2_replicas(res1b, res1c)
    onset_6x6(res1c)
    light_vs_free(rows)


if __name__ == "__main__":
    main()
