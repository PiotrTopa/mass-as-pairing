#!/usr/bin/env python3
"""Result tables of the N1 stage claims (reads data/derived only):

results/K4_composite_L8.csv   K4.3: elementary and composite amplitudes per unit h, rho, A on 8^4, 6^4, 6^3x12
results/K5_verdict_stage1.csv K5.5: r_L (t* log-ratio / free), g, e per (y, lattice, h)
results/K5_verdict_1b.csv     K5.8: the y = 3.0 clauses on the new stage-1b samples
results/K6_alpha_L.csv        K6.3: alpha_L, alpha_R with free references, p-odd norm ratio, swap distance
results/K8_gap_vs_h.csv       K8.1/K8.2/M.1: light pair-channel midpoint cosh mass, S(pi), |Sigma_stag| vs h
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis import stage1b as B
from masspairing.data import RESULTS

LAT = {(8, 8): "8^4", (6, 6): "6^4", (6, 12): "6^3x12"}


def write(name, header, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in r])
    print("written", RESULTS / name)


def k4(rows):
    out = []
    for e in sorted(stage1.k4(rows)["rows"].values(), key=lambda e: (e["Lt"], -e["L"], e["y"], e["h"])):
        out.append(
            [LAT[(e["L"], e["Lt"])], e["y"], e["h"], *e["phi_R_per_h"], e["phi_R_over_free"], *e["phi_T_R_per_h"]]
            + [
                e["rho"][0],
                e["rho"][1] if np.isfinite(e["rho"][1]) else "",
                "resolved" if e["rho"][2] == "resolved" else "lower bound",
            ]
            + list(e["A"])
        )
    write(
        "K4_composite_L8.csv",
        [
            "lattice",
            "y",
            "h",
            "phi_R/h",
            "err",
            "phi_R/h over free",
            "phi_T_R/h",
            "err",
            "rho",
            "rho_err",
            "rho_kind",
            "A",
            "A_err",
        ],
        out,
    )


def verdict_stage1(rows):
    out = []
    for y in (2.41, 3.0):
        for L, Lt in ((8, 8), (6, 6), (6, 12)):
            for h in (0.5, 1.0, 2.0):
                r = stage1.get(rows, y, L, Lt, h)
                e = r.get("e", (float("nan"), float("nan")))
                out.append([y, LAT[(L, Lt)], h, *r["r_L"][:2], r["r_L"][3], *r["g"], e[0], e[1]])
    write(
        "K5_verdict_stage1.csv",
        ["y", "lattice", "h", "r_L", "err", "estimator", "g=GL_p0/free", "err", "e (6^4,8^4)", "err"],
        out,
    )


def verdict_1b(rows):
    v = B.verdict_y3(rows, "new")
    out = []
    for h, d in v["rows"].items():
        out.append(
            [
                float(h),
                *d["r_L8"],
                *d["r_L6"],
                *d["g8"],
                *d["g6"],
                *d["e_minus_free"],
                *d["R_peak8"],
                *d["chiLf8"],
                *d["chiLf6"],
            ]
            + [d["a"], d["b1"], d["b2"], d["b3"], d["c"], d["partial_growth"], v["class"]]
        )
    write(
        "K5_verdict_1b.csv",
        [
            "h",
            "r_L^cosh 8^4",
            "err",
            "r_L^cosh 6^4",
            "err",
            "g 8^4",
            "err",
            "g 6^4",
            "err",
            "e-e_free",
            "err",
            "R_peak 8^4",
            "err",
        ]
        + ["chi_L/free 8^4", "err", "chi_L/free 6^4", "err", "a", "b1", "b2", "b3", "c", "partial growth", "class"],
        out,
    )


def alpha():
    out = []
    for r in sorted(stage1.alpha_records("S1"), key=lambda r: (r["base"][0] == 0, r["Lt"], -r["L"], r["y"], r["h"])):
        lat = LAT[(r["L"], r["Lt"])] + ("" if r["base"][0] > 0 else " pppa")
        nf = r["rd_L"]["normO1"] / r["free"]["rd_L"]["normO1"]
        out.append(
            [lat, r["y"], r["h"], r["n"], r["rd_L"]["alpha"], r["rd_L"]["alpha_err"], r["free"]["rd_L"]["alpha"]]
            + [
                r["rd_R"]["alpha"],
                r["rd_R"]["alpha_err"],
                r["free"]["rd_R"]["alpha"],
                nf,
                r["swap_sigma"],
                r["floor_m"],
            ]
        )
    write(
        "K6_alpha_L.csv",
        [
            "lattice",
            "y",
            "h",
            "n_cfg",
            "alpha_L",
            "err",
            "alpha_L free",
            "alpha_R",
            "err",
            "alpha_R free",
            "|O_L(p1)|/free",
            "swap sigma",
            "floor m",
        ],
        out,
    )


def gap(rows):
    out = []
    for lat, name in (("L8", "8^4"), ("L6", "6^4"), ("L6x12", "6^3x12")):
        for y in (2.41, 3.0):
            for h in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                for sel in ("stored", "new"):
                    r = B.R(rows, lat, y, h, sel)
                    if r is None:
                        continue
                    t = r["Lt"] // 2 - 1
                    out.append([name, y, h, sel, t, *r["m"], r["free"]["m"], *r["Spi"], *r["Sab"], *B.g_of(r)])
    write(
        "K8_gap_vs_h.csv",
        [
            "lattice",
            "y",
            "h",
            "samples",
            "t",
            "m_cosh light pair channel",
            "err",
            "free",
            "S(pi)",
            "err",
            "|Sigma_stag|",
            "err",
            "GL_p0/free",
            "err",
        ],
        out,
    )


def main():
    rows1 = stage1.stage1_rows()
    k4(rows1)
    verdict_stage1(rows1)
    alpha()
    rows, _ = B.analyse()
    verdict_1b(rows)
    gap(rows)


if __name__ == "__main__":
    main()
