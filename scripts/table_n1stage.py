#!/usr/bin/env python3
"""Result tables of the N1 stage claims (reads data/derived only):

results/K4_composite_L8.csv   K4.3: elementary and composite amplitudes per unit h, rho, A on 8^4, 6^4, 6^3x12
results/K5_verdict_stage1.csv K5.5: r_L (t* log-ratio / free), g, e per (y, lattice, h)
results/K5_verdict_1b.csv     K5.8: the y = 3.0 clauses on the new stage-1b samples
results/K6_alpha_L.csv        K6.3: alpha_L, alpha_R with free references, p-odd norm ratio, swap distance
results/K8_gap_vs_h.csv       K8.1/K8.2/M.1: light pair-channel midpoint cosh mass, S(pi) (C0 and 10-block errors),
                              |Sigma_stag| vs h
results/K5_verdict_1c.csv     K5.8: the y = 3.0 clauses on the stage-1c 8^4 replicas, their pool, the pool with 1b
results/K8_onset_6x6.csv      K8.3: the 6^4 P_c curve m(t = 2) vs h and the F1 onset fit
results/K8_free_compare.csv   K8.4: the light observables at P_c (and the 6^4 y = 2.0 / 3.0 references) against free
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from masspairing.analysis import direction as D
from masspairing.analysis import stage1
from masspairing.analysis import stage1b as B
from masspairing.analysis import stage1c as C1c
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
    s10 = stage1.sigma_channel_rows()  # the plain 10-block errors of K8.1 (stored stage-1 chains)
    out = []
    for lat, name in (("L8", "8^4"), ("L6", "6^4"), ("L6x12", "6^3x12")):
        for y in (2.41, 3.0):
            for h in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                for sel in ("stored", "new"):
                    r = B.R(rows, lat, y, h, sel)
                    if r is None:
                        continue
                    t = r["Lt"] // 2 - 1
                    e10 = s10.get((lat, y, h), {}).get("Spi", ("", ""))[1] if sel == "stored" else ""
                    out.append([name, y, h, sel, t, *r["m"], r["free"]["m"], *r["Spi"], e10, *r["Sab"], *B.g_of(r)])
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
            "err (C0)",
            "err (10 blocks, K8.1)",
            "|Sigma_stag|",
            "err",
            "GL_p0/free",
            "err",
        ],
        out,
    )


def verdict_1c(res1c):
    S = res1c["A_scores"]
    sets = [(f"replica {r}", S["per_replica"][r]) for r in ("rA", "rB")]
    sets += [("pooled rA+rB", S["pooled"]), ("pooled rA+rB+1b", S["pooled_with_1b"])]
    out = []
    for name, v in sets:
        for h in ("1.0", "2.0"):
            d = v["rows"][h]
            out.append(
                [name, float(h), *d["r_L8"], *d["g8"], *d["e_minus_free"], *d["R_peak8"], *d["chiLf8"]]
                + [d["a"], d["b"], d["partial_growth"], v["class"]]
            )
    write(
        "K5_verdict_1c.csv",
        ["set", "h", "r_L^cosh 8^4", "err", "g 8^4", "err", "e-e_free", "err", "R_peak 8^4", "err"]
        + ["chi_L/free 8^4", "err", "a", "b", "partial growth", "class"],
        out,
    )


def onset(res1c):
    Bres = res1c["B"]
    out = [[pt["h"], pt["origin"], *pt["m"]] for pt in sorted(Bres["curve"], key=lambda p: (p["h"], p["origin"]))]
    F = Bres["F1"]
    out.append(
        ["F1", f"p = {F['p'][0]:.4f}({F['p'][1]:.4f}), c = {F['c']:.5f}, chi2/dof = {F['chi2_dof']:.3f}", "", ""]
    )
    write("K8_onset_6x6.csv", ["h", "origin", "m(t=2)", "err"], out)


def free_compare():
    out = []
    for r in D.free_compare():
        Rm = r["Rm"] or ["", ""]
        out.append(
            [r["src"], r["lat"], r["y"], r["h"], r["n"], *r["m"], r["m_free"], *Rm, *r["dmeff"], *r["g"], *r["c"]]
            + [*r["S"], *r["O4"]]
        )
    write(
        "K8_free_compare.csv",
        ["source", "lattice", "y", "h", "n", "m", "err", "m_free", "m/m_free", "err", "dmeff", "err", "g", "err"]
        + ["c=chi_L/free", "err", "S(pi)", "err", "O4", "err"],
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
    res1c = C1c.analyse()[2]
    verdict_1c(res1c)
    onset(res1c)
    free_compare()


if __name__ == "__main__":
    main()
