#!/usr/bin/env python3
"""K8.5 -- direction of the shift under the explicit heavy mass: at P_c the light half moves toward the symmetric
(y = 2.0) side, not into SMG. The pre-registered tests P1 (6^4, three clauses; MIXED as written) and P3 ((y, h) =
(3.0, 3): deep SMG, the large-N size of the shift falsified) applied as written; the stored P_c indicators on three
lattices and the 4^4 stage-0 positions (post hoc, labelled); on the SMG side (y = 3.0) g and c rise toward free with h
on all three lattices and on the four independent stage-1c 8^4 replicas, while the pair-channel cosh mass falls only
on 6^4. Rows by the stage-1b reader from data/derived/n1stage/{S0,S1,S1b,S1c,TH} (about 20 s)."""

import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import direction as D
from masspairing.analysis import stage1b as B
from masspairing.claimcheck import Check

c = Check("K8.5")
bl = D.bl_tables()
res = D.analyse_direction(bl)
R = res["rows"]
for k, r in R.items():
    c.record(
        f"6^4 row {k} (y = {r['y']:g}, h = {r['h']:g}, n = {r['n']})",
        f"g {r['g'][0]:.4f}({r['g'][1]:.4f}) c {r['c'][0]:.4f}({r['c'][1]:.4f}) S(pi) {r['S'][0]:.3f}({r['S'][1]:.3f}) "
        f"O4 {r['O4'][0]:.4f} m {r['m'][0]:.4f}({r['m'][1]:.4f}) min 50-traj acceptance {r['acc']:.2f}",
    )

# 1. P1 as written
P = res["P1"]
c.item(
    "P1: f_g(0) -> f_g(3), Delta f_g, significance; f_c(0) -> f_c(3), Delta f_c, significance -- both toward SYM "
    "beyond 2 sigma",
    (
        P["f_g(0)"][0],
        P["f_g(3)"][0],
        P["df_g"],
        abs(P["df_g"][0] / P["df_g"][1]),
        P["f_c(0)"][0],
        P["f_c(3)"][0],
        P["df_c"],
    )
    + (abs(P["df_c"][0] / P["df_c"][1]),),
    P["df_g"][0] + 2 * P["df_g"][1] < 0
    and P["df_c"][0] + 2 * P["df_c"][1] < 0
    and abs(P["df_g"][0] + 0.12839068223243014) < 1e-12
    and abs(P["df_c"][0] + 0.2893511098290409) < 1e-12,
    "f_g {0[0]:.3f} -> {0[1]:.3f} ({0[2][0]:.3f}({0[2][1]:.3f}), {0[3]:.0f} sigma); f_c {0[4]:.3f} -> {0[5]:.3f} "
    "({0[6][0]:.3f}({0[6][1]:.3f}), {0[7]:.0f} sigma)",
)
c.item(
    "P1 S clause short: dS = |S - S_SMG(0)| - |S - S_SYM(3)| (closer to SYM, not beyond 2 sigma)",
    (P["dS"][0], P["dS"][1], P["dS"][0] / P["dS"][1]),
    0 < P["dS"][0] < 2 * P["dS"][1],
    "{0[0]:.2f}({0[1]:.2f}) = {0[2]:.1f} sigma",
)
c.item("P1 verdict as written (no 'toward SMG' clause holds)", P["verdict"], P["verdict"] == "mixed / undecided")
c.item(
    "P1 classifier controls: synthetic SMG-ward row; row at SYM(3)",
    tuple(res["P1_controls"].values()),
    res["P1_controls"] == {"synthetic_toward_SMG": "toward SMG", "synthetic_at_SYM": "toward SYM"},
)

# 2. P3 as written
Q = res["P3"]
c.item(
    "P3 at (y, h) = (3.0, 3): g, c (< 0.03: deep SMG) -- the large-N size of the shift is falsified",
    (Q["verdict"], Q["g"][0], Q["c"][0]),
    Q["verdict"] == "deep SMG",
    "{0[0]}: g {0[1]:.4f}, c {0[2]:.4f}",
)
c.item(
    "recorded direction at y = 3.0, h = 2 -> 3 on 6^4: Delta g, Delta c (toward free)",
    (Q["dg_from_h2"], Q["dc_from_h2"]),
    Q["dg_from_h2"][0] > 2 * Q["dg_from_h2"][1] and Q["dc_from_h2"][0] > 2 * Q["dc_from_h2"][1],
    "+{0[0][0]:.4f}({0[0][1]:.4f}), +{0[1][0]:.4f}({0[1][1]:.4f})",
)

# 3. post hoc, labelled: the stored P_c series and stage 0
rows1b = B.read_all(B.free_tables(B.FREE_1B))
for lat in ("L8", "L6", "L6x12"):
    O4 = D.ref_series(rows1b, lat, "O4")
    g = D.ref_series(rows1b, lat, "g")
    cc = D.ref_series(rows1b, lat, "c")
    c.item(
        f"{lat} P_c, h = 0 -> 3 (post hoc): O4 falls monotonically, g rises monotonically; c",
        (O4[0], O4[-1], g[0], g[-1], cc[0], cc[-1]),
        np.all(np.diff(O4) < 0) and np.all(np.diff(g) > 0),
        "O4 {0[0]:.4f} -> {0[1]:.4f}, g {0[2]:.3f} -> {0[3]:.3f}, c {0[4]:.3f} -> {0[5]:.3f}",
    )
f4 = D.stage0_positions(bl)
c.item(
    "4^4 stage 0 (post hoc): f_g, f_c of P_c between y = 2.0 and 3.0 at h = 0 -> 2 (toward SYM)",
    (f4[("g", 0)], f4[("g", 2)], f4[("c", 0)], f4[("c", 2)]),
    f4[("g", 2)] < f4[("g", 0)] and f4[("c", 2)] < f4[("c", 0)],
    "f_g {0[0]:.3f} -> {0[1]:.3f}, f_c {0[2]:.3f} -> {0[3]:.3f}",
)

# 4. the SMG side (y = 3.0)
srows, rep = D.smg_rows(bl)
SER = (("L8", (0.0, 0.5, 1.0, 2.0)), ("L6", (0.0, 0.5, 1.0, 2.0, 3.0)), ("L6x12", (0.0, 0.5, 1.0, 2.0)))
for lat, hs in SER:
    for O in ("g", "c"):
        good, (a, b) = D.monotone_toward_free(srows, lat, O, hs)
        c.item(
            f"y = 3.0 {lat}: {O} from h = 0 to {hs[-1]:g}, every step non-decreasing within 2 sigma, overall rise "
            "> 3 sigma",
            (a, b),
            good,
            "{0[0][0]:.4f}({0[0][1]:.4f}) -> {0[1][0]:.4f}({0[1][1]:.4f})",
        )
g0 = srows[("L8", 0.0)]["g"]
for d, v in rep.items():
    c.item(
        f"y = 3.0 8^4 stage-1c replica {d} (h = {v['h']:g}): g above the h = 0 value ({g0[0]:.4f}) beyond 2 sigma",
        v["g"],
        v["g"][0] - g0[0] > 2 * np.hypot(v["g"][1], g0[1]),
        "{0[0]:.4f}({0[1]:.4f})",
    )
for lat, hs in SER:
    c.record(
        f"y = 3.0 {lat}: light pair-channel cosh mass m at h = {hs}",
        " ".join(f"{srows[(lat, h)]['m'][0]:.4f}" for h in hs),
    )
m8 = [srows[("L8", h)]["m"][0] for h in (0.0, 0.5, 1.0, 2.0)]
m6 = srows[("L6", 0.0)]["m"][0], srows[("L6", 2.0)]["m"][0], srows[("L6", 3.0)]["m"][0]
mt = srows[("L6x12", 0.0)]["m"][0], srows[("L6x12", 2.0)]["m"][0]
c.item(
    "y = 3.0: the falling pair-channel mass is a 6^4 statement -- 8^4 spread over h = 0 ... 2; 6^4 h = 0, 2, 3; 6^3x12 "
    "h = 0 -> 2 (rises)",
    (max(m8) - min(m8),) + m6 + mt,
    max(m8) - min(m8) < 0.01 and m6[1] < m6[0] - 0.03 and mt[1] > mt[0],
    "{0[0]:.3f}; {0[1]:.3f}, {0[2]:.3f}, {0[3]:.3f}; {0[4]:.3f} -> {0[5]:.3f}",
)

# 5. sabotage
fake = copy.deepcopy(srows)
fake[("L8", 2.0)]["g"] = (0.8 * g0[0], g0[1])
c.item(
    "sabotage: a synthetic 8^4 y = 3.0 h = 2 row with g at 80 % of its h = 0 value is flagged",
    True,
    not D.monotone_toward_free(fake, "L8", "g", (0.0, 0.5, 1.0, 2.0))[0],
)
c.done()
