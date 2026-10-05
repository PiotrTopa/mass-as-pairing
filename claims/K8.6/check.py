#!/usr/bin/env python3
"""K8.6 -- the large-N (Gaussian) sign of the shift: the explicit mass h C_chi removes the heavy doublet from the
fermion loop that drives the staggered-sigma instability, so the large-N critical coupling RISES with h. The closed form
is checked against the exact doublet determinant and re-derived from the chain runner's own operator; the full
mean-field potential keeps the sign at finite sigma0; the shift is second order at small h and is not a power of h
over [0.5, 3]. The stored P_c structure factor is fitted (post hoc) by 1/S(pi) = s0 + b delta_L(h) (the fit's
discriminating weight is the subject of K8.8). Dense inverses up to V = 4096 (about 6-7 min on one thread)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import largen as G
from masspairing.analysis import stage1c as C1c
from masspairing.claimcheck import Check

c = Check("K8.6")

# 1. closed form = exact determinant
fd, cf = G.finite_difference_check()
c.item(
    "closed form vs exact doublet determinant (4^4, h = 0.7, y = 1.3): d^2/dsigma0^2 1/2 log det(D^dag D) vs V y^2 "
    "Pi(h) (relative difference)",
    (fd, cf, abs(fd - cf) / abs(cf)),
    abs(fd - cf) / abs(cf) < 1e-5,
    "{0[0]:.5f} vs {0[1]:.5f} ({0[2]:.0e})",
)

# 2. Pi(h), y_MF(h)/y_MF(0) and delta(h) on four boxes
hs = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
for L, Lt in ((4, 4), (6, 6), (8, 8), (6, 12)):
    P = np.array([G.pi(L, Lt, h) for h in hs])
    r = np.sqrt(P[0] / P)
    d = 1 - P / P[0]
    c.record(
        f"{L}^3x{Lt}: Pi(0); y_MF(h)/y_MF(0) at h = 0.25 ... 3", f"{P[0]:.5f}; " + " ".join(f"{x:.4f}" for x in r[1:])
    )
    c.item(
        f"{L}^3x{Lt}: Pi strictly decreasing over h in [0, 3] -> the sigma instability moves to larger y",
        True,
        np.all(np.diff(P) < 0),
    )
    c.item(
        f"{L}^3x{Lt}: small-h shift second order, delta(0.25)/delta(0.5) (h^2: 0.25)",
        d[1] / d[2],
        abs(d[1] / d[2] - 0.25) < 0.04,
        "{:.3f}",
    )
    loc = [np.log(d[i + 1] / d[i]) / np.log(hs[i + 1] / hs[i]) for i in range(2, 6)]
    if (L, Lt) == (4, 4):
        c.record(
            "4^4 (not a data lattice): local exponent of delta over h = 0.5 ... 3", " -> ".join(f"{x:.2f}" for x in loc)
        )
    else:
        c.item(
            f"{L}^3x{Lt}: no power law -- local exponent of delta(h) over h = 0.5 ... 3",
            loc,
            loc[0] - loc[-1] > 0.5,
            " -> ".join(f"{{0[{i}]:.2f}}" for i in range(4)),
        )

# 3. the same sign from the production operator (no closed form), and beyond the Gaussian
TH = {4: [1.0058, 1.0230, 1.0874, 1.1794], 6: [1.0121, 1.0448, 1.1431, 1.2538]}  # the closed-form ratios printed above
hp = (0.0, 0.5, 1.0, 2.0, 3.0)
for L in (4, 6):
    Pp = np.array([-G.curvature(L, h) for h in hp])
    r = np.sqrt(Pp[0] / Pp)
    c.item(
        f"{L}^4 production operator: Pi_prod strictly decreasing; y_MF ratios at h = 0.5, 1, 2, 3 equal the closed "
        "form to <= 2e-4",
        tuple(r[1:]) + (np.max(np.abs(r[1:] - TH[L])),),
        np.all(np.diff(Pp) < 0) and np.max(np.abs(r[1:] - TH[L])) < 2e-4,
        "{0[0]:.4f} {0[1]:.4f} {0[2]:.4f} {0[3]:.4f} (max dev {0[4]:.1e})",
    )
    cl = [G.pi(L, L, h) for h in (0.5, 1.0, 2.0, 3.0)]
    p0 = G.pi(L, L, 0.0)
    c.item(
        f"{L}^4: the closed-form ratios quoted for the production check",
        np.max(np.abs(np.sqrt(p0 / np.array(cl)) - TH[L])),
        np.max(np.abs(np.sqrt(p0 / np.array(cl)) - TH[L])) < 6e-5,
        "max |dev| {:.1e}",
    )
Pa = [-G.curvature(4, 2.0, a=a) for a in (0, 1, 2)]
c.item(
    "4^4, h = 2: the three sigma directions give the same staggered curvature (relative spread < 1e-7)",
    (max(Pa) - min(Pa)) / Pa[2],
    (max(Pa) - min(Pa)) / Pa[2] < 1e-7,
    "{:.1e}",
)
ys = {}
for stag in (True, False):
    ys[stag] = [G.ystar(4, h, stag) for h in hp]
    nm = "staggered" if stag else "uniform"
    c.item(
        f"4^4 full mean-field transition, {nm}: y*(h) at h = 0, 0.5, 1, 2, 3 non-decreasing, y*(3) - y*(0) > 0.1",
        [y[0] for y in ys[stag]],
        all(ys[stag][i + 1][0] >= ys[stag][i][0] for i in range(4)) and ys[stag][-1][0] > ys[stag][0][0] + 0.1,
        " ".join(f"{{0[{i}]:.2f}}" for i in range(5)),
    )
c.item(
    "staggered transition continuous (sigma0 at onset < 0.2), uniform first order (sigma0 jump > 0.5)",
    (max(y[1] for y in ys[True]), min(y[1] for y in ys[False])),
    max(y[1] for y in ys[True]) < 0.2 and min(y[1] for y in ys[False]) > 0.5,
    "{0[0]:.3f}; {0[1]:.3f}",
)
rec = G.recorded_ystar()
c.item(
    "recorded 6^4 staggered y* at h = 0, 3 (stored scan, same code), and the 4^4 rows of that scan equal the live ones",
    (rec["L6_stag"][0.0][0], rec["L6_stag"][3.0][0]),
    rec["L6_stag"][3.0][0] > rec["L6_stag"][0.0][0]
    and all(abs(rec["L4_stag"][h][0] - y[0]) < 1e-9 for h, y in zip(hp, ys[True], strict=True)),
    "{0[0]:.2f} -> {0[1]:.2f}",
)
yb = G.ystar(4, 0.0, True, bmass=1.21)[0]
c.item(
    "control: bosonic mass 1/2 sigma^2 -> 1/2 1.21 sigma^2 raises y*(0) by sqrt(1.21) = 1.1",
    yb / ys[True][0][0],
    abs(yb / ys[True][0][0] - 1.1) < 0.02,
    "{:.3f}",
)

# 4. the stored P_c data (post hoc): 1/S(pi) against delta_L(h) and the competitors
drows = G.data_rows()
res = G.analyse(drows)
for lat, r in res.items():
    R, Pw, Q = r["RPA"], r["POW"], r["QUAD"]
    bare = 2.41**2 * r["Pi"][0] / 3
    c.item(
        f"{lat}: RPA 1/S(pi) = s0 + b delta_L fits (chi^2/dof <= 2, dof 4); b, b/bare large-N y^2 Pi(0)/3",
        (R["chi2_dof"], R["p"][1], R["err"][1], R["p"][1] / bare),
        R["chi2_dof"] <= 2.0,
        "{0[0]:.2f}; b {0[1]:.3f}({0[2]:.3f}), ratio {0[3]:.2f}",
    )
    c.item(
        f"{lat}: a single power of h (3 parameters, chi^2/dof >= 3) and h^2 (chi^2/dof >= 10) fail",
        (Pw["p"][2], Pw["chi2_dof"], Q["chi2_dof"]),
        Pw["chi2_dof"] >= 3.0 and Q["chi2_dof"] >= 10.0,
        "kappa {0[0]:.2f}, chi^2/dof {0[1]:.2f}; h^2 {0[2]:.1f}",
    )
    c.record(
        f"{lat}: local exponents of 1/S - 1/S0 (data) vs delta_L",
        " ".join(f"{x:.2f}" for x in r["local_exp_invS"]) + " vs " + " ".join(f"{x:.2f}" for x in r["local_exp_delta"]),
    )
b = [res[lat]["RPA"]["p"][1] for lat in res]
c.item("RPA amplitude b nearly lattice-independent (spread < 0.25)", max(b) - min(b), max(b) - min(b) < 0.25, "{:.3f}")
rp = G.analyse(drows, G.permuted)
c.item(
    "sabotage: delta_L with permuted h labels fails (chi^2/dof > 50)",
    [rp[lat]["RPA"]["chi2_dof"] for lat in rp],
    all(rp[lat]["RPA"]["chi2_dof"] > 50 for lat in rp),
    "{0[0]:.0f}, {0[1]:.0f}, {0[2]:.0f}",
)
rf = G.analyse(drows, G.both_doublets)
c.record(
    "specificity limit: delta from a mass on BOTH doublets fits too (chi^2/dof, b)",
    ", ".join(f"{lat} {rf[lat]['RPA']['chi2_dof']:.2f}, {rf[lat]['RPA']['p'][1]:.3f}" for lat in rf),
)
for lat, r in res.items():
    c.record(
        f"{lat}: light pair-channel mass m^2 = m0^2 + a delta_L^p (p), Gaussian p = 1 (chi^2/dof), "
        "m^2 = m0^2 + (c h^k)^2",
        f"p = {r['NU_m']['p'][2]:.2f}({r['NU_m']['err'][2]:.2f}) chi^2/dof {r['NU_m']['chi2_dof']:.2f}; p = 1 "
        f"{r['MF_m']['chi2_dof']:.2f}; kappa {r['POW_m']['p'][2]:.2f} chi^2/dof {r['POW_m']['chi2_dof']:.2f}",
    )

# 5. the independent stage-1c 6^4 points against the Gaussian 6^4 curve m^2 = m0^2 + a delta_L
cur = C1c.analyse()[2]["B"]["curve"]
pm = res["L6"]["MF_m"]["p"]
pulls = []
for pt in cur:
    if pt["origin"].startswith("stage 1c"):
        dl = 1 - G.pi(6, 6, pt["h"]) / G.pi(6, 6, 0.0)
        pred = np.sqrt(pm[0] + pm[1] * dl)
        pulls.append((pt["h"], pt["m"][0], pred, (pt["m"][0] - pred) / pt["m"][1]))
c.item(
    "the three stage-1c 6^4 points (h = 0.25, 0.5 pooled, 0.75; in no fit): measured, Gaussian curve, pull (within 2 "
    "sigma)",
    pulls,
    len(pulls) == 3 and all(abs(p[3]) < 2 for p in pulls),
    "; ".join(f"{{0[{i}][1]:.4f}} vs {{0[{i}][2]:.4f}} ({{0[{i}][3]:+.2f}})" for i in range(3)),
)
c.done()
