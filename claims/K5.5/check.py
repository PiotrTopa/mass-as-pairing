#!/usr/bin/env python3
"""K5.5 -- stage 1 by the pre-registered letter: the gap-floor gate, the SSB clause (b), the (a)/(c) clauses and the
controls of the verdict logic. Reads data/derived/n1stage/S1 (26 chains), data/derived/n1stage/S0/L4 (the S4 control)
and the free baselines (about 1-2 min)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis.n1stage import N1DIR, exponent, open_chain
from masspairing.claimcheck import Check

c = Check("K5.5")
rows = stage1.stage1_rows()
G = lambda y, L, Lt, h, bc="aaaa": stage1.get(rows, y, L, Lt, h, bc)
V = {y: stage1.verdict(rows, y) for y in (2.41, 3.0)}

# 1. gap-floor gate on 6^3x12
for y in (2.41, 3.0):
    r0 = G(y, 6, 12, 0.0)
    fits = [G(y, 6, 12, h)["fit_L"] for h in (0.0, 0.5, 1.0, 2.0)]
    c.record(
        f"y={y} 6^3x12 fit chi^2/dof at h = 0, 0.5, 1, 2; exists",
        f"{[round(f['chi2'] / 2, 2) for f in fits]}; {[f['exists'] for f in fits]}",
    )
    c.item(
        f"y={y}: primary estimator (t* log-ratio) m_L(0) on 6^3x12 < 0.3; cosh mass t = 5, fit m (recorded, > 0.3)",
        (r0["mL"], r0["mcosh_L_t5"], (r0["fit_L"]["m"], r0["fit_L"]["err"])),
        not V[y]["gap_readable"] and r0["mL"][0] < 0.3 and r0["mcosh_L_t5"][0] > 0.3 and r0["fit_L"]["m"] > 0.3,
        "{0[0][0]:.3f}({0[0][1]:.3f}); {0[1][0]:.2f}({0[1][1]:.2f}), {0[2][0]:.2f}({0[2][1]:.2f})",
    )
    c.item(f"y={y}: verdict string", V[y]["verdict"], V[y]["verdict"].startswith("NO VERDICT at L ≤ 8"))
f241 = [G(2.41, 6, 12, h)["fit_L"]["chi2"] / 2 for h in (0.0, 0.5, 1.0, 2.0)]
c.item("y = 2.41: the fit exists at h <= 1 (chi^2/dof), not at h = 2", f241, max(f241[:3]) < 0.12 and f241[3] > 2, "{}")
f3 = [G(3.0, 6, 12, h)["fit_L"]["chi2"] / 2 for h in (0.0, 0.5, 1.0, 2.0)]
r0 = G(3.0, 6, 12, 0.0)
pull = (r0["mcosh_L_t4"][0] - r0["mcosh_L_t5"][0]) / np.hypot(r0["mcosh_L_t4"][1], r0["mcosh_L_t5"][1])
c.item("y = 3.0: fit chi^2/dof range at every h", (min(f3), max(f3)), min(f3) > 9, "{0[0]:.1f}-{0[1]:.1f}")
c.item(
    "y = 3.0, h = 0: cosh effective mass t = 4 -> 5 (not a single cosh)",
    (r0["mcosh_L_t4"], r0["mcosh_L_t5"], pull),
    pull > 15,
    "{0[0][0]:.3f}({0[0][1]:.3f}) -> {0[1][0]:.3f}({0[1][1]:.3f}), {0[2]:.0f} sigma",
)

# 2. SSB clause (b)
for y in (2.41, 3.0):
    for h, (e, ee) in V[y]["e"].items():
        r6, r8 = G(y, 6, 6, h), G(y, 8, 8, h)
        ef = np.log(r8["free"]["chi_L"] / r6["free"]["chi_L"]) / np.log(4096 / 1296)
        c.record(
            f"y={y} h={h}: e, e - 2 sigma, e_free, chi_L/free 6^4 / 8^4",
            f"{e:+.3f}({ee:.3f}), {e - 2 * ee:+.3f}, {ef:.3f}, {r6['chi_L_over_free']:.4f} / "
            f"{r8['chi_L_over_free']:.4f}",
        )
e3 = V[3.0]["e"]
c.item(
    "(b) flagged at y = 3.0 by the h = 2 row only: e, e - 2 sigma",
    (e3[2.0][0], e3[2.0][1], e3[2.0][0] - 2 * e3[2.0][1]),
    V[3.0]["b_flag"]
    and e3[2.0][0] - 2 * e3[2.0][1] > 0.5
    and not any(e - 2 * ee > 0.5 for h, (e, ee) in e3.items() if h != 2.0),
    "{0[0]:.3f}({0[1]:.3f}), {0[2]:.3f}",
)
c.item(
    "y = 3.0: e at h = 0.5, 1 (unreadable)",
    (e3[0.5], e3[1.0]),
    True,
    "{0[0][0]:.2f}({0[0][1]:.2f}), {0[1][0]:.2f}({0[1][1]:.2f})",
)
r60, r80 = G(3.0, 6, 6, 0.0), G(3.0, 8, 8, 0.0)
e0 = exponent(r60["chi_L"][0], r60["chi_L"][1], r80["chi_L"][0], r80["chi_L"][1], r60["V"], r80["V"])
c.item("y = 3.0: e at h = 0 (recorded)", e0, True, "{0[0]:.2f}({0[1]:.2f})")
e2 = V[2.41]["e"]
c.item(
    "not flagged at y = 2.41: e at h = 0.5, 1, 2",
    [round(v[0], 2) for v in e2.values()],
    not V[2.41]["b_flag"] and all(abs(e - 0.44) < 0.06 for e, _ in e2.values()),
)
efree = [
    np.log(G(3.0, 8, 8, h)["free"]["chi_L"] / G(3.0, 6, 6, h)["free"]["chi_L"]) / np.log(4096 / 1296)
    for h in (0.0, 0.5, 1.0, 2.0)
]
c.item(
    "free massless doublet: e_free on (6^4, 8^4), range",
    (min(efree), max(efree)),
    0.38 < min(efree) and max(efree) < 0.46,
    "{0[0]:.2f}-{0[1]:.2f}",
)
qq = {}
for L in (8, 6):
    d, _ = open_chain(N1DIR / "S1" / f"L{L}" / f"L{L}_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz")
    k = d["ts_cfg_traj"] >= 100
    qq[L] = d["ts_chi_L_sum_pmin"][k].mean() / d["ts_chi_L_sum"][k].mean()
c.item(
    "flagged row (3.0, h = 2): chi_L(p_min)/chi_L(0) at 8^4 / 6^4 (no p = 0 peak); chi_L/free at 8^4 / 6^4",
    (qq[8], qq[6], G(3.0, 8, 8, 2.0)["chi_L_over_free"], G(3.0, 6, 6, 2.0)["chi_L_over_free"]),
    qq[8] > 1.0 and qq[6] < 0.8 and G(3.0, 8, 8, 2.0)["chi_L_over_free"] < 0.013,
    "{0[0]:.2f} / {0[1]:.2f}; {0[2]:.4f} / {0[3]:.4f}",
)

# 3. (a)/(c) clauses
for y in (2.41, 3.0):
    for lab in ("L8", "L6", "L6x12"):
        c.record(
            f"y={y} {lab}: r_L (primary estimator, t* log-ratio / free), g",
            "  ".join(
                f"h={n['h']}: {n['r_L'][0]:.3f}({n['r_L'][1]:.3f}), {n['g'][0]:.4f}({n['g'][1]:.4f})"
                for n in V[y][lab]["rows"]
            ),
        )
for y in (2.41, 3.0):
    raw = [G(y, 6, 12, h)["mL"][0] / G(y, 6, 12, 0.0)["mL"][0] for h in (0.5, 1.0, 2.0)]
    c.record(
        f"y={y} 6^3x12: raw t* ratio m_L(h)/m_L(0) at h = 0.5, 1, 2 (the free normalisation is void there)",
        ", ".join(f"{v:.2f}" for v in raw),
    )
c.item(
    "r_L >= 0.7 - 2 sigma on every lattice and h at both y; (c) nowhere",
    True,
    all(V[y][lab]["a_rL"] for y in V for lab in ("L8", "L6", "L6x12"))
    and not any(V[y][lab]["c"] for y in V for lab in ("L8", "L6", "L6x12")),
)
g8 = [G(3.0, 8, 8, h)["g"][0] for h in (0.0, 0.5, 1.0, 2.0)]
g6 = [G(3.0, 6, 6, h)["g"][0] for h in (0.0, 0.5, 1.0, 2.0)]
g12 = [G(3.0, 6, 12, h)["g"][0] for h in (0.0, 0.5, 1.0, 2.0)]
c.item(
    "y = 3.0: g = GL_p0/free at 8^4 (range), 6^4, 6^3x12 -- the (a) g-clause holds",
    (min(g8), max(g8), min(g6), max(g6), min(g12), max(g12)),
    V[3.0]["L8"]["a_g"] and max(g8) < 0.004 and max(g6) < 0.02,
    "8^4 {0[0]:.4f}-{0[1]:.4f}; 6^4 {0[2]:.3f}-{0[3]:.3f}; 6^3x12 {0[4]:.4f}-{0[5]:.4f}",
)
gp = {h: G(2.41, 8, 8, h)["g"] for h in (0.0, 0.5, 1.0, 2.0)}
c.item(
    "y = 2.41: g at 8^4 h = 0, 0.5, 1, 2 -- the (a) g-clause fails at h = 2 only; 6^4 h = 2",
    (gp[0.0], gp[0.5], gp[1.0], gp[2.0], G(2.41, 6, 6, 2.0)["g"]),
    not V[2.41]["L8"]["a_g"]
    and gp[2.0][0] > 0.8 + 2 * gp[2.0][1]
    and all(gp[h][0] <= 0.8 + 2 * gp[h][1] for h in (0.5, 1.0)),
    "{0[0][0]:.3f}({0[0][1]:.3f}), {0[1][0]:.3f}({0[1][1]:.3f}), {0[2][0]:.3f}({0[2][1]:.3f}), "
    "{0[3][0]:.3f}({0[3][1]:.3f}); {0[4][0]:.3f}",
)
eb = {y: [e + 2 * ee for e, ee in V[y]["e"].values()] for y in V}
c.item(
    "e + 2 sigma < 0.5: passes at y = 2.41 (values), fails at y = 3.0",
    [round(v, 2) for v in eb[2.41]],
    V[2.41]["a_e"] and not V[3.0]["a_e"],
)
c.item(
    "class readings: y = 3.0 at L = 8 / 6; y = 2.41 at L = 8 / 6",
    (V[3.0]["class_L8"], V[3.0]["class_L6"], V[2.41]["class_L8"], V[2.41]["class_L6"]),
    V[3.0]["class_L8"] == "(b)"
    and V[3.0]["class_L6"] == "(b)"
    and V[2.41]["class_L8"] == "UNDECIDED"
    and V[2.41]["class_L6"] == "UNDECIDED",
)
c.item(
    "V-stability of the class reading between 6^4 and 8^4 (both y)", True, V[2.41]["V_stable"] and V[3.0]["V_stable"]
)

# 5. controls
ctl = stage1.controls(rows)
c.item("S2 injected SSB -> (b) at both y", True, ctl["S2"]["y2.41"]["b_flag"] and ctl["S2"]["y3.0"]["b_flag"])
s1 = stage1.inject_s1(rows)
c.item(
    "S1 as written (m_L -> m_L/(1 + 3h)) -> (c) at 6^4 P_c only; at 8^4 r_L(2) stays (measured rise exceeds the "
    "injected fall)",
    stage1.get(s1, 2.41, 8, 8, 2.0)["r_L"][0],
    ctl["S1"]["y2.41"]["class_L6"] == "(c)"
    and ctl["S1"]["y2.41"]["class_L8"] == "UNDECIDED"
    and not ctl["S1"]["M"]["triggered"],
    "{:.2f}",
)
c.item(
    "S1'' (r_L := 1/(1 + 3h), g free) -> (c) at both volumes at P_c",
    True,
    ctl["S1_double_prime"]["y2.41"]["class_L8"] == "(c)" and ctl["S1_double_prime"]["y2.41"]["class_L6"] == "(c)",
)
c.item(
    "free-null injection -> never (a), never (b)",
    True,
    all(
        ctl["null_free"][y]["class_L8"] == "UNDECIDED"
        and ctl["null_free"][y]["class_L6"] == "UNDECIDED"
        and not ctl["null_free"][y]["b_flag"]
        for y in ("y2.41", "y3.0")
    ),
)
s4 = ctl["S4"]["rows"]
c.item(
    "S4: stage-0 y = 2.0 rows within max(2 sigma, 3 %): max |r_L - 1|, chi_L/free, GL_p0/free ranges",
    (
        max(abs(r["r_L"] - 1) for r in s4 if "r_L" in r),
        min(r["chi_L_over_free"] for r in s4),
        max(r["chi_L_over_free"] for r in s4),
        min(r["GL_over_free"] for r in s4),
        max(r["GL_over_free"] for r in s4),
    ),
    all(r.get("ok", True) and r["ok_chi"] and r["ok_g"] for r in s4),
    "{0[0]:.3f}; {0[1]:.2f}-{0[2]:.2f}; {0[3]:.2f}-{0[4]:.2f}",
)
gate = ctl["S4"]["gate"]
c.item(
    "S4: free instrument gate on every stage-1 lattice: max |GL_p0^free - 1| on 6^4/8^4 aaaa",
    max(abs(g["GL_p0"] - 1) for g in gate if g["bc"] == "aaaa" and g["Lt"] == g["L"]),
    all(abs(g["GL_p0"] - 1) < 1e-9 for g in gate if g["bc"] == "aaaa" and g["Lt"] == g["L"]),
    "{:.1e}",
)
c.item(
    "the reader's synthetic self-test ((a) flat, (c) + M injected fall, (b) injected growth)",
    ctl["selftest"]["ok"],
    ctl["selftest"]["ok"],
)
c.done()
