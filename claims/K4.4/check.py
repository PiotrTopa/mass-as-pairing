#!/usr/bin/env python3
"""K4.4 -- production-bc continuity at P_c: 6^4 pppa vs 6^4 aaaa (y = 2.41, h in {0, 0.5}), every number next to its
free value at the same bc. Reads data/derived/n1stage/S1 (the four chains; about 1 min with the full stage-1 read)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.claimcheck import Check

c = Check("K4.4")
rows = stage1.stage1_rows()
P = {h: stage1.get(rows, 2.41, 6, 6, h, "pppa") for h in (0.0, 0.5)}
Q = {h: stage1.get(rows, 2.41, 6, 6, h, "aaaa") for h in (0.0, 0.5)}
pull = lambda a, b: (a[0] - b[0]) / np.hypot(a[1], b[1])
for h in (0.0, 0.5):
    for bc, r in (("pppa", P[h]), ("aaaa", Q[h])):
        f = r["free"]
        c.record(
            f"{bc} h={h}",
            f"chi_L/free {r['chi_L_over_free']:.3f} (free chi_L {f['chi_L']:.4f}); GL_p0/free "
            f"{r['g'][0]:.3f}({r['g'][1]:.3f}); "
            f"m_L(t*) {r['mL'][0]:.4f}({r['mL'][1]:.4f}) [free {f['mL']:.3f}]; cosh mass t=2 "
            f"{r['mcosh_L_t2'][0]:.3f}({r['mcosh_L_t2'][1]:.3f}); "
            f"A {r['A'][0]:+.4f}({r['A'][1]:.4f}) [free {f['A']:.3f}]; tau_B {r['tau_B']:.1f}; acceptance "
            f"{r['acc_min50']:.2f}",
        )
p0 = P[0.0]
nfail = sum(abs(v["half_pull"]) > 3 for v in p0["stats"].values())
c.item(
    "1. pppa h = 0 is the slow chain: tau_B (trajectories), acceptance, half-split failures of 10 (aaaa partner: "
    "0), tau_B/Delta",
    (p0["tau_B"], p0["acc_min50"], nfail, p0["tau_B_over_D"]),
    p0["tau_B"] > 20
    and p0["acc_min50"] < 0.7
    and nfail >= 5
    and all(abs(v["half_pull"]) <= 3 for v in Q[0.0]["stats"].values()),
    "{0[0]:.0f}, {0[1]:.2f}, {0[2]}, {0[3]:.1f}",
)
c.item(
    "2. chi_L/free: pppa h = 0 / 0.5 vs aaaa h = 0 / 0.5",
    (P[0.0]["chi_L_over_free"], P[0.5]["chi_L_over_free"], Q[0.0]["chi_L_over_free"], Q[0.5]["chi_L_over_free"]),
    0.28 < P[0.0]["chi_L_over_free"] < 0.34
    and 0.54 < Q[0.0]["chi_L_over_free"] < 0.60
    and 0.33 < P[0.5]["chi_L_over_free"] < 0.39,
    "{0[0]:.2f} / {0[1]:.2f} vs {0[2]:.2f} / {0[3]:.2f}",
)
c.item(
    "2. GL_p0/free bc-independent within 1 sigma: pulls at h = 0, 0.5",
    (pull(P[0.0]["g"], Q[0.0]["g"]), pull(P[0.5]["g"], Q[0.5]["g"])),
    abs(pull(P[0.0]["g"], Q[0.0]["g"])) < 1 and abs(pull(P[0.5]["g"], Q[0.5]["g"])) < 1,
    "{0[0]:+.2f}, {0[1]:+.2f}",
)
c.item(
    "2. h-response chi_L(0.5)/chi_L(0) pppa vs aaaa; r_L(0.5) (t* log-ratio / free) pppa vs aaaa",
    (P[0.5]["chi_L_over_h0"], Q[0.5]["chi_L_over_h0"], P[0.5]["r_L"][:2], Q[0.5]["r_L"][:2]),
    abs(pull(P[0.5]["chi_L_over_h0"], Q[0.5]["chi_L_over_h0"])) < 1.5
    and abs(pull(P[0.5]["r_L"][:2], Q[0.5]["r_L"][:2])) < 2,
    "{0[0][0]:.2f}({0[0][1]:.2f}) vs {0[1][0]:.2f}({0[1][1]:.2f}); {0[2][0]:.2f}({0[2][1]:.2f}) vs "
    "{0[3][0]:.2f}({0[3][1]:.2f})",
)
c.item(
    "3. light t* log-ratio at h = 0: pppa vs aaaa (free values) -- bc-dependent as in the free theory",
    (P[0.0]["mL"][0], Q[0.0]["mL"][0], P[0.0]["free"]["mL"], Q[0.0]["free"]["mL"]),
    P[0.0]["mL"][0] < 0.04
    and Q[0.0]["mL"][0] > 0.07
    and 0.20 < P[0.0]["free"]["mL"] < 0.22
    and 0.40 < Q[0.0]["free"]["mL"] < 0.41,
    "{0[0]:.3f} vs {0[1]:.3f} ({0[2]:.3f} vs {0[3]:.3f})",
)
rp, ra = P[0.5]["rho"], Q[0.5]["rho"]
c.item(
    "4. rho = |phi_T_R/phi_R| at h = 0.5: pppa, aaaa, ratio, significance",
    (rp[0], rp[1], ra[0], ra[1], rp[0] / ra[0], (rp[0] - ra[0]) / np.hypot(rp[1], ra[1])),
    rp[0] / ra[0] > 3.5 and (rp[0] - ra[0]) / np.hypot(rp[1], ra[1]) > 5 and rp[0] < 0.2,
    "{0[0]:.4f}({0[1]:.4f}), {0[2]:.4f}({0[3]:.4f}), {0[4]:.1f}x, {0[5]:.0f} sigma",
)
c.item(
    "4. phi_R/h over free and phi_T_R/h at h = 0.5: pppa vs aaaa",
    (P[0.5]["phi_R_over_free"], Q[0.5]["phi_R_over_free"], P[0.5]["phi_T_R_per_h"][0], Q[0.5]["phi_T_R_per_h"][0]),
    True,
    "{0[0]:.2f} vs {0[1]:.2f}; {0[2]:+.5f} vs {0[3]:+.5f}",
)
c.item(
    "5. A(0.5) pppa vs aaaa (free values), pull",
    (P[0.5]["A"], Q[0.5]["A"], P[0.5]["free"]["A"], Q[0.5]["free"]["A"], pull(P[0.5]["A"], Q[0.5]["A"])),
    abs(pull(P[0.5]["A"], Q[0.5]["A"])) < 1,
    "{0[0][0]:.3f}({0[0][1]:.3f}) vs {0[1][0]:.3f}({0[1][1]:.3f}) (free {0[2]:.3f} / {0[3]:.3f}), {0[4]:+.1f} sigma",
)
c.done()
