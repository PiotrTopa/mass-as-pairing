#!/usr/bin/env python3
"""K8.2 -- anti-seesaw, tier 2, by the pre-registered tests T1-T5 on the new P_c chains with the stored h <= 2 rows as
the fixed reference: T1 PASS (monotone rise), T2 FAIL (exponent outside the window), T3 UNDECIDED (not a single
power; saturation holds), T4 PASS (no SSB), T5 PASS (V-check) -> "anti-seesaw holds, scaling prediction wrong": a
crossover from a quadratic-like onset to saturation, not a power law. Recorded: E2 (sigma channel), E3 (8^4 quench).
Reads data/derived/n1stage/{S1,S1b,e3_quench_L8_Pc.npz} and the free baselines (about 1-2 min)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1b as B
from masspairing.claimcheck import Check

c = Check("K8.2")
rows, res = B.analyse()
an = res["anti_seesaw"]
fmt = B.fmt
lats = ("L6x12", "L6", "L8")

# T1
t1 = an["T1"]
for lat in ("L6x12", "L6"):
    for k, s in t1[lat].items():
        c.item(
            f"T1 {lat} h = {k}: light pair-channel midpoint cosh mass, step",
            (fmt(s["m_lo"], 3), fmt(s["m_hi"], 3), s["pull"]),
            s["ok"],
            "{0[0]} -> {0[1]}, {0[2]:.1f} sigma",
        )
c.item(
    "T1 8^4: m(1.5) between m(1) and m(2) (distances)",
    (
        fmt(t1["L8"]["m1"], 3),
        fmt(t1["L8"]["m1_5"], 3),
        fmt(t1["L8"]["m2"], 3),
        t1["L8"]["pull_above_1"],
        t1["L8"]["pull_below_2"],
    ),
    t1["L8"]["ok"],
    "{0[0]} < {0[1]} < {0[2]} ({0[3]:.0f} / {0[4]:.0f} sigma)",
)
c.item(
    "T1 PASS",
    an["T1"]["ok"],
    an["T1"]["ok"] and all(s["pull"] > 5 for lat in ("L6x12", "L6") for s in t1[lat].values()),
)
c.item(
    "recorded: 8^4 m(3) (above m(2))",
    (fmt(t1["L8_h3_recorded"]["m3"], 3), t1["L8_h3_recorded"]["pull"]),
    t1["L8_h3_recorded"]["pull"] > 20,
    "{0[0]}, {0[1]:.0f} sigma",
)

# T2
for lat in lats:
    t2, ph = an["T2"][lat], an["post_hoc_shape"][lat]
    c.record(
        f"T2 {lat}",
        f"kappa {fmt(t2['kappa'], 3)} c {t2['c'][0]:.3f} chi^2/dof {t2['chi2_dof']:.2f} -> {t2['status']}; post hoc: "
        f"additive kappa {fmt(ph['additive']['kappa'], 2)} "
        f"(chi^2/dof {ph['additive']['chi2_dof']:.1f}); quadrature m0 free: kappa "
        f"{fmt(ph['quadrature_m0_free']['kappa'], 2)} m0 {fmt(ph['quadrature_m0_free']['m0'], 3)}; local kappa "
        f"{np.round(ph['local_kappa_quadrature'], 2).tolist()}",
    )
t2 = an["T2"]["L6x12"]
c.item(
    "T2 6^3x12: kappa, c, chi^2/dof -> FAIL (kappa + 2 sigma < 0.7 with chi^2/dof <= 3)",
    (t2["kappa"][0], t2["kappa"][1], t2["c"][0], t2["chi2_dof"], t2["status"]),
    t2["status"] == "FAIL"
    and t2["chi2_dof"] <= 3
    and t2["kappa"][0] + 2 * t2["kappa"][1] < 0.7
    and abs(t2["kappa"][0] - 0.44) < 0.03,
    "{0[0]:.2f}({0[1]:.2f}), {0[2]:.2f}, {0[3]:.1f} -> {0[4]}",
)
c.item(
    "T2 6^4 / 8^4: kappa, chi^2/dof -> UNDECIDED (not a single power of h)",
    tuple(
        v
        for lat in ("L6", "L8")
        for v in (an["T2"][lat]["kappa"][0], an["T2"][lat]["kappa"][1], an["T2"][lat]["chi2_dof"])
    ),
    all(an["T2"][lat]["status"] == "UNDECIDED" and an["T2"][lat]["chi2_dof"] > 3 for lat in ("L6", "L8")),
    "{0[0]:.2f}({0[1]:.2f}) {0[2]:.0f}; {0[3]:.2f}({0[4]:.2f}) {0[5]:.0f}",
)
lk = {lat: an["post_hoc_shape"][lat]["local_kappa_quadrature"] for lat in lats}
c.item(
    "recorded (post hoc): local quadrature exponent between adjacent h falls monotonically at 8^4 (crossover)",
    np.round(lk["L8"], 1).tolist(),
    all(lk["L8"][i + 1] < lk["L8"][i] for i in range(3)) and lk["L8"][0] > 1.5 and lk["L8"][-1] < 0.7,
)
c.item(
    "recorded (post hoc): local exponent first / last step on 6^4 and 6^3x12",
    (lk["L6"][0], lk["L6"][-1], lk["L6x12"][0], lk["L6x12"][-1]),
    True,
    "{0[0]:.1f} -> {0[1]:.1f}; {0[2]:.1f} -> {0[3]:.1f}",
)
ad = {lat: an["post_hoc_shape"][lat]["additive"] for lat in lats}
c.item(
    "recorded (post hoc): additive form m0 + c h^kappa: kappa (chi^2/dof) on 6^3x12, 6^4, 8^4",
    tuple(v for lat in lats for v in (ad[lat]["kappa"][0], ad[lat]["kappa"][1], ad[lat]["chi2_dof"])),
    True,
    "{0[0]:.2f}({0[1]:.2f}) [{0[2]:.1f}], {0[3]:.2f}({0[4]:.2f}) [{0[5]:.0f}], {0[6]:.2f}({0[7]:.2f}) [{0[8]:.0f}]",
)
c.item(
    "recorded (post hoc): the quadrature form with m0 free prefers m0 -> 0 at 8^4",
    an["post_hoc_shape"]["L8"]["quadrature_m0_free"]["m0"],
    True,
    "{0[0]:.3f}({0[1]:.3f})",
)

# T3 and saturation
for lat in lats:
    t3 = an["T3"][lat]
    c.record(
        f"T3 {lat}",
        f"x {fmt(t3['x'], 2)} vs kappa(2 - eta) {fmt(t3['predicted_x'], 2)}; chi^2/dof {t3['chi2_dof']:.1f} -> "
        f"{t3['status']}; S(pi) "
        + " ".join(f"{h}:{fmt(v, 2)}" for h, v in t3["S_values"].items())
        + f"; S_inf {fmt(t3['Sinf'], 2)}",
    )
c.item(
    "T3 UNDECIDED on every lattice (chi^2/dof > 3)",
    [round(an["T3"][lat]["chi2_dof"]) for lat in lats],
    all(an["T3"][lat]["status"] == "UNDECIDED" and an["T3"][lat]["chi2_dof"] > 3 for lat in lats),
)
c.item(
    "recorded: fitted decay exponent x on 6^3x12, 6^4, 8^4 (> 2, far above kappa(2 - eta))",
    [fmt(an["T3"][lat]["x"], 1) for lat in lats],
    all(an["T3"][lat]["x"][0] > 2 for lat in lats),
)
sat = an["saturation"]
c.item(
    "saturation: S(pi)(3)/S_inf and |Sigma_stag|(3)/SMG on 6^3x12, 6^4, 8^4 (within 25 % at 2 sigma; central values "
    "below)",
    [(round(sat[lat]["Spi"]["ratio"], 2), round(sat[lat]["Sab"]["ratio"], 2)) for lat in lats],
    all(sat[lat]["ok"] for lat in lats)
    and all(0.65 < sat[lat]["Spi"]["ratio"] < 0.76 and 0.8 < sat[lat]["Sab"]["ratio"] < 0.9 for lat in lats),
)

# T4 / T5
t4, t5 = an["T4"], an["T5"]
c.item(
    "T4: e(1.5) - e_free (8^4 / 6^4 new chains), R_peak(1.5, 8^4) -> PASS",
    (t4["e_minus_free"], t4["R_peak8"]),
    t4["ok"] and t4["e_minus_free"][0] < 0.1 and abs(t4["R_peak8"][0] - 1) < 0.05,
    "{0[0][0]:.3f}({0[0][1]:.3f}), {0[1][0]:.2f}({0[1][1]:.2f})",
)
steps = {lat: t4["S_never_rises"][lat]["steps"] for lat in lats}
c.item(
    "T4: S(pi) never rises; every step from h >= 0.5 is a fall at > 2.5 sigma on every lattice; the 6^4 0 -> 0.5 fall",
    -steps["L6"][0]["rise_pull"],
    all(s["rise_pull"] <= 2 for lat in steps for s in steps[lat])
    and all(s["rise_pull"] < -2.5 for lat in steps for s in steps[lat][1:])
    and -2 < steps["L6"][0]["rise_pull"] < 0,
    "{:.1f} sigma",
)
h3 = t4["h3_recorded"]
c.item(
    "recorded (h = 3, not in the letter): e - e_free; R_peak(3, 8^4) and its distance from 1; 6^4 R_peak not compared",
    (h3["e_minus_free"], h3["R_peak8"], abs(h3["R_peak8"][0] - 1) / h3["R_peak8"][1]),
    h3["ok_e"] and not h3["ok_peak"] and abs(h3["R_peak8"][0] - 1) < 0.04 and h3["e_minus_free"][0] < 0.1,
    "{0[0][0]:.3f}({0[0][1]:.3f}); {0[1][0]:.3f}({0[1][1]:.3f}), {0[2]:.1f} sigma",
)
r3_6 = B.R_peak(B.R(rows, "L6", 2.41, 3.0, "new"))
chi3 = (B.chi_over_free(B.R(rows, "L8", 2.41, 3.0, "new"))[0], B.chi_over_free(B.R(rows, "L6", 2.41, 3.0, "new"))[0])
c.item(
    "recorded (h = 3): R_peak on 6^4; chi_L/free 8^4 vs 6^4",
    (r3_6, chi3),
    True,
    "{0[0][0]:.3f}({0[0][1]:.3f}); {0[1][0]:.2f} vs {0[1][1]:.2f}",
)
c.item(
    "T5: t = 2 cosh mass 8^4/6^4 at h = 2 (stored), 1.5, 3 (new) within 15 %; h = 0 midpoint 8^4/6^4 (halves)",
    (t5["h2_t2"]["ratio"], t5["h1.5_t2"]["ratio"], t5["h3_t2"]["ratio"], t5["h0_midpoint"]["ratio"]),
    t5["ok"] and t5["h1.5_t2"]["within15"] and t5["h3_t2"]["within15"],
    "{0[0]:.3f}, {0[1]:.2f}, {0[2]:.2f}; {0[3]:.2f}",
)
c.item("reading by the order", an["reading"], an["reading"].startswith("anti-seesaw holds, scaling prediction wrong"))

# E2 (recorded)
for lat in lats:
    seq = t4["S_never_rises"][lat]["S"]
    c.record(f"E2 {lat}: S(pi) at P_c, h = 0 ... 3", " -> ".join(f"{v[0]:.1f}" for _, v in seq))
sab8 = [B.ref(rows, "L8", h)["Sab"][0] for h in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
c.item(
    "E2: |Sigma_stag| at P_c falls monotonically through h = 3 on 8^4",
    np.round(sab8, 3).tolist(),
    all(sab8[i + 1] < sab8[i] for i in range(5)),
)

# E3 (recorded)
e3 = res["E3"]
for h in ("h1", "h2"):
    e = e3[h]
    c.record(
        f"E3 {h}",
        f"quench {fmt(e['m_quench'], 3)} vs h=0 same configurations {fmt(e['m_h0_same_cfgs'], 3)} vs annealed "
        f"{fmt(e['m_annealed'], 3)}; fraction {fmt(e['fraction'], 2)}; "
        f"paired {fmt(e['paired_dm'], 3)} ({e['paired_pull']:.1f} sigma, {e['n_positive']}/{e['n']} > 0); "
        f"G_L(p_min)/free quench {fmt(e['GL_p0_quench_over_free'], 2)} annealed "
        f"{fmt(e['GL_p0_annealed_over_free'], 3)}",
    )
c.item(
    "E3: the stored re-measurement at h = 0 reproduces the stored CL_t (identity)",
    max(e3["identity"]),
    max(e3["identity"]) < 1e-12,
    "{:.1e}",
)
c.item(
    "E3 h = 2: rise on every configuration at fixed sigma (paired, n positive), fixed-sigma fraction of the "
    "annealed rise",
    (e3["h2"]["paired_dm"], e3["h2"]["paired_pull"], e3["h2"]["n_positive"], e3["h2"]["n"], e3["h2"]["fraction"]),
    e3["h2"]["paired_pull"] > 10 and e3["h2"]["n_positive"] == e3["h2"]["n"] and 0.2 < e3["h2"]["fraction"][0] < 0.5,
    "{0[0][0]:+.3f}({0[0][1]:.3f}) {0[1]:.1f} sigma, {0[2]}/{0[3]}; {0[4][0]:.2f}({0[4][1]:.2f})",
)
c.item(
    "E3 h = 1: no resolved fixed-sigma rise (fraction, paired)",
    (e3["h1"]["fraction"], e3["h1"]["paired_dm"], e3["h1"]["paired_pull"]),
    abs(e3["h1"]["fraction"][0]) < 0.3 and e3["h1"]["paired_pull"] < 4,
    "{0[0][0]:.2f}({0[0][1]:.2f}); {0[1][0]:+.3f}({0[1][1]:.3f}) {0[2]:.1f} sigma",
)
g15 = B.g_of(B.R(rows, "L8", 2.41, 1.5, "new"))
c.item(
    "recorded: G_L(p_min)/free at (8^4, h = 1.5) inside the predicted 0.784-0.821",
    g15,
    0.784 <= g15[0] <= 0.821,
    "{0[0]:.3f}({0[1]:.3f})",
)

# controls (live)
s1 = B.inject_s1(rows)
c.item(
    "S1' injected seesaw (m -> m0/(1 + 3h) on the new rows) -> T1 FAIL, fall alert fires",
    True,
    not s1["ok"] and s1["C4_alert"]["fires"],
)
s5 = B.control_s5(rows)
c.item(
    "S5' synthetic m^2 = m0^2 + (0.3h)^2 with the real block scatter -> kappa (PASS); synthetic saturating m0 + "
    "0.5(1 - e^-h) -> status (chi^2/dof)",
    (s5["linear"]["kappa"], s5["linear"]["status"], s5["saturating"]["status"], s5["saturating"]["chi2_dof"]),
    s5["linear_ok"] and s5["saturating_ok"] and s5["linear"]["status"] == "PASS",
    "{0[0][0]:.3f}({0[0][1]:.3f}) {0[1]}; {0[2]} ({0[3]:.0f})",
)
k = an["T2"]["L6x12"]["kappa"]
sw2, sws = B.T3(rows, "L6x12", k, swap="h2"), B.T3(rows, "L6x12", k, swap="series")
c.item("S4' label swap y <-> 3.0 at h = 2 -> T3 not PASS", sw2["status"], sw2["status"] in ("UNDECIDED", "FAIL"))
c.item(
    "S4' full-series swap (y = 3.0 at h = 0.5, 1, 2) returns a vacuous PASS (x, chi^2/dof): T3's PASS clause has no "
    "resolution requirement (recorded criterion defect)",
    (sws["status"], sws["x"], sws["chi2_dof"]),
    sws["status"] == "PASS" and sws["x"][1] > 5,
    "{0[0]} x = {0[1][0]:.0f}({0[1][1]:.0f}), {0[2]:.1f}",
)
c.done()
