#!/usr/bin/env python3
"""K8.9 -- the pre-registered direction test P1 repeated unchanged on 8^4 (P1'), with a new 8^4 y = 2.0 pair at
h = 0 and 3, and its amendment 1 (volume trend of g and c at h = 3, P_c against the symmetric point). The claim
certifies the outcome as it fell: P1' MIXED / UNDECIDED by the letter (the light clauses toward SYM, the S(pi)
clause short), amendment 1 SPLIT. Rows by the stage-1b reader from data/derived/n1stage/{S1,S1b,TH} (about 15 s)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from masspairing.analysis import direction as D
from masspairing.claimcheck import Check

c = Check("K8.9")
res = D.p1prime()
R = res["rows"]
for k, r in R.items():
    c.record(
        f"8^4 row {k} (y = {r['y']:g}, h = {r['h']:g}, trajectories {r['nprod']}, dense {r['n']})",
        f"g {r['g'][0]:.4f}({r['g'][1]:.4f}) c {r['c'][0]:.4f}({r['c'][1]:.4f}) S(pi) {r['S'][0]:.3f}({r['S'][1]:.3f}) "
        f"O4 {r['O4'][0]:.4f} m {r['m'][0]:.4f} [free {r['m_free']:.3f}] min 50-traj acceptance {r['acc']:.2f}",
    )
c.item(
    "data sufficiency as pre-registered (FULL: >= 300 production trajectories per new chain)",
    res["sufficiency"],
    res["sufficiency"] == "FULL",
)
V = res["P1prime"]
c.item(
    "clause (i): f_g(0) -> f_g(3), Delta f_g, significance; f_c(0) -> f_c(3), Delta f_c, significance -- both toward "
    "SYM beyond 2 sigma",
    (V["f_g(0)"][0], V["f_g(3)"][0], V["df_g"], abs(V["df_g"][0] / V["df_g"][1]))
    + (V["f_c(0)"][0], V["f_c(3)"][0], V["df_c"], abs(V["df_c"][0] / V["df_c"][1])),
    V["df_g"][0] + 2 * V["df_g"][1] < 0
    and V["df_c"][0] + 2 * V["df_c"][1] < 0
    and abs(V["df_g"][0] + 0.105687095175679) < 1e-12,
    "f_g {0[0]:.3f} -> {0[1]:.3f} ({0[2][0]:.3f}({0[2][1]:.3f}), {0[3]:.1f} sigma); f_c {0[4]:.3f} -> {0[5]:.3f} "
    "({0[6][0]:.3f}({0[6][1]:.3f}), {0[7]:.0f} sigma)",
)
c.item(
    "clause (ii) does NOT hold: dS = |S_Pc(3) - S_SMG(0)| - |S_Pc(3) - S_SYM(3)|, significance",
    (V["dS"][0], V["dS"][1], V["dS"][0] / V["dS"][1], res["S_clause"]),
    abs(V["dS"][0]) < 2 * V["dS"][1] and abs(V["dS"][0] - 0.2996415572323546) < 1e-12,
    "{0[0]:+.3f}({0[1]:.3f}) = {0[2]:.1f} sigma -> {0[3]}",
)
c.item(
    "P1' verdict as written on 8^4 (no 'toward SMG' clause holds either)",
    V["verdict"],
    V["verdict"] == "mixed / undecided",
)
c.item(
    "classifier controls (sabotage): synthetic SMG-ward row; row at SYM(3)",
    tuple(res["controls"].values()),
    res["controls"] == {"synthetic_toward_SMG": "toward SMG", "synthetic_at_SYM": "toward SYM"},
)
A = res["amendment1"]
for O in ("g", "c"):
    a = A[O]
    c.record(
        f"amendment 1, {O}: v = {O}(8^4, h = 3)/{O}(6^4, h = 3) at P_c vs SYM, difference",
        f"{a['v_Pc'][0]:.4f}({a['v_Pc'][1]:.4f}) vs {a['v_SYM'][0]:.4f}({a['v_SYM'][1]:.4f}), "
        f"{a['diff'][0]:+.4f}({a['diff'][1]:.4f}) -> {a['verdict']}",
    )
c.item(
    "amendment 1 as written: SPLIT -- v_Pc vs v_SYM for g ('gap beyond SYM dressing', significance) and for c "
    "('free-like volume trend')",
    (A["g"]["v_Pc"], A["g"]["v_SYM"], abs(A["g"]["diff"][0] / A["g"]["diff"][1]), A["c"]["v_Pc"], A["c"]["v_SYM"]),
    A["g"]["verdict"] == "gap beyond SYM dressing" and A["c"]["verdict"] == "free-like volume trend",
    "g {0[0][0]:.4f}({0[0][1]:.4f}) vs {0[1][0]:.4f}({0[1][1]:.4f}), {0[2]:.1f} sigma; "
    "c {0[3][0]:.4f}({0[3][1]:.4f}) vs {0[4][0]:.4f}({0[4][1]:.4f})",
)
c.item(
    "recorded: 8^4 h-matched O4_Pc(3)/O4_SYM(3); pair-channel m/m_free at SYM h = 0, 3, SMG, P_c(3)",
    (
        R["Pc3"]["O4"][0] / R["SYM3"]["O4"][0],
        R["SYM0"]["m"][0] / R["SYM0"]["m_free"],
        R["SYM3"]["m"][0] / R["SYM3"]["m_free"],
        R["SMG0"]["m"][0] / R["SMG0"]["m_free"],
        R["Pc3"]["m"][0] / R["Pc3"]["m_free"],
    ),
    R["Pc3"]["O4"][0] / R["SYM3"]["O4"][0] > 1.5,
    "{0[0]:.2f}; {0[1]:.2f}, {0[2]:.2f}, {0[3]:.2f}, {0[4]:.2f}",
)
c.done()
