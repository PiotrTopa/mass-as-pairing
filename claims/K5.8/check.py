#!/usr/bin/env python3
"""K5.8 -- stage 1b at y = 3.0 (SMG) by the pre-registered clauses on the NEW samples only: (a) the light half keeps
its own gap at h = 1 and h = 2; (b) does not fire; the partial outcome at h = 2 recorded; pooled values labelled;
the controls of the clause logic. Stage 1c (asserted beside): the same clauses on two independent 8^4 replicas per h,
pooled -- class (a), the h = 2 partial outcome confirmed, (b) not fired; free-scaling sabotage. Reads
data/derived/n1stage/{S1,S1b,S1c}, the alpha files of the new configurations and the free baselines (about 1-2 min)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage0, stage1
from masspairing.analysis import stage1b as B
from masspairing.analysis import stage1c as C1c
from masspairing.claimcheck import Check

c = Check("K5.8")
rows, res = B.analyse()
V, Vp = res["verdict_y3_new"], res["verdict_y3_pooled"]
fmt = B.fmt

# 1. floor (recorded) and the free gate
fl = V["floor_6x12_stored"]
c.item(
    "floor (recorded): stored 6^3x12 h = 0 light midpoint cosh mass >= 0.3 at 2 sigma",
    fmt(fl["m"], 3),
    fl["passes"] and abs(fl["m"][0] - 0.423) < 0.005,
)
g4 = V["S4_free_gate"]
c.item("free gate: GL_p0^free(h) = 1 to 1e-9 on 6^4 / 8^4 at h in {1, 1.5, 2, 3}", True, g4["ok_6_8"])
c.item(
    "free midpoint cosh-mass shift at 8^4 for h = 1, 2; at 6^4 (h = 2)",
    (g4["L8|1.0"]["dm_free"], g4["L8|2.0"]["dm_free"], g4["L6|2.0"]["dm_free"]),
    abs(g4["L8|1.0"]["dm_free"] - 0.037) < 0.003
    and abs(g4["L8|2.0"]["dm_free"] - 0.123) < 0.003
    and g4["L6|2.0"]["dm_free"] == 0.0,
    "{0[0]:.3f}, {0[1]:.3f}; {0[2]:.3f}",
)
gl12 = [g4[f"L6x12|{h}"]["GL_p0_free"] for h in (1.0, 1.5, 2.0, 3.0)]
c.item(
    "6^3x12 free GL_p0 (boundary modes; recorded)",
    (min(gl12), max(gl12)),
    all(0.99 < v < 1.0 for v in gl12),
    "{0[0]:.4f}-{0[1]:.4f}",
)

# 2./3. the clauses on the new samples
for h, d in V["rows"].items():
    c.record(
        f"h={h} (n 8^4 {d['n8']}, 6^4 {d['n6']})",
        f"r_L^cosh 8^4 {fmt(d['r_L8'])} 6^4 {fmt(d['r_L6'])}; g 8^4 {fmt(d['g8'])} 6^4 {fmt(d['g6'])}; e "
        f"{fmt(d['e'], 3)} e_free {d['e_free']:.3f}; "
        f"R_peak 8^4 {fmt(d['R_peak8'], 2)} 6^4 {fmt(d['R_peak6'], 2)}; chi_L/free 8^4 {fmt(d['chiLf8'])} 6^4 "
        f"{fmt(d['chiLf6'])}",
    )
d1, d2 = V["rows"]["1.0"], V["rows"]["2.0"]
c.item(
    "(a) r_L^cosh(8^4) at h = 1, 2 >= 0.7 - 2 sigma",
    (d1["r_L8"], d2["r_L8"]),
    d1["a_rL"]
    and d2["a_rL"]
    and abs(d1["r_L8"][0] - 0.976) < 0.004
    and abs(d2["r_L8"][0] - 0.920) < 0.004
    and d1["r_L8"][1] < 0.004,
    "{0[0][0]:.3f}({0[0][1]:.3f}), {0[1][0]:.3f}({0[1][1]:.3f})",
)
c.item(
    "6^4 reads the same: r_L^cosh at h = 1, 2",
    (d1["r_L6"], d2["r_L6"]),
    True,
    "{0[0][0]:.3f}({0[0][1]:.3f}), {0[1][0]:.3f}({0[1][1]:.3f})",
)
c.item(
    "(a) g = GL_p0/free <= 0.1 + 2 sigma at 8^4 (h = 1, 2); 6^4 values",
    (d1["g8"][0], d2["g8"][0], d1["g6"][0], d2["g6"][0]),
    d1["a_g"] and d2["a_g"] and d1["g8"][0] < 0.004 and d2["g8"][0] < 0.004 and d2["g6"][0] < 0.02,
    "{0[0]:.4f}, {0[1]:.4f}; {0[2]:.3f}, {0[3]:.3f}",
)
c.item(
    "class on the new samples: (a) at both h; not (b), not (c)",
    V["class"],
    V["class"] == "(a)" and not d1["b"] and not d2["b"] and d1["a"] and d2["a"] and not d1["c"] and not d2["c"],
)
c.item(
    "(b1) e - e_free - 2 sigma > 0.5 fails: e - e_free at h = 1, 2 and the clause margin",
    (d1["e_minus_free"], d2["e_minus_free"], d2["e_minus_free"][0] - 2 * d2["e_minus_free"][1]),
    not d1["b1"] and not d2["b1"] and d2["e_minus_free"][0] - 2 * d2["e_minus_free"][1] < 0.5,
    "{0[0][0]:.2f}({0[0][1]:.2f}), {0[1][0]:.3f}({0[1][1]:.3f}); {0[2]:.2f}",
)
c.item(
    "(b2) no p = 0 peak sharper than free: R_peak(8^4) at h = 1, 2",
    (d1["R_peak8"], d2["R_peak8"]),
    not d1["b2"] and not d2["b2"] and d2["R_peak8"][0] < 1.0,
    "{0[0][0]:.2f}({0[0][1]:.2f}), {0[1][0]:.2f}({0[1][1]:.2f})",
)
p3 = lambda d: (d["chiLf8"][0] - d["chiLf6"][0]) / np.hypot(d["chiLf8"][1], d["chiLf6"][1])
c.item(
    "(b3) chi_L/free 8^4 vs 6^4 at h = 1 (pull), h = 2 (pull): holds at h = 2 only",
    (d1["chiLf8"], d1["chiLf6"], p3(d1), d2["chiLf8"], d2["chiLf6"], p3(d2)),
    d2["b3"] and not d1["b3"],
    "{0[0][0]:.4f}({0[0][1]:.4f}) vs {0[1][0]:.4f}({0[1][1]:.4f}) {0[2]:.1f} sigma; {0[3][0]:.4f}({0[3][1]:.4f}) vs "
    "{0[4][0]:.4f}({0[4][1]:.4f}) {0[5]:.1f} sigma",
)
c.item(
    "h = 1: null (e - e_free within 3 sigma of 0); h = 2: partial outcome 'growth relative to free, no SSB-scale "
    "exponent' (significance)",
    d2["e_minus_free"][0] / d2["e_minus_free"][1],
    d1["null_e"] and not d1["partial_growth"] and d2["partial_growth"],
    "{:.1f} sigma",
)
c.item(
    "h = 2 at 8^4: condensate bound sqrt(chi_L/V) (free); heavy amplitude phi_R (free)",
    (d2["cond_bound8"][0], d2["cond_bound8"][1], d2["phi_R8"][0], d2["phi_R8_free"]),
    d2["chiLf8"][0] < 0.02 and d2["cond_bound8"][0] < 5e-4 and d2["phi_R8"][0] < 1e-3,
    "{0[0]:.1e} ({0[1]:.1e}); {0[2]:.1e} ({0[3]:.3f})",
)

# 4. pooled (labelled, not the verdict)
c.item(
    "pooled (stored + new; R3, L5 pooled not used): class, e - e_free at h = 1, 2",
    (Vp["class"], Vp["rows"]["1.0"]["e_minus_free"], Vp["rows"]["2.0"]["e_minus_free"]),
    Vp["class"] == "(a)" and V["pooled_usable"] == {"R1": True, "R3": False, "L5": False, "L6": True},
    "{0[0]}; {0[1][0]:.2f}({0[1][1]:.2f}), {0[2][0]:.3f}({0[2][1]:.3f})",
)

# 5. controls (live)
s2, s3, s6 = B.inject_s2(rows), B.inject_s3(rows), B.inject_free_null(rows)
d = s2["rows"]["2.0"]
c.item(
    "S2' injected SSB (chi_L(8^4, h = 2) x V8/V6, chi_L(p_min) x 0.5) -> (b) with all three clauses",
    s2["class"],
    s2["class"] == "(b)" and d["b1"] and d["b2"] and d["b3"],
)
c.item("S3' injected free return (g(2) := 1, m(2) := 0.25 m(0)) -> (c)", s3["class"], s3["class"] == "(c)")
c.item(
    "S6' free null (r_L = 1, chi_L = free, g = 1) -> not (a), not (b)",
    s6["verdict"]["class"],
    s6["verdict"]["class"] not in ("(a)", "(b)"),
)
c.item("the stage-0 reader's synthetic self-test", True, stage0.selftest())

# recorded beside the clauses
r8, r6 = B.R(rows, "L8", 3.0, 2.0, "new"), B.R(rows, "L6", 3.0, 2.0, "new")
r08, r06 = B.R(rows, "L8", 3.0, 0.0), B.R(rows, "L6", 3.0, 0.0)
c.item(
    "light midpoint cosh mass h = 0 -> 2 (new samples): 8^4, 6^4 (h-stable within 10 %)",
    (r8["m"][0] / r08["m"][0] - 1, r6["m"][0] / r06["m"][0] - 1),
    abs(r8["m"][0] / r08["m"][0] - 1) < 0.02 and abs(r6["m"][0] / r06["m"][0] - 1) < 0.10,
    "{0[0]:+.1%}, {0[1]:+.1%}",
)
f1 = B.R(rows, "L8", 3.0, 1.0, "new")["fit3"]
c.item(
    "3-point cosh fit: accepted at (8^4, h = 2) (m, chi^2/dof), rejected at h = 1 (chi^2/dof)",
    (r8["fit3"]["m"], r8["fit3"]["err"], r8["fit3"]["chi2"], f1["chi2"]),
    r8["fit3"]["accepted"] and not f1["accepted"],
    "{0[0]:.3f}({0[1]:.3f}) {0[2]:.1f}; {0[3]:.0f}",
)
for r in stage1.alpha_records("S1b"):
    c.record(
        f"alpha_L on the new configurations {r['L']}^3x{r['Lt']} y={r['y']} h={r['h']} (n {r['n']})",
        f"{r['rd_L']['alpha']:.3f}({r['rd_L']['alpha_err']:.3f}) [free {r['free']['rd_L']['alpha']:.2f}], "
        f"|O_L(p1)|/free {r['rd_L']['normO1'] / r['free']['rd_L']['normO1']:.3f}",
    )
# 6. stage 1c: two independent 8^4 replicas per h (notebook C204); the 6^4 side is the stage-1b new row
rows1b_1c, _, R1c = C1c.analyse()
S1c = R1c["A_scores"]
for r in ("rA", "rB"):
    v = S1c["per_replica"][r]["rows"]
    c.item(
        f"stage 1c replica {r}: class; e - e_free at h = 1, 2",
        (S1c["per_replica"][r]["class"], v["1.0"]["e_minus_free"], v["2.0"]["e_minus_free"]),
        S1c["per_replica"][r]["class"] == "(a)"
        and v["2.0"]["partial_growth"]
        and v["1.0"]["null_e"]
        and not v["1.0"]["b"]
        and not v["2.0"]["b"],
        "{0[0]}; {0[1][0]:.3f}({0[1][1]:.3f}), {0[2][0]:.3f}({0[2][1]:.3f})",
    )
P, V1c = S1c["pooled"]["rows"], R1c["A_verdict"]
p1, p2 = P["1.0"], P["2.0"]
c.item(
    "stage 1c pooled rA u rB: class (a); r_L^cosh(8^4) at h = 1, 2; g(8^4) at h = 1, 2",
    (p1["r_L8"], p2["r_L8"], p1["g8"][0], p2["g8"][0]),
    S1c["pooled"]["class"] == "(a)"
    and p1["a"]
    and p2["a"]
    and abs(p1["r_L8"][0] - 0.977) < 0.001
    and abs(p2["r_L8"][0] - 0.921) < 0.001
    and abs(p1["g8"][0] - 0.0030) < 5e-5
    and abs(p2["g8"][0] - 0.0034) < 5e-5,
    "{0[0][0]:.3f}({0[0][1]:.3f}), {0[1][0]:.3f}({0[1][1]:.3f}); {0[2]:.4f}, {0[3]:.4f}",
)
e2, s2_ = p2["e_minus_free"]
c.item(
    "stage 1c pooled, h = 2: e - e_free, its 2 sigma lower edge (> 0: the partial outcome CONFIRMED; < 0.5: (b1) "
    "does not fire), significance",
    (e2, s2_, e2 - 2 * s2_, e2 / s2_),
    V1c["2.0"]["line"] == "CONFIRMED"
    and abs(e2 - 0.600) < 0.002
    and abs(s2_ - 0.104) < 0.002
    and 0 < e2 - 2 * s2_ < 0.5
    and not p2["b1"]
    and p2["partial_growth"],
    "{0[0]:.3f}({0[1]:.3f}), {0[2]:+.2f}, {0[3]:.1f} sigma",
)
c.item(
    "stage 1c pooled, h = 2: (b) does not fire -- R_peak(8^4) < 1 (b2), b3 holds",
    p2["R_peak8"],
    not p2["b2"] and p2["R_peak8"][0] < 1 and abs(p2["R_peak8"][0] - 0.88) < 0.005 and p2["b3"] and not p2["b"],
    "{0[0]:.2f}({0[1]:.2f})",
)
c.item(
    "stage 1c pooled, h = 2: the light pair susceptibility chi_L/free at 8^4 (in per cent of free)",
    (p2["chiLf8"][0], p2["chiLf8"][1], 100 * p2["chiLf8"][0]),
    abs(p2["chiLf8"][0] - 0.0167) < 5e-4 and p2["chiLf8"][0] < 0.02,
    "{0[0]:.4f}({0[1]:.4f}) = {0[2]:.1f} % of free",
)
e1, s1_ = p1["e_minus_free"]
c.item(
    "stage 1c pooled, h = 1: e - e_free -> not resolved",
    (e1, s1_),
    V1c["1.0"]["line"] == "not resolved" and abs(e1 - 0.296) < 0.002 and abs(s1_ - 0.167) < 0.002 and p1["null_e"],
    "{0[0]:.3f}({0[1]:.3f})",
)
W = S1c["pooled_with_1b"]["rows"]
c.item(
    "stage 1c pooled with the stage-1b extensions (labelled): class (a); e - e_free at h = 2",
    (S1c["pooled_with_1b"]["class"], W["2.0"]["e_minus_free"]),
    S1c["pooled_with_1b"]["class"] == "(a)"
    and abs(W["2.0"]["e_minus_free"][0] - 0.551) < 0.002
    and W["2.0"]["partial_growth"]
    and not W["2.0"]["b"],
    "{0[0]}; {0[1][0]:.3f}({0[1][1]:.3f})",
)
ca = R1c["A_controls"]
c.item(
    "stage 1c controls on the pooled rows: S2' -> (b), S3' -> (c), S6' -> not (a)/(b)",
    (ca["S2'"]["cls"], ca["S3'"]["cls"], ca["S6'"]["cls"]),
    ca["S2'"]["ok"] and ca["S3'"]["ok"] and ca["S6'"]["ok"],
)
sb = C1c.sabotage_free_scaling(rows1b_1c, S1c)
c.item(
    "sabotage: pooled chi_L(8^4, h = 2) set to its free-scaling value -> e - e_free, NOT REPRODUCED",
    (sb["row"]["e_minus_free"], sb["line"]),
    sb["line"] == "NOT REPRODUCED" and sb["row"]["null_e"] and not sb["row"]["partial_growth"],
    "{0[0][0]:.3f}({0[0][1]:.3f}) -> {0[1]}",
)
c.done()
