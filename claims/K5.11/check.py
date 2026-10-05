#!/usr/bin/env python3
"""K5.11 -- stage 1c at y = 3.0 (SMG): the pre-registered K5 clauses on two independent 8^4 replicas per h (h = 1, 2),
each replica and their pool, with the stored h = 0 chains as denominators and the stage-1b 6^4 rows as the 6^4 side:
class (a) on every set; at h = 2 the light pair susceptibility grows faster with V than free (the partial outcome
"growth relative to free, no SSB-scale exponent", CONFIRMED), (b) not fired; at h = 1 the growth is not resolved.
Controls S2', S3', S6' on the pooled rows and the free-scaling sabotage. Reads data/derived/n1stage/{S1,S1b,S1c} and
the free baselines (about 10 s)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from masspairing.analysis import stage1c as C1c
from masspairing.claimcheck import Check

c = Check("K5.11")
rows, _, res = C1c.analyse()
S, V = res["A_scores"], res["A_verdict"]

for r in ("rA", "rB"):
    v = S["per_replica"][r]
    d1, d2 = v["rows"]["1.0"], v["rows"]["2.0"]
    c.item(
        f"replica {r}: class; r_L^cosh(8^4) at h = 1, 2; e - e_free at h = 1, 2",
        (v["class"], d1["r_L8"], d2["r_L8"], d1["e_minus_free"], d2["e_minus_free"]),
        v["class"] == "(a)" and d1["a"] and d2["a"] and not d1["b"] and not d2["b"],
        "{0[0]}; {0[1][0]:.3f}({0[1][1]:.3f}), {0[2][0]:.3f}({0[2][1]:.3f}); {0[3][0]:.3f}({0[3][1]:.3f}), "
        "{0[4][0]:.3f}({0[4][1]:.3f})",
    )
P = S["pooled"]["rows"]
p1, p2 = P["1.0"], P["2.0"]
c.item(
    "pooled rA u rB: class (a); r_L^cosh(8^4) at h = 1, 2",
    (S["pooled"]["class"], p1["r_L8"], p2["r_L8"]),
    S["pooled"]["class"] == "(a)"
    and abs(p1["r_L8"][0] - 0.977) < 0.001
    and abs(p2["r_L8"][0] - 0.921) < 0.001
    and p1["a"]
    and p2["a"],
    "{0[0]}; {0[1][0]:.3f}({0[1][1]:.3f}), {0[2][0]:.3f}({0[2][1]:.3f})",
)
c.item(
    "pooled: G_L(p_min)/free at 8^4, h = 1, 2 (clause (a): <= 0.1 + 2 sigma)",
    (p1["g8"], p2["g8"]),
    abs(p1["g8"][0] - 0.0030) < 5e-5 and abs(p2["g8"][0] - 0.0034) < 5e-5,
    "{0[0][0]:.4f}({0[0][1]:.4f}), {0[1][0]:.4f}({0[1][1]:.4f})",
)
e2, s2 = p2["e_minus_free"]
c.item(
    "pooled, h = 2: e - e_free on (6^4, 8^4), its 2 sigma lower edge (> 0: growth CONFIRMED; < 0.5: (b1) does not "
    "fire), significance",
    (e2, s2, e2 - 2 * s2, e2 / s2),
    V["2.0"]["line"] == "CONFIRMED" and abs(e2 - 0.600) < 0.002 and 0 < e2 - 2 * s2 < 0.5 and not p2["b1"],
    "{0[0]:.3f}({0[1]:.3f}), {0[2]:+.2f}, {0[3]:.1f} sigma",
)
c.item(
    "pooled, h = 2: R_peak(8^4) = [chi_L(0)/chi_L(p_min)]/free (b2 needs > 1 at 2 sigma)",
    p2["R_peak8"],
    not p2["b2"] and abs(p2["R_peak8"][0] - 0.880) < 0.002,
    "{0[0]:.3f}({0[1]:.3f})",
)
c.item(
    "pooled, h = 2: chi_L/free at 8^4 (in per cent of free); (b) as a whole not fired",
    (p2["chiLf8"][0], p2["chiLf8"][1], 100 * p2["chiLf8"][0]),
    abs(p2["chiLf8"][0] - 0.0167) < 5e-4 and not p2["b"],
    "{0[0]:.4f}({0[1]:.4f}) = {0[2]:.1f} % of free",
)
e1, s1 = p1["e_minus_free"]
c.item(
    "pooled, h = 1: e - e_free -> not resolved",
    (e1, s1),
    V["1.0"]["line"] == "not resolved" and abs(e1 - 0.296) < 0.002 and p1["null_e"],
    "{0[0]:.3f}({0[1]:.3f})",
)
W = S["pooled_with_1b"]
c.item(
    "pooled with the stage-1b extensions (labelled): class; e - e_free at h = 2",
    (W["class"], W["rows"]["2.0"]["e_minus_free"]),
    W["class"] == "(a)" and abs(W["rows"]["2.0"]["e_minus_free"][0] - 0.551) < 0.002,
    "{0[0]}; {0[1][0]:.3f}({0[1][1]:.3f})",
)
ca = res["A_controls"]
c.item(
    "controls on the pooled rows: S2' -> (b), S3' -> (c), S6' -> not (a)/(b)",
    (ca["S2'"]["cls"], ca["S3'"]["cls"], ca["S6'"]["cls"]),
    ca["S2'"]["ok"] and ca["S3'"]["ok"] and ca["S6'"]["ok"],
)
sb = C1c.sabotage_free_scaling(rows, S)
c.item(
    "sabotage: pooled chi_L(8^4, h = 2) set to its free-scaling value -> e - e_free, NOT REPRODUCED",
    (sb["row"]["e_minus_free"], sb["line"]),
    sb["line"] == "NOT REPRODUCED" and not sb["row"]["partial_growth"],
    "{0[0][0]:.3f}({0[0][1]:.3f}) -> {0[1]}",
)
c.done()
