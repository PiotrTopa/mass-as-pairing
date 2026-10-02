#!/usr/bin/env python3
"""M.1 -- the moonshot trigger is not met at L <= 8: at P_c (y = 2.41) the light pair-channel effective mass rises with
the partner's Majorana mass h on every lattice (the opposite of a critical seesaw); the pre-registered M trigger of
stage 1 does not fire, its control has power, and the stage-1b fall alert stays silent.
Reads data/derived/n1stage/{S1,S1b} and the free baselines (about 2 min)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis import stage1b as B
from masspairing.claimcheck import Check

c = Check("M.1")
rows = stage1.stage1_rows()
M = stage1.m_trigger(rows)
for L in ("L6", "L8"):
    m = M[L]
    c.record(f"{L}: r_L (t* log-ratio / free) at h = 0.5, 1, 2", ", ".join(f"{v[0]:.2f}({v[1]:.2f})" for v in m["r_L"]))
rising = all(
    v[i + 1][0] > v[i][0] + 2 * np.hypot(v[i][1], v[i + 1][1])
    for v in (M["L6"]["r_L"], M["L8"]["r_L"])
    for i in range(2)
)
c.item(
    "r_L rises monotonically in 2 sigma steps on 6^4 and 8^4 (no fall)",
    True,
    rising and not M["L6"]["fall"] and not M["L8"]["fall"],
)
c.item(
    "r_L(2) + 2 sigma < 0.7 fails on 6^4 and 8^4",
    (M["L6"]["r_L"][-1][0], M["L8"]["r_L"][-1][0]),
    not M["L6"]["last_below"] and not M["L8"]["last_below"],
    "{0[0]:.2f}, {0[1]:.2f}",
)
c.item(
    "log-slope e_r on the last two h: 6^4 vs 8^4 (not V-stable)",
    (M["L6"]["e_r"], M["L8"]["e_r"], M["e_r_Vstable"]),
    not M["e_r_Vstable"],
    "{0[0][0]:+.2f}({0[0][1]:.2f}) vs {0[1][0]:+.2f}({0[1][1]:.2f})",
)
c.item("absence clause at y = 3.0 holds: r_L(2) on 8^4", M["rL2_smg_L8"], M["absent_at_smg"], "{0[0]:.3f}({0[1]:.3f})")
r12 = [stage1.get(rows, 2.41, 6, 12, h)["r_L"][0] for h in (0.5, 1.0, 2.0)]
raw = [stage1.get(rows, 2.41, 6, 12, h)["mL"][0] / stage1.get(rows, 2.41, 6, 12, 0.0)["mL"][0] for h in (0.5, 1.0, 2.0)]
c.item(
    "6^3x12 at P_c: r_L as written (free normalisation void there) and the raw t* ratio, h = 0.5, 1, 2",
    (r12, raw),
    all(r12[i + 1] > r12[i] for i in range(2)) and all(raw[i + 1] > raw[i] for i in range(2)),
    "{}",
)
c.item("M trigger", M["triggered"], not M["triggered"])
mc = {}
for L, Lt, key in ((8, 8, "mcosh_L_t3"), (6, 6, "mcosh_L_t2"), (6, 12, "mcosh_L_t5")):
    mc[(L, Lt)] = [stage1.get(rows, 2.41, L, Lt, h)[key] for h in (0.0, 2.0)]
c.item(
    "light pair-channel midpoint cosh mass at P_c, h = 0 -> 2: 8^4, 6^4, 6^3x12",
    tuple(v for k in ((8, 8), (6, 6), (6, 12)) for v in mc[k]),
    all(mc[k][1][0] > mc[k][0][0] for k in mc),
    "{0[0][0]:.3f}({0[0][1]:.3f}) -> {0[1][0]:.3f}({0[1][1]:.3f}); {0[2][0]:.3f}({0[2][1]:.3f}) -> "
    "{0[3][0]:.3f}({0[3][1]:.3f}); {0[4][0]:.3f}({0[4][1]:.3f}) -> {0[5][0]:.3f}({0[5][1]:.3f})",
)
c.item(
    "at h = 2 the 8^4 and 6^4 midpoint values agree within 10 % while the h = 0 value halves from 6^4 to 8^4",
    (mc[(8, 8)][1][0] / mc[(6, 6)][1][0], mc[(8, 8)][0][0] / mc[(6, 6)][0][0]),
    abs(mc[(8, 8)][1][0] / mc[(6, 6)][1][0] - 1) < 0.1 and mc[(8, 8)][0][0] / mc[(6, 6)][0][0] < 0.6,
    "{0[0]:.2f}; {0[1]:.2f}",
)
g = [stage1.get(rows, 2.41, 8, 8, h)["g"][0] for h in (0.0, 0.5, 1.0, 2.0)]
c.item("G_L(p_min)/free at 8^4 P_c rises with h", np.round(g, 3).tolist(), all(g[i + 1] > g[i] for i in range(3)))
ctl = stage1.controls(rows)
c.item(
    "control S1_M (r_L := 1/(1 + 3h) at y = 2.41 only) -> M triggered (the pipeline has power)",
    ctl["S1_M"]["M"]["triggered"],
    ctl["S1_M"]["M"]["triggered"],
)
c.item(
    "control: free-null injection -> M not triggered",
    ctl["null_free"]["M"]["triggered"],
    not ctl["null_free"]["M"]["triggered"],
)

# stage 1b: the fall alert of the anti-seesaw order (a fall from h = 1 to 1.5 at > 3 sigma on 6^3x12 and 6^4)
rows_b, res = B.analyse()
t1 = res["anti_seesaw"]["T1"]
c.item(
    "stage 1b fall alert (h = 1 -> 1.5 fall significance on 6^3x12, 6^4) is silent",
    t1["C4_alert"]["pulls_fall_1_to_1_5"],
    not t1["C4_alert"]["fires"],
)
s1 = B.inject_s1(rows_b)
c.item(
    "control S1' (injected seesaw on the new rows) fires the fall alert",
    s1["C4_alert"]["fires"],
    s1["C4_alert"]["fires"],
)
c.done()
