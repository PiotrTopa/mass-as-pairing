#!/usr/bin/env python3
"""K5.7 -- stage 1b integrity: the ten stage-1b chains are complete, seam-continuous and kill-silent and are read on
new trajectories only; the stored-vs-new consistency S7'; the stored stage-1 rows re-scored with the stage-1b
estimators (E1) and the fit-window table (E8); the box-shape facts of the light Majorana pattern (E9).
Reads data/derived/n1stage/{S1,S1b,e9_box_shape.json} and the free baselines; recomputes the E9 group tables and free
values on the small boxes live (about 2 min)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1b as B
from masspairing.analysis.n1stage import N1DIR, free_lookup, free_tables
from masspairing.claimcheck import Check

c = Check("K5.7")
rows, res = B.analyse()
I, S7 = res["integrity"], res["S7"]

# 1. files
for tag, r in I.items():
    c.record(
        tag,
        f"{r['kind']} trajectories {r['ntraj']} new {r['n_new']} cadence {r['cadence']} acc>= "
        f"{r['acc_min50_new']:.2f} dup {r['dup']} "
        f"md5 {r['md5'][:8]} seam {r.get('seam_ok')} code {r['git']} A {r['A'][0]:+.4f}({r['A'][1]:.4f})",
    )
c.item("every chain file has the md5 recorded at delivery", True, all(r["md5_matches_delivered"] for r in I.values()))
c.item(
    "extensions: 2000 trajectories whose first 1000 equal the pre-extension copy (every stored series, "
    "configuration digests); new chains 1000",
    True,
    all(r["seam_ok"] for r in I.values() if r["kind"] == "ext")
    and all(r["ntraj"] == (2000 if r["kind"] == "ext" else 1000) for r in I.values()),
)
c.item(
    "no repeated trajectory index; code revision efcc72d on every chain",
    True,
    all(r["dup"] == 0 and r["git"] == "efcc72d" for r in I.values()),
)
exp = {"R1": 100, "R3": 100, "L5": 250, "L6": 250, "R2": 46, "D-1b-3": 45, "L1": 112, "L2": 112, "L3": 225, "L4": 225}
c.item("new samples after the cut", {t: I[t]["n_new"] for t in exp}, all(I[t]["n_new"] == n for t, n in exp.items()))
cad = {
    "R1": [10],
    "R3": [10],
    "L5": [4],
    "L6": [4],
    "D-1b-3": [20],
    "L1": [8],
    "L2": [8],
    "L3": [4],
    "L4": [4],
    "R2": [10, 20],
}
c.item(
    "cadences (R2: one 10-step at a restart, no duplicate)",
    {t: I[t]["cadence"] for t in cad},
    all(I[t]["cadence"] == v for t, v in cad.items()),
)

# 2. health, kills, errors
c.item(
    "acceptance per 50-trajectory window on every new stretch, min",
    min(r["acc_min50_new"] for r in I.values()),
    all(r["acc_ok"] and r["acc_min50_new"] >= 0.94 for r in I.values()),
    "{:.2f}",
)
c.item("kill A(h) < 0 beyond 3 sigma: silent on every chain", True, not any(r["kill_A_neg_3sigma"] for r in I.values()))
pc_t, smg_t = ("R2", "D-1b-3", "L1", "L2", "L3", "L4"), ("R1", "R3", "L5", "L6")
c.item(
    "A(h) significance: P_c chains (min, max), y = 3.0 chains (max |pull|, screened)",
    (min(I[t]["A_pull"] for t in pc_t), max(I[t]["A_pull"] for t in pc_t), max(abs(I[t]["A_pull"]) for t in smg_t)),
    all(I[t]["A_pull"] > 90 for t in pc_t) and all(abs(I[t]["A_pull"]) < 2 for t in smg_t),
    "{0[0]:.0f}-{0[1]:.0f}; {0[2]:.1f}",
)
mx = (
    max(r["tau_B_over_D"] for r in I.values()),
    max(r["infl_CL"] for r in I.values()),
    max(abs(v) for r in I.values() for v in r["half_pulls"].values()),
)
c.item(
    "tau_B/Delta, error inflation, half-split pulls of the primary series: max",
    mx,
    mx[0] <= 0.45 and mx[1] <= 1.3 and mx[2] <= 2.6,
    "{0[0]:.2f}, {0[1]:.2f}, {0[2]:.2f}",
)

# 3. S7'
for t, r in S7.items():
    if t != "all_ok":
        c.record(
            f"S7' {t}",
            f"pulls m {r['pulls']['m']:+.1f} chi_L {r['pulls']['chi_L']:+.1f} GL_p0 {r['pulls']['GL_p0']:+.1f}; m "
            f"new {B.fmt(r['new']['m'])} stored {B.fmt(r['stored']['m'])}",
        )
c.item(
    "S7': R1 and L6 consistent (3 sigma) on m, chi_L, GL_p0; chi_L and GL_p0 consistent on all four",
    True,
    S7["R1"]["ok"]
    and S7["L6"]["ok"]
    and all(abs(S7[t]["pulls"][k]) <= 3 for t in S7 if t != "all_ok" for k in ("chi_L", "GL_p0")),
)
c.item(
    "S7': light pair-channel midpoint cosh mass shifts of R3 (8^4, h = 1) and L5 (6^4, h = 2): stored -> new, pull, "
    "relative shift",
    tuple(
        (
            S7[t]["stored"]["m"],
            S7[t]["new"]["m"],
            S7[t]["pulls"]["m"],
            S7[t]["new"]["m"][0] / S7[t]["stored"]["m"][0] - 1,
        )
        for t in ("R3", "L5")
    ),
    abs(S7["R3"]["pulls"]["m"]) > 3
    and abs(S7["L5"]["pulls"]["m"]) > 3
    and abs(S7["R3"]["new"]["m"][0] / S7["R3"]["stored"]["m"][0] - 1) < 0.012
    and abs(S7["L5"]["new"]["m"][0] / S7["L5"]["stored"]["m"][0] - 1) < 0.015,
    "R3 {0[0][0][0]:.4f}({0[0][0][1]:.4f}) -> {0[0][1][0]:.4f}({0[0][1][1]:.4f}) {0[0][2]:+.1f} sigma {0[0][3]:+.1%}; "
    "L5 {0[1][0][0]:.4f}({0[1][0][1]:.4f}) -> {0[1][1][0]:.4f}({0[1][1][1]:.4f}) {0[1][2]:+.1f} sigma {0[1][3]:+.1%}",
)
c.item(
    "pooled values usable (R1, R3, L5, L6)",
    res["verdict_y3_new"]["pooled_usable"],
    res["verdict_y3_new"]["pooled_usable"] == {"R1": True, "R3": False, "L5": False, "L6": True},
)

# 4. E1 / E8
E1 = {(d["lat"], d["y"], d["h"]): d for d in B.e1_table(rows)}
m8 = [E1[("L8", 2.41, h)]["m"] for h in (0.0, 0.5, 1.0, 2.0)]
c.item(
    "E1: 8^4 P_c light pair-channel midpoint cosh mass at h = 0, 0.5, 1, 2",
    [B.fmt(m, 3) for m in m8],
    abs(m8[0][0] - 0.197) < 0.003 and abs(m8[3][0] - 0.614) < 0.003 and all(m8[i + 1][0] > m8[i][0] for i in range(3)),
)
st = {lat: E1[(lat, 3.0, 2.0)]["m"][0] / E1[(lat, 3.0, 0.0)]["m"][0] - 1 for lat in ("L8", "L6", "L6x12")}
c.item(
    "E1: y = 3.0 midpoint cosh mass h = 0 -> 2 (8^4, 6^4, 6^3x12), within 10 %",
    st,
    all(abs(v) < 0.10 for v in st.values()),
)
ef = [E1[("L8", 3.0, h)]["e_free"] for h in (0.0, 0.5, 1.0, 2.0)]
c.item("E1: e_free on (6^4, 8^4) at h = 0, 0.5, 1, 2", np.round(ef, 3).tolist(), 0.38 < min(ef) and max(ef) < 0.46)
E8 = {(d["lat"], d["y"], d["h"], d["sel"]): d for d in B.e8_table(rows)}
r = E8[("L6x12", 3.0, 0.0, "stored")]
mc = [r["mcosh_t"][str(t)][0] for t in range(6)]
c.item(
    "E8: 6^3x12 (3.0, h = 0) cosh effective mass at t = 1..5 (falling, no plateau)",
    np.round(mc[1:], 2).tolist(),
    mc[1] > 0.85 and mc[5] < 0.45 and all(mc[i + 1] < mc[i] for i in range(1, 5)),
)
bad = [w for w in r["windows"] if w["chi2_dof"] > 3]
c.item(
    "E8: single-cosh windows with chi^2/dof > 3 at (3.0, h = 0) on 6^3x12",
    (len(bad), len(r["windows"])),
    len(bad) >= len(r["windows"]) - 2,
    "{0[0]} of {0[1]}",
)
c.item(
    "E8: the free 6^3x12 light correlator has no cosh solution at the midpoint (raw r_L there)",
    True,
    not np.isfinite(B.R(rows, "L6x12", 3.0, 0.0)["free"]["m"]),
)

# 5. E9 box-shape facts
e9 = json.loads((N1DIR / "e9_box_shape.json").read_text())
for shape in ((4, 4, 4, 4), (4, 4, 4, 8)):
    t = B.element_table(shape)
    fz = e9["elements"]["x".join(map(str, shape))]
    c.item(
        f"E9 {shape}: |G|, stabiliser of C_chi, keep / flip C_asd(0,3), all flips move the time axis, mean action "
        "(live = frozen)",
        (t["G"], t["stab"], t["n_keep"], t["n_flip"], t["all_flips_move_time"], t["mean_action"]),
        all(t[k] == fz[k] for k in ("G", "stab", "n_keep", "n_flip", "all_flips_move_time"))
        and abs(t["mean_action"] - fz["mean_action"]) < 1e-12,
    )
t4 = {k: e9["elements"]["4x4x4x4"][k] for k in ("stab", "n_keep", "n_flip", "all_flips_move_time")}
t48 = e9["elements"]["4x4x4x8"]
c.item(
    "E9: on 4^4 the stabiliser (32) splits 16 / 16, every flipping element exchanges a spatial axis with time",
    t4,
    t4 == dict(stab=32, n_keep=16, n_flip=16, all_flips_move_time=True),
)
c.item(
    "E9: on 4^3x8 the box stabiliser (8) keeps C_asd(0,3) (mean action)",
    t48["mean_action"],
    t48["stab"] == 8 and t48["n_flip"] == 0 and abs(t48["mean_action"] - 1) < 1e-9,
    "{:+.3f}",
)
base = {
    (r["L"], r["Lt"], r["bc"], float(r["h"])): r["phi_L"][0]
    for r in free_tables(("free_L6x12_aaaa.json", "free_L4x8_aaaa.json", "free_baselines.json"))
}
live = {
    (6, 12): B.free_phiL(6, 12, 2.0),
    (4, 8): B.free_phiL(4, 8, 2.0),
    (4, 4): B.free_phiL(4, 4, 2.0),
    (6, 6): B.free_phiL(6, 6, 2.0),
}
c.item(
    "E9 free <phi_L> per flavour at h = 2 (live): 6^3x12, 4^3x8 = the free baselines; 4^4, 6^4 zero",
    (live[(6, 12)], live[(4, 8)], live[(4, 4)], live[(6, 6)]),
    abs(live[(6, 12)] - base[(6, 12, "aaaa", 2.0)]) < 1e-12
    and abs(live[(4, 8)] - base[(4, 8, "aaaa", 2.0)]) < 1e-12
    and abs(live[(4, 4)]) < 1e-12
    and abs(live[(6, 6)]) < 1e-12,
    "{0[0]:+.6f}, {0[1]:+.6f}, {0[2]:.0e}, {0[3]:.0e}",
)
asp = {d["Lt"]: d["phi_L"] for d in e9["aspect_series_L6"]}
ls = {(d["L"], d["h"]): d["phi_L"] for d in e9["L_series_aspect2"]}
hc = {d["L"]: d["phi_L"] for d in e9["hypercubic"]}
c.item(
    "E9 (frozen scan): free <phi_L> zero on 4^4 / 6^4 / 8^4 (max |value|)",
    max(abs(v) for v in hc.values()),
    all(abs(v) < 1e-12 for v in hc.values()),
    "{:.0e}",
)
c.item(
    "E9 (frozen scan): 6^3 x L_t at h = 2 for L_t = 12, 24 (set by the spatial box)",
    (asp[12], asp[24]),
    abs(asp[12] + 0.000846) < 2e-6 and abs(asp[24] + 0.00084) < 3e-5,
    "{0[0]:+.6f}, {0[1]:+.6f}",
)
c.item(
    "E9 (frozen scan): falls with L within each L mod 4 class at aspect 2, h = 2: 4^3x8 -> 8^3x16; 6^3x12 -> 10^3x20",
    (ls[(4, 2.0)], ls[(8, 2.0)], ls[(6, 2.0)], ls[(10, 2.0)]),
    abs(ls[(8, 2.0)]) < abs(ls[(4, 2.0)]) and abs(ls[(10, 2.0)]) < abs(ls[(6, 2.0)]),
    "{0[0]:+.6f} -> {0[1]:+.6f}; {0[2]:+.6f} -> {0[3]:+.5f}",
)
heavy = free_lookup(free_tables(("free_L6x12_aaaa.json",)), 6, 12, "aaaa", 2.0)
c.record("6^3x12 free heavy amplitude per flavour at h = 2 (scale)", f"{heavy['phi_f'][0]:+.5f}")
tab = []
for (lat, y, h, sel), r in rows.items():
    if sel == "pooled" or h == 0:
        continue
    v, e = r["phi_L"]
    fr = r["free"]["phi_L"]
    tab.append(dict(lat=lat, y=y, h=h, sel=sel, pull=v / e, over=v / fr if abs(fr) > 1e-10 else None))
cube = [d for d in tab if d["lat"] in ("L8", "L6")]
c.item(
    "E9 chains: <phi_L> on 6^4 / 8^4 zero within errors at every (y, h): max |pull|",
    max(abs(d["pull"]) for d in cube),
    max(abs(d["pull"]) for d in cube) < 2.5,
    "{:.1f}",
)
p12 = sorted([d for d in tab if d["lat"] == "L6x12" and d["y"] == 2.41], key=lambda d: d["h"])
c.item(
    "E9 chains: 6^3x12 P_c <phi_L>/free at h = 0.5, 1, 1.5, 2, 3 (nonzero at > 5 sigma, below free)",
    [round(d["over"], 2) for d in p12],
    len(p12) == 5 and all(0.55 < d["over"] < 0.95 and d["pull"] < -5 for d in p12),
)
s12 = sorted([d for d in tab if d["lat"] == "L6x12" and d["y"] == 3.0], key=lambda d: d["h"])
c.item(
    "E9 chains: 6^3x12 y = 3.0 <phi_L>/free at h = 0.5, 1, 2 (screened)",
    [round(d["over"], 3) for d in s12],
    all(abs(d["over"]) < 0.03 for d in s12),
)
c.done()
