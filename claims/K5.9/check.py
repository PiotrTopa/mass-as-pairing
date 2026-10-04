#!/usr/bin/env python3
"""K5.9 -- stage 1c integrity and replica consistency: the nine stage-1c chains are intact; at y = 3.0 on 8^4 the two
independent replicas per h agree with each other and with the stage-1b extension (3-member chi^2, every p > 0.01); the
drift recorded on the 8^4 h = 1 extension (K5.7) lies in its stage-1 stretch; the 6^4 P_c replicas at h = 0.25, 0.5
agree with the first chains; no half-split flag; the injected 3 sigma shift is flagged, and so is a 3 sigma shift
sabotaging a 6^4 replica. Reads data/derived/n1stage/{S1,S1b,S1c} and the free baselines (a few seconds)."""

import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1c as C
from masspairing.claimcheck import Check

c = Check("K5.9")
rows1b, tagrow, res = C.analyse()
fmt = C.fmt

# 1. integrity
I = res["integrity"]
for tag, r in I.items():
    if tag == "all_ok":
        continue
    c.record(
        f"{tag}: {r['file'].split('K_N1_S1c/')[1]}",
        f"md5 {r['md5'][:8]} seed {r['seed']} h {r['h']:g} trajectories {r['ntraj']} rows >= 100 {r['n_cut']} "
        f"duplicates {r['dup']} min acceptance (50-window) {r['acc_min50']:.2f} A {fmt(r['A'])} "
        f"tau_B/Delta {r['tau_B_over_D']:.2f} CL inflation {r['infl_CL']:.2f} code {r['git']}",
    )
tags8 = [t for t in C.CH_A]
tags6 = [t for t in C.CH_B]
acc8 = [I[t]["acc_min50"] for t in tags8]
acc6 = [I[t]["acc_min50"] for t in tags6]
c.item(
    "nine chains: md5 as delivered, seeds and h as ordered, 1000 trajectories, no duplicate trajectory index, 90 (8^4) "
    "/ 225 (6^4) rows after the cut, acceptance >= 0.5, A(h) never below -3 sigma, code commit as stored "
    "(8^4: 8304fbd, 6^4: 5ab6fbe; same physics code); min acceptance 8^4, 6^4",
    (min(acc8), max(acc8), min(acc6), max(acc6)),
    I["all_ok"] and {I[t]["git"] for t in tags8} == {"8304fbd"} and {I[t]["git"] for t in tags6} == {"5ab6fbe"},
    "{0[0]:.2f}-{0[1]:.2f}, {0[2]:.2f}-{0[3]:.2f}",
)
infl8 = [I[t]["infl_CL"] for t in tags8]
infl6 = [I[t]["infl_CL"] for t in tags6]
c.item(
    "error model: CL-midpoint inflation on 8^4, on 6^4; tau_B/Delta on 8^4",
    (min(infl8), max(infl8), min(infl6), max(infl6), min(I[t]["tau_B_over_D"] for t in tags8)),
    max(infl8) < 1.5 and max(infl6) < 2.0,
    "{0[0]:.2f}-{0[1]:.2f}, {0[2]:.2f}-{0[3]:.2f}; >= {0[4]:.2f}",
)
c.item(
    "the one-member pool reproduces the single-chain reader (pooling implementation)",
    res["A_scores"]["pool_identity"]["ok"],
    res["A_scores"]["pool_identity"]["ok"],
)

# 2. consistency at y = 3.0 (8^4)
lab = {"m": "m_cosh(mid)", "chi_L": "<chi_L_sum>", "GL_p0": "<GL_p0>"}
for h in ("2.0", "1.0"):
    cc = res["A_consistency"][h]
    for O, v in cc["obs"].items():
        val = v["values"]
        c.record(
            f"h = {float(h):g} {lab[O]}",
            f"rA {fmt(val['rA'], 5)} rB {fmt(val['rB'], 5)} stage 1b ({cc['member_1b']}) {fmt(val['one_b_new'], 5)}; "
            f"p 3-member {v['three']['p']:.2f}, rA vs rB {v['rA_vs_rB']['p']:.2f}, pool vs stage 1b "
            f"{v['pooled_vs_1b']['p']:.2f}",
        )
    c.item(
        f"h = {float(h):g}: CONSISTENT -- 3-member, rA vs rB and pool vs stage 1b all p > 0.01; smallest 3-member p",
        min(v["three"]["p"] for v in cc["obs"].values()),
        cc["consistent"] and cc["rA_vs_rB_ok"] and all(v["pooled_vs_1b"]["ok"] for v in cc["obs"].values()),
        "{:.2f}",
    )
obs = [v for h in ("2.0", "1.0") for v in res["A_consistency"][h]["obs"].values()]
c.item(
    "smallest p over both h and the three observables: 3-member; rA vs rB; pool vs stage 1b",
    (min(v["three"]["p"] for v in obs), min(v["rA_vs_rB"]["p"] for v in obs), min(v["pooled_vs_1b"]["p"] for v in obs)),
    min(v["three"]["p"] for v in obs) > 0.1,
    "{0[0]:.2f}; {0[1]:.2f}; {0[2]:.2f}",
)
v1 = res["A_consistency"]["1.0"]["obs"]["m"]
others = [v1["values"][k][0] for k in ("rA", "rB", "one_b_new")]
c.item(
    "located drift (labelled, not a criterion): with the stage-1 stretch (trajectories 100-999) of the 8^4 h = 1 "
    "chain as a fourth member, the midpoint cosh mass fails (p); that stretch sits above the three later or "
    "independent members",
    (v1["four_with_stored"]["p"], fmt(v1["values"]["stored"], 5), min(others), max(others)),
    v1["four_with_stored"]["p"] < 0.01 and v1["values"]["stored"][0] > max(others),
    "p = {0[0]:.3f}; {0[1]} vs {0[2]:.4f}-{0[3]:.4f}",
)
v2 = res["A_consistency"]["2.0"]["obs"]
c.item(
    "the same four-member test at h = 2 passes (smallest p)",
    min(v["four_with_stored"]["p"] for v in v2.values()),
    all(v["four_with_stored"]["ok"] for v in v2.values()),
    "{:.2f}",
)

# 3. half-splits
hs = {**res["A_half_split"], **res["B_half_split"]}
mx = max((abs(p), k, o) for k, v in hs.items() for o, p in v["pulls"].items())
c.item(
    "half-split pulls (m, chi_L, GL_p0) on the nine chains: none above 3 (largest, chain, observable)",
    mx,
    not any(v["flag"] for v in hs.values()),
    "{0[0]:.1f} ({0[1]}, {0[2]})",
)

# 4. control
cc = res["A_controls"]["C2_control"]
pmax = max(max(d.values()) for d in cc["p_three"].values())
pmax_ab = max(max(d.values()) for d in cc["p_ab"].values())
c.item(
    "control: a 3 sigma_c shift (sigma_c = sqrt(sigma_rA^2 + sigma_rB^2)) into rB, away from the mean of the other two "
    "members -> largest p in the 3-member test, in rA vs rB (every observable, both h)",
    (pmax, pmax_ab),
    cc["ok"] and pmax <= 0.01,
    "{0[0]:.0e}, {0[1]:.0e}",
)

# 5. 6^4 replicas
for h, cB in res["B"]["consistency"].items():
    c.record(
        f"6^4 P_c h = {float(h):g} ({cB['pair'][0]} vs {cB['pair'][1]})",
        "; ".join(f"{lab[O]} {fmt(v['orig'], 5)} vs {fmt(v['r2'], 5)} p = {v['p']:.2f}" for O, v in cB["obs"].items()),
    )
pB = min(v["p"] for cB in res["B"]["consistency"].values() for v in cB["obs"].values())
c.item(
    "6^4 P_c replica test (dof 1) at h = 0.25, 0.5: CONSISTENT, the pooled points enter the onset fit (smallest p)",
    pB,
    all(cB["consistent"] for cB in res["B"]["consistency"].values())
    and res["B"]["used_pooled"] == {0.25: True, 0.5: True},
    "{:.2f}",
)
# 6. sabotage of the 6^4 replica test: the replica's midpoint cosh mass shifted by 3 sigma_c (sigma_c = sqrt(sigma^2 +
# sigma_r2^2)) away from the first chain must be flagged (p <= 0.01) at both h
sab = copy.deepcopy(tagrow)
for a, b in (("B2", "B2r"), ("B1", "B1r")):
    x, y = sab[a]["m"], sab[b]["m"]
    sab[b]["m"] = (y[0] + float(np.sign(y[0] - x[0]) or 1.0) * 3 * float(np.hypot(x[1], y[1])), y[1])
cs = C.consistency_B(sab)
psab = max(cs[h]["obs"]["m"]["p"] for h in cs)
c.item(
    "sabotage: the 6^4 replica's midpoint cosh mass shifted by 3 sigma_c away from the first chain -> flagged at "
    "h = 0.25 and 0.5 (largest p)",
    psab,
    psab <= 0.01 and not any(cs[h]["consistent"] for h in cs),
    "{:.1e}",
)
c.done()
