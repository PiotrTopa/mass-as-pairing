#!/usr/bin/env python3
"""K5.4 -- stage 1: data integrity, the thinned cadence, the order's kills (silent), the recorded instrument checks,
the error model and the free baselines. Reads data/derived/n1stage/S1 (26 chains) and the free baselines (about 1 min).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis.n1stage import N1DIR, free_lookup, free_tables, open_chain
from masspairing.claimcheck import Check

c = Check("K5.4")
rows = stage1.stage1_rows()
bl = free_tables(stage1.FREE_S1)

# 1. integrity
c.item("chains", len(rows), len(rows) == 26)
c.item(
    "code revision 29f33bd, pattern chiral (0,3) dual sign +1 on every chain",
    True,
    all(
        r["git_hash"].startswith("29f33bd")
        and r["pattern"] == "chiral"
        and list(r["comp"]) == [0, 3]
        and r["dual_sign"] == 1
        for r in rows
    ),
)
c.item(
    "aaaa / pppa chains",
    (sum(r["bc"] == "aaaa" for r in rows), sum(r["bc"] == "pppa" for r in rows)),
    sum(r["bc"] == "aaaa" for r in rows) == 24 and sum(r["bc"] == "pppa" for r in rows) == 2,
)
cad = {(r["L"], r["Lt"]): (r["cadence"], r["n"]) for r in rows}
c.item(
    "cadence (trajectories) and measurements after the cut: 8^4, 6^3x12, 6^4",
    (cad[(8, 8)], cad[(6, 12)], cad[(6, 6)]),
    cad[(8, 8)] == ([20], 45)
    and cad[(6, 12)] == ([16], 56)
    and cad[(6, 6)] == ([4], 225)
    and all(r["n"] == {8: 45, 12: 56, 6: 225}[r["Lt"]] for r in rows),
)
c.item(
    "repeated trajectory indices on any chain",
    sum(len(r["duplicates"]) for r in rows),
    all(len(r["duplicates"]) == 0 for r in rows),
)
d, _ = open_chain(N1DIR / "S1" / "L6" / "L6_y3_k-0.01_g0_g60_chiral_bcaaaa.npz")
tr, dig = d["ts_cfg_traj"], d["cfg_digest"]
c.item(
    "6^4 (3.0, h = 0): 251 rows, all trajectory indices and stored configurations distinct; cadence around the resume",
    list(tr[11:15]),
    len(tr) == 251 and len(np.unique(tr)) == 251 and len(np.unique(dig)) == 251 and list(tr[11:15]) == [44, 48, 50, 54],
)
acc = (min(r["acc_min50"] for r in rows), min(r["acc_min50"] for r in rows if r["bc"] == "aaaa"))
c.item(
    "acceptance per 50-trajectory window, min (all / aaaa)",
    acc,
    acc[0] >= 0.60 and acc[1] >= 0.84,
    "{0[0]:.2f} / {0[1]:.2f}",
)

# 2. the order's kills
h0 = [r for r in rows if r["h"] == 0]
c.item(
    "h = 0: CL_t = CR_t within 4 sigma at every t (max pull) and |m_L - m_R| (max pull), 7 chains",
    (max(r["h0_check"]["max_pull"] for r in h0), max(r["h0_check"]["dm_pull"] for r in h0)),
    len(h0) == 7 and max(r["h0_check"]["max_pull"] for r in h0) < 4 and max(r["h0_check"]["dm_pull"] for r in h0) < 4,
    "{0[0]:.2f}, {0[1]:.2f}",
)
hp = [r for r in rows if r["h"] > 0]
c.item(
    "A(h) < 0 beyond 3 sigma nowhere: min pull",
    min(r["A_pull"] for r in hp),
    min(r["A_pull"] for r in hp) > -3,
    "{:+.1f}",
)

# 3. recorded instrument checks
pc = [r for r in hp if r["y"] == 2.41 and r["bc"] == "aaaa"]
smg = [r for r in hp if r["y"] == 3.0]
c.item(
    "y = 2.41: phi_R/h > 0 on all 9 rows, min significance",
    min(r["phi_R_pull"] for r in pc),
    len(pc) == 9 and all(r["phi_R_pull"] > 5 for r in pc),
    "{:.0f}",
)
c.item(
    "y = 2.41, h >= 1: A > 0, min significance",
    min(r["A_pull"] for r in pc if r["h"] >= 1),
    all(r["A_pull"] > 3 for r in pc if r["h"] >= 1),
    "{:.1f}",
)
a05 = {(r["L"], r["Lt"]): r["A_pull"] for r in pc if r["h"] == 0.5}
c.item(
    "y = 2.41, h = 0.5: A significance on 6^3x12 (passes 3 sigma), 8^4 and 6^4 (fail the 3 sigma check as written)",
    (a05[(6, 12)], a05[(8, 8)], a05[(6, 6)]),
    a05[(6, 12)] > 3 and a05[(8, 8)] < 3 and abs(a05[(6, 6)] - 3.0) < 0.3,
    "{0[0]:.1f}, {0[1]:.1f}, {0[2]:.1f}",
)
fA = [free_lookup(bl, L, Lt, "aaaa", 0.5) for L, Lt in ((8, 8), (6, 6), (6, 12))]
fA = [(np.sum(f["CL_t"]) - np.sum(f["CR_t"])) / np.sum(f["CL_t"]) for f in fA]
c.record("free A(0.5) on 8^4, 6^4, 6^3x12", ", ".join(f"{a:.3f}" for a in fA))
c.item(
    "y = 3.0: phi_T_R/h nonzero (min significance) and phi_R/h >= 0 within 2 sigma on all 9 rows",
    min(abs(r["phi_T_R_pull"]) for r in smg),
    len(smg) == 9 and all(abs(r["phi_T_R_pull"]) > 5 for r in smg) and all(r["phi_R_pull"] > -2 for r in smg),
    "{:.0f}",
)
c.item(
    "y = 3.0: max phi_R/h over free, max |A| (6^4, 6^3x12, 8^4)",
    (max(abs(r["phi_R_over_free"]) for r in smg), max(abs(r["A"][0]) for r in smg)),
    all(abs(r["phi_R_over_free"]) <= 0.012 and abs(r["A"][0]) <= 0.01 for r in smg),
    "{0[0]:.3f}, {0[1]:.4f}",
)

# 4. error model
taus = [v["tau_x"] for r in rows for v in r["stats"].values()]
c.item("tau_int of every primary series (measurements), max", max(taus), max(taus) < 3.0, "{:.1f}")
pp0 = next(r for r in rows if r["bc"] == "pppa" and r["h"] == 0)
c.item(
    "tau_B/Delta: max on aaaa; 6^4 pppa h = 0 (tau_B in trajectories)",
    (max(r["tau_B_over_D"] for r in rows if r["bc"] == "aaaa"), pp0["tau_B_over_D"], pp0["tau_B"]),
    max(r["tau_B_over_D"] for r in rows if r["bc"] == "aaaa") < 1.2 and 5 < pp0["tau_B_over_D"] < 7,
    "{0[0]:.2f}; {0[1]:.1f} ({0[2]:.0f})",
)
infl = [
    v["infl"]
    for r in rows
    if r["bc"] == "aaaa"
    for k, v in r["stats"].items()
    if k in ("mL", "chi_L", "GL_p0", "phi_R", "phi_T_R")
]
c.item("error inflation sigma/sigma_block on aaaa chains (primary series), max", max(infl), max(infl) < 2.0, "{:.2f}")
flags = [(r["bc"], r["h"]) for r in rows for v in r["stats"].values() if abs(v["half_pull"]) > 3]
npp = sum(b == "pppa" and h == 0 for b, h in flags)
c.item(
    "half-split pulls > 3 sigma: series flagged of all, of them on the pppa h = 0 chain",
    (len(flags), sum(len(r["stats"]) for r in rows), npp),
    len(flags) <= 12 and npp >= 5,
)

# 5. free baselines
for L, Lt, bc, tol in ((6, 6, "aaaa", 1e-10), (8, 8, "aaaa", 1e-10), (6, 12, "aaaa", 3e-3), (6, 6, "pppa", 4e-3)):
    g = [abs(free_lookup(bl, L, Lt, bc, h)["GL_p0"] - 1) for h in (0.0, 0.5, 1.0, 2.0) if free_lookup(bl, L, Lt, bc, h)]
    c.item(f"{L}^3x{Lt} {bc}: max |GL_p0^free - 1|", max(g), max(g) < tol, "{:.1e}")
f6 = [free_lookup(bl, 6, 12, "aaaa", h)["mL_eff"][5] for h in (0.0, 0.5, 1.0, 2.0)]
c.item(
    "6^3x12: free light log-ratio at t* = 5 (no decaying midpoint) at h = 0, 0.5, 1, 2",
    f6,
    all(v < 0 for v in f6),
    "{}",
)
sh = [
    abs(
        free_lookup(bl, L, L, "aaaa", h)["mL_eff"][L // 2 - 1]
        - free_lookup(bl, L, L, "aaaa", 0.0)["mL_eff"][L // 2 - 1]
    )
    for L in (6, 8)
    for h in (0.5, 1.0, 2.0)
]
c.item(
    "free light shift at t* (6^4, 8^4): max at h = 0.5, max at h <= 2",
    (max(sh[0], sh[3]), max(sh)),
    max(sh[0], sh[3]) <= 0.009 and max(sh) <= 0.12,
    "{0[0]:.4f}, {0[1]:.3f}",
)
c.done()
