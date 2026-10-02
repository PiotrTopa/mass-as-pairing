#!/usr/bin/env python3
"""K4.2 -- stage 0 (4^4, 4^3x8): the SMG phase screens the elementary taste-chiral Majorana mass and the composite
carries the response; the light sector is suppressed but h-stable; the screening is a property of the configurations
(quench at fixed sigma on stored configurations).

Reads data/derived/n1stage/S0 (26 chains), the free baselines data/derived/free/free_L4*.json and the stored
configurations data/configs/n1stage_quench_*.npz (dense re-measurement, about 4 min single-threaded).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import copy

import numpy as np

from masspairing.analysis import stage0
from masspairing.analysis.n1stage import remeasure, stored_configs
from masspairing.claimcheck import Check

c = Check("K4.2")
rows = stage0.stage0_rows()
aaaa = [r for r in rows if r["bc"] == "aaaa"]
get = lambda y, Lt, h: stage0.get(rows, y, 4, Lt, h)
c.item("chains read (26 at 600 trajectories)", len(rows), len(rows) == 26 and all(r["n"] == 600 for r in rows))

# ---------------------------------------------------------------- G4: composite / elementary ratio
g4 = stage0.gate4(rows)
s4 = [e for e in g4 if e["y"] == 2.0]
m4 = [e for e in g4 if e["y"] == 3.0 and e["psig"] > 2]
p4 = [e for e in g4 if e["y"] == 2.41]
for e in g4:
    c.record(
        f"rho = phi_T_R/phi_R 4^3x{e['Lt']} y={e['y']} h={e['h']}",
        f"{e['ratio']:+.4f}({e['err']:.4f}); phi_T_R/h {e['pT']:+.6f} [free {e['pTfree']:+.1e}]",
    )
c.item(
    "|rho| at y = 2.0 (SYM), max over h",
    max(abs(e["ratio"]) for e in s4),
    all(abs(e["ratio"]) < 0.002 for e in s4),
    "{:.4f}",
)
c.item(
    "|rho| at y = 3.0 (SMG) where phi_R is resolved (> 2 sigma), range",
    (min(abs(e["ratio"]) for e in m4), max(abs(e["ratio"]) for e in m4)),
    len(m4) >= 3 and all(abs(e["ratio"]) > 2.5 for e in m4),
    "{0[0]:.1f}-{0[1]:.1f}",
)
c.item(
    "phi_T_R/h at y = 3.0 above 50 x free at every h",
    min(abs(e["pT"] / e["pTfree"]) for e in g4 if e["y"] == 3.0),
    all(abs(e["pT"]) > 50 * abs(e["pTfree"]) for e in g4 if e["y"] == 3.0),
    "{:.0f}",
)
c.item(
    "|rho| at P_c (y = 2.41, aaaa), range",
    (min(abs(e["ratio"]) for e in p4), max(abs(e["ratio"]) for e in p4)),
    all(0.004 < abs(e["ratio"]) < 0.007 for e in p4),
    "{0[0]:.4f}-{0[1]:.4f}",
)
R = copy.deepcopy(rows)
for r in R:
    r["y"] = {2.0: 3.0, 3.0: 2.0}.get(r["y"], r["y"])
sw = stage0.gate4(R)
c.item(
    "control: the ordering fails with the y labels swapped",
    True,
    not all(abs(e["ratio"]) < 0.02 for e in sw if e["y"] == 2.0)
    and not all(abs(e["ratio"]) > 1 for e in sw if e["y"] == 3.0 and e["psig"] > 2),
)

# ---------------------------------------------------------------- 1. on-configuration asymmetry A(h)
sym = [r for r in aaaa if r["y"] < 2.5 and r["h"] >= 0.5]
smg = [r for r in aaaa if r["y"] == 3.0 and r["h"] > 0]
for r in sorted(aaaa, key=lambda r: (r["Lt"], r["y"], r["h"])):
    c.record(f"A 4^3x{r['Lt']} y={r['y']} h={r['h']}", f"{r['A'][0]:+.4f}({r['A'][1]:.4f}) [free {r['A_free']:+.4f}]")
c.item(
    "A(h >= 0.5) > 0 at y = 2.0/2.41: min pull",
    min(r["A"][0] / r["A"][1] for r in sym),
    all(r["A"][0] / r["A"][1] > 3 for r in sym),
    "{:.1f}",
)
a2 = [r for r in sym if r["h"] == 2.0]
c.item(
    "A(2) at y = 2.0/2.41 (free 0.111-0.119)",
    (min(r["A"][0] for r in a2), max(r["A"][0] for r in a2)),
    all(0.13 < r["A"][0] < 0.16 and 0.11 < r["A_free"] < 0.12 for r in a2),
    "{0[0]:.3f}-{0[1]:.3f}",
)
c.item(
    "y = 3.0: max |A|, max |pull| (every h <= 2)",
    (max(abs(r["A"][0]) for r in smg), max(abs(r["A"][0] / r["A"][1]) for r in smg)),
    all(abs(r["A"][0]) < 0.012 and abs(r["A"][0] / r["A"][1]) < 3 for r in smg),
    "{0[0]:.4f}, {0[1]:.1f}",
)
c.item(
    "y = 3.0 on 4^3x8: max |A| at h = 0.5, 2 (free 0.009 / 0.119)",
    max(abs(r["A"][0]) for r in smg if r["Lt"] == 8),
    all(abs(r["A"][0]) < 0.001 for r in smg if r["Lt"] == 8),
    "{:.4f}",
)

# ---------------------------------------------------------------- 2. elementary vs composite at y = 3.0
for r in sorted(smg, key=lambda r: (r["Lt"], r["h"])):
    c.record(
        f"y=3.0 4^3x{r['Lt']} h={r['h']}",
        f"phi_R/h {r['phi_R'][0] / r['h']:+.5f}({r['phi_R'][1] / r['h']:.5f}) = "
        f"{r['phi_R'][0] / r['phi_R_free']:.3f} x free;"
        f" phi_T_R/h {r['phi_T_R'][0] / r['h']:+.6f}({r['phi_T_R'][1] / r['h']:.6f})",
    )
c.item(
    "y = 3.0: max phi_R/h over free",
    max(r["phi_R"][0] / r["phi_R_free"] for r in smg),
    all(r["phi_R"][0] / r["phi_R_free"] <= 0.03 for r in smg),
    "{:.3f}",
)
pT4 = {r["h"]: r["phi_T_R"][0] / r["h"] for r in smg if r["Lt"] == 4}
pT8 = {r["h"]: r["phi_T_R"][0] / r["h"] for r in smg if r["Lt"] == 8}
spread = (max(pT4.values()) - min(pT4.values())) / abs(np.mean(list(pT4.values())))
vol = max(abs(pT8[h] - pT4[h]) / abs(pT4[h]) for h in (0.5, 2.0))
c.item(
    "phi_T_R/h range on 4^4, h in [0.1, 2]", (min(pT4.values()), max(pT4.values())), True, "{0[0]:+.5f} ... {0[1]:+.5f}"
)
c.item("phi_T_R/h h-spread on 4^4 (< 6 %)", spread, spread < 0.06, "{:.1%}")
c.item("phi_T_R/h 4^4 vs 4^3x8 (< 6 %)", vol, vol < 0.06, "{:.1%}")

# ---------------------------------------------------------------- 3. light sector vs free at h = 0
seq = {}
for Lt, ys in ((4, (2.0, 2.41, 3.0)), (8, (2.41, 3.0))):
    seq[Lt] = []
    for y in ys:
        r = get(y, Lt, 0.0)
        v = (y, r["chi_L"][0] / r["chi_L_free"], r["GL_p0"][0] / r["GL_p0_free"], r["chi_R"][0] / r["chi_R_free"])
        seq[Lt].append(v)
        c.record(f"h=0 4^3x{Lt} y={y}: chi_L/free, GL_p0/free, chi_R/free", f"{v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}")
c.item(
    "chi_L/free and GL_p0/free decrease SYM -> P_c -> SMG on both lattices",
    True,
    all(s[i][1] > s[i + 1][1] and s[i][2] > s[i + 1][2] for s in seq.values() for i in range(len(s) - 1)),
)
c.item(
    "SMG h = 0: chi_L/free and GL_p0/free at L_t = 4 / 8 (falling with L_t)",
    (seq[4][2][1], seq[8][1][1], seq[4][2][2], seq[8][1][2]),
    seq[4][2][1] < 0.02 and seq[8][1][1] < 0.01 and seq[4][2][2] < 0.10 and seq[8][1][2] < 0.5 * seq[4][2][2],
    "{0[0]:.4f} / {0[1]:.4f}; {0[2]:.4f} / {0[3]:.4f}",
)
c.item(
    "chi_L/free at SYM (4^4) and P_c (4^4 / 4^3x8)",
    (seq[4][0][1], seq[4][1][1], seq[8][0][1]),
    0.85 < seq[4][0][1] < 0.95 and 0.65 < seq[4][1][1] < 0.75 and 0.65 < seq[8][0][1] < 0.75,
    "{0[0]:.2f}; {0[1]:.2f} / {0[2]:.2f}",
)

# ---------------------------------------------------------------- 4. light sector vs h (pair-channel t* log-ratio)
for r in aaaa:
    if r["h"] > 0:
        v, e, fr = stage0.r_L(r, get(r["y"], r["Lt"], 0.0))
        r["raw"], r["rL"] = (v, e), (v / fr, e / fr)
        c.record(
            f"4^3x{r['Lt']} y={r['y']} h={r['h']}: m_L(h)/m_L(0) [free ratio], r_L",
            f"{v:.4f}({e:.4f}) [{fr:.4f}], {v / fr:.4f}({e / fr:.4f})",
        )
low = [r for r in aaaa if 0 < r["h"] <= 0.5]
two = [r for r in aaaa if r["h"] == 2.0]
c.item(
    "raw m_L(h)/m_L(0) within 2.5 sigma of 1 for h <= 0.5: max pull",
    max(abs(r["raw"][0] - 1) / r["raw"][1] for r in low),
    all(abs(r["raw"][0] - 1) / r["raw"][1] < 2.5 for r in low),
    "{:.1f}",
)
odd = [r for r in low if abs(r["rL"][0] - 1) / r["rL"][1] >= 2.5]
c.item(
    "free-normalised r_L within 2.5 sigma of 1 for h <= 0.5 except (y = 3.0, 4^3x8, h = 0.5)",
    [(r["y"], r["Lt"], r["h"], round(float((r["rL"][0] - 1) / r["rL"][1]), 1)) for r in odd],
    len(odd) <= 1 and all(r["y"] == 3.0 and r["Lt"] == 8 and r["h"] == 0.5 for r in odd),
)
c.item(
    "r_L(2) range over (y, lattice)",
    (min(r["rL"][0] for r in two), max(r["rL"][0] for r in two)),
    all(0.83 <= r["rL"][0] <= 1.10 for r in two),
    "{0[0]:.2f}-{0[1]:.2f}",
)
c.item(
    "y = 3.0: chi_L/free < 0.04 and GL_p0/free < 0.12 at every h",
    (
        max(r["chi_L"][0] / r["chi_L_free"] for r in aaaa if r["y"] == 3.0),
        max(r["GL_p0"][0] / r["GL_p0_free"] for r in aaaa if r["y"] == 3.0),
    ),
    all(
        r["chi_L"][0] / r["chi_L_free"] < 0.04 and r["GL_p0"][0] / r["GL_p0_free"] < 0.12 for r in aaaa if r["y"] == 3.0
    ),
    "{0[0]:.4f}, {0[1]:.4f}",
)
c.item(
    "control: the SMG signature (|A| < 0.012, phi_R < 0.03 free) never at y <= 2.41; A > 0.1 never at y = 3.0",
    True,
    not any(abs(r["A"][0]) < 0.012 and r["phi_R"][0] / r["phi_R_free"] < 0.03 for r in sym)
    and not any(r["A"][0] > 0.1 for r in smg),
)

# ---------------------------------------------------------------- 5. quench at fixed sigma (stored configurations)
Q = {}
for name in ("n1stage_quench_L4_y2_h2", "n1stage_quench_L4_y3_h2", "n1stage_quench_L4x8_y3_h2"):
    z, ch = stored_configs(name)
    hh = ch["h10"]
    res = {h: remeasure(ch, z["fields"], h) for h in (hh, 0.0)}
    ident = max(
        max(
            np.max(np.abs(b["CL_t"] - z["CL_t"][j])),
            np.max(np.abs(b["CR_t"] - z["CR_t"][j])),
            abs(np.sum(b["phi_f"]) - np.sum(z["phi_f"][j])),
        )
        for j, b in enumerate(res[hh])
    )
    mean = lambda k, h: np.mean([b[k] for b in res[h]], 0)
    dR = (mean("CR_t", hh) - mean("CR_t", 0.0)) / mean("CR_t", 0.0)
    dL = (mean("CL_t", hh) - mean("CL_t", 0.0)) / mean("CL_t", 0.0)
    Q[name] = dict(ident=ident, dR=dR, dL=dL, Lt=ch["Lt"])
    c.record(f"{name}: shift h = 2 -> 0 heavy / light", f"{np.round(dR, 3)} / {np.round(dL, 3)}")
c.item(
    "re-measurement of the stored configurations (CL_t, CR_t, phi_f), max deviation",
    max(q["ident"] for q in Q.values()),
    all(q["ident"] < 1e-12 for q in Q.values()),
    "{:.1e}",
)
q2 = Q["n1stage_quench_L4_y2_h2"]
c.item(
    "y = 2.0 fixed sigma: heavy shift range, light max (doublet-selective)",
    (np.min(np.abs(q2["dR"])), np.max(np.abs(q2["dR"])), np.max(np.abs(q2["dL"]))),
    np.all(np.abs(q2["dR"]) > 0.12)
    and np.all(np.abs(q2["dL"]) < 0.04)
    and np.all(np.abs(q2["dL"]) < np.abs(q2["dR"]) / 3),
    "{0[0]:.0%}-{0[1]:.0%}, {0[2]:.1%}",
)
for name in ("n1stage_quench_L4_y3_h2", "n1stage_quench_L4x8_y3_h2"):
    q = Q[name]
    c.item(
        f"y = 3.0 4^3x{q['Lt']} fixed sigma: max |Delta_R - Delta_L| (light and heavy move together), shift range",
        (np.max(np.abs(q["dR"] - q["dL"])), np.min(np.abs(q["dR"])), np.max(np.abs(q["dR"]))),
        np.all(np.abs(q["dR"] - q["dL"]) < 0.02) and np.all(np.abs(q["dR"]) > 0.05),
        "{0[0]:.3f}; {0[1]:.0%}-{0[2]:.0%}",
    )
c.item(
    "control: with L <-> R swapped the y = 2.0 selectivity statement fails",
    True,
    not (np.all(np.abs(q2["dL"]) > 0.12) and np.all(np.abs(q2["dR"]) < 0.04)),
)
c.done()
