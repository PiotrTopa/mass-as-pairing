#!/usr/bin/env python3
"""K5.3 -- stage 0: the pre-registered gates G1-G4 applied as written to the 26 stage-0 chains, with the sabotage
controls of the gate logic. Reads data/derived/n1stage/S0 and data/derived/free/free_L4*.json (a few seconds)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import copy

from masspairing.analysis import stage0
from masspairing.claimcheck import Check

c = Check("K5.3")
rows = stage0.stage0_rows()
c.item(
    "26 chains at 600 trajectories; acceptance range",
    (min(r["acc"] for r in rows), max(r["acc"] for r in rows)),
    len(rows) == 26 and all(r["n"] == 600 for r in rows),
    "{0[0]:.2f}-{0[1]:.2f}",
)
taus = [t for r in rows for t in r["taus"]]
c.item("tau_int of the m_L, chi_L, GL_p0, phi_R series (measurements), max", max(taus), max(taus) < 1.5, "{:.2f}")

# G1
g1 = stage0.gate1(rows)
for y, h, e, _ in g1:
    c.record(f"G1 4^3x8 y={y} h={h}: rel. errors m_L, m_R, chi_L", f"{e[0]:.1%}, {e[1]:.1%}, {e[2]:.1%}")
c.item(
    "G1 PASS on all six 4^3x8 rows: max rel. error m_L, m_R, chi_L",
    tuple(max(e[i] for _, _, e, _ in g1) for i in range(3)),
    len(g1) == 6 and all(o for *_, o in g1),
    "{0[0]:.1%}, {0[1]:.1%}, {0[2]:.1%}",
)

# G2
g2 = stage0.gate2(rows)
for h, v, e, fr, pull, cr, gr, _ in g2:
    c.record(
        f"G2 h={h}: r_L [free ratio] pull, chi_L/free, GL_p0/free",
        f"{v:.4f}({e:.4f}) [{fr:.4f}] {pull:+.2f}, {cr:.3f}, {gr:.3f}",
    )
d2 = {h: (pull, cr, gr, o, v, e) for h, v, e, fr, pull, cr, gr, o in g2}
low = (0.1, 0.2, 0.5, 1.0)
c.item(
    "G2 passes at h <= 1: max |pull|",
    max(abs(d2[h][0]) for h in low),
    all(d2[h][3] for h in low) and max(abs(d2[h][0]) for h in low) < 2.0,
    "{:.2f}",
)
c.item(
    "G2 fails at h = 2 on r_L alone: r_L, pull",
    (d2[2.0][4], d2[2.0][5], d2[2.0][0]),
    not d2[2.0][3] and 3.0 < d2[2.0][0] < 4.0 and 1.015 < d2[2.0][4] < 1.026,
    "{0[0]:.3f}({0[1]:.3f}), {0[2]:+.2f} sigma",
)
c.item(
    "G2 chi_L/free and GL_p0/free ranges over h",
    (
        min(v[1] for v in d2.values()),
        max(v[1] for v in d2.values()),
        min(v[2] for v in d2.values()),
        max(v[2] for v in d2.values()),
    ),
    all(0.90 <= v[1] <= 0.93 and 0.94 <= v[2] <= 0.96 for v in d2.values()),
    "{0[0]:.2f}-{0[1]:.2f}, {0[2]:.2f}-{0[3]:.2f}",
)

# G3
g3 = stage0.gate3(rows)
for e in g3:
    c.record(
        f"G3 4^3x{e['Lt']} y={e['y']} h={e['h']}",
        f"m_R shift {e['dm']:+.4f}({e['edm']:.4f}) = {e['sig']:+.1f} sigma [free {e['fdm']:+.4f}]; phi_R/h "
        f"{e['psig']:+.1f} sigma, {e['pratio']:.3f} x free",
    )
h05 = [e for e in g3 if e["h"] == 0.5]
c.item(
    "G3 (i) h = 0.5: max m_R-shift significance; free shifts (4^4, 4^3x8)",
    (
        max(e["sig"] for e in h05),
        [e["fdm"] for e in h05 if e["Lt"] == 4][0],
        [e["fdm"] for e in h05 if e["Lt"] == 8][0],
    ),
    not any(e["okm"] for e in h05) and max(e["sig"] for e in h05) < 3 and all(abs(e["fdm"]) < 0.01 for e in h05),
    "{0[0]:+.1f} sigma; {0[1]:+.4f}, {0[2]:+.4f}",
)
smg = [e for e in g3 if e["y"] == 3.0]
c.item(
    "G3 (ii) y = 3.0: m_R shift significance range (<= +1 sigma), max phi_R/h over free",
    (min(e["sig"] for e in smg), max(e["sig"] for e in smg), max(e["pratio"] for e in smg)),
    all(e["sig"] < 1.0 for e in smg) and all(e["pratio"] < 0.03 for e in smg) and any(e["sig"] < -10 for e in smg),
    "{0[0]:+.1f}...{0[1]:+.1f} sigma; {0[2]:.3f}",
)
s1 = [e for e in smg if e["Lt"] == 4 and e["h"] in (1.0, 2.0)]
c.item(
    "G3 (ii) 4^4 y = 3.0: m_R shift at h = 1, 2 (free +0.03, +0.12)",
    tuple(e["sig"] for e in s1),
    True,
    "{0[0]:+.1f}, {0[1]:+.1f} sigma",
)
c.item(
    "G3 (ii) y = 3.0: phi_R/h > 5 sigma only at h = 2",
    True,
    all(e["psig"] > 5 for e in smg if e["h"] == 2.0) and not any(e["psig"] > 5 for e in smg if e["h"] < 2.0),
)
sym = [e for e in g3 if e["y"] < 2.5 and e["h"] >= 1.0]
c.item(
    "G3 passes at y = 2.0/2.41, h >= 1: min shift significance, phi_R/h over free range",
    (min(e["sig"] for e in sym), min(e["pratio"] for e in sym), max(e["pratio"] for e in sym)),
    all(e["okm"] and e["okp"] for e in sym)
    and min(e["sig"] for e in sym) > 3.5
    and all(0.69 < e["pratio"] < 0.94 for e in sym),
    "{0[0]:.1f} sigma; {0[1]:.2f}-{0[2]:.2f}",
)
c.item("phi_R/h > 5 sigma at every h at y = 2.0 and 2.41", True, all(e["okp"] for e in g3 if e["y"] < 2.5))

# G4 (recorded)
g4 = stage0.gate4(rows)
s4 = [e for e in g4 if e["y"] == 2.0]
m4 = [e for e in g4 if e["y"] == 3.0 and e["psig"] > 2]
p4 = [e for e in g4 if e["y"] == 2.41]
c.item(
    "G4 |phi_T_R/phi_R| at y = 2.0, max",
    max(abs(e["ratio"]) for e in s4),
    all(abs(e["ratio"]) < 0.002 for e in s4),
    "{:.4f}",
)
c.item(
    "G4 at y = 3.0 (phi_R resolved): range; phi_T_R/h > 50 x free",
    (min(abs(e["ratio"]) for e in m4), max(abs(e["ratio"]) for e in m4)),
    len(m4) >= 3
    and all(abs(e["ratio"]) > 2.5 for e in m4)
    and all(abs(e["pT"]) > 50 * abs(e["pTfree"]) for e in g4 if e["y"] == 3.0),
    "{0[0]:.1f}-{0[1]:.1f}",
)
c.item(
    "G4 at P_c (aaaa), range",
    (min(abs(e["ratio"]) for e in p4), max(abs(e["ratio"]) for e in p4)),
    all(0.004 < abs(e["ratio"]) < 0.007 for e in p4),
    "{0[0]:.4f}-{0[1]:.4f}",
)

# controls of the gate logic
get = lambda y, Lt, h: stage0.get(rows, y, 4, Lt, h)
sym2 = lambda r: abs(r["y"] - 2.0) < 1e-9 and r["L"] == 4 and r["Lt"] == 4 and r["bc"] == "aaaa"
R = copy.deepcopy(rows)
for r in R:
    if sym2(r):
        r["mL"] = (0.42 * r["mL_free"] / 0.6931 * (1 + 0.01 * r["h"]), 0.004)
        r["chi_L"] = (0.9 * r["chi_L_free"], 1e-4)
        r["GL_p0"] = (0.95 * r["GL_p0_free"], 0.002)
cc = stage0.gate2(R)
c.item(
    "control: an injected free-like SYM series passes G2 (max |pull|)",
    max(abs(x[4]) for x in cc),
    all(x[7] for x in cc) and max(abs(x[4]) for x in cc) < 2,
    "{:.2f}",
)
R = copy.deepcopy(rows)
for r in R:
    if sym2(r) and r["h"] > 0:
        r["mL"] = (r["mL"][0] / (1 + 3 * r["h"]), r["mL"][1])
cc = stage0.gate2(R)
c.item(
    "control: an injected 1/(1 + 3h) fall of m_L is flagged by G2 at every h (pulls)",
    [round(float(x[4]), 1) for x in cc],
    not any(x[7] for x in cc) and all(x[4] < -2 for x in cc),
)
R = copy.deepcopy(rows)
for r in R:
    if r["y"] < 2.5 and r["bc"] == "aaaa" and r["h"] > 0:
        b = get(r["y"], r["Lt"], 0.0)
        r["mR"] = (2 * b["mR"][0] - r["mR"][0], r["mR"][1])
        r["phi_R"] = (-r["phi_R"][0], r["phi_R"][1])
cc = [e for e in stage0.gate3(R) if e["y"] < 2.5 and e["h"] >= 1.0]
c.item(
    "control: a mis-signed h is flagged by G3 on every y = 2.0/2.41 row with h >= 1",
    True,
    not any(e["okm"] or e["okp"] for e in cc),
)
R = copy.deepcopy(rows)
for r in R:
    for k in ("mL", "mR", "chi_L"):
        r[k] = (r[k][0], 20 * r[k][1])
c.item("control: errors inflated x20 are flagged by G1", True, not all(o for *_, o in stage0.gate1(R)))
R = copy.deepcopy(rows)
for r in R:
    r["y"] = {2.0: 3.0, 3.0: 2.0}.get(r["y"], r["y"])
cc = stage0.gate4(R)
c.item(
    "control: the G4 ordering fails with the y labels swapped",
    True,
    not all(abs(e["ratio"]) < 0.02 for e in cc if e["y"] == 2.0)
    and not all(abs(e["ratio"]) > 1 for e in cc if e["y"] == 3.0 and e["psig"] > 2),
)
c.done()
