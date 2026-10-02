#!/usr/bin/env python3
"""K4.3 -- the composite takeover with the taste-chiral Majorana mass on 8^4, 6^4 and 6^3x12 (stage-1 chains):
SMG screens the elementary heavy channel, the epsilon-vertex composite carries the response, linear in h and
lattice-independent; at P_c (aaaa) the response is on the elementary side. Reads data/derived/n1stage/S1 (about 1 min).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.claimcheck import Check

c = Check("K4.3")
rows = stage1.stage1_rows()
K = stage1.k4(rows)
R, C = K["rows"], K["criteria"]
for k, e in sorted(R.items(), key=lambda kv: (kv[1]["Lt"], kv[1]["L"], kv[1]["y"], kv[1]["h"])):
    rho = (
        f"{e['rho'][0]:.3f}({e['rho'][1]:.3f})"
        if e["rho"][2] == "resolved"
        else f"> {e['rho'][0]:.1f} (phi_R unresolved)"
    )
    c.record(
        k,
        f"phi_R/h {e['phi_R_per_h'][0]:+.5f}({e['phi_R_per_h'][1]:.5f}) = {e['phi_R_over_free']:.3f} x free; "
        f"phi_T_R/h {e['phi_T_R_per_h'][0]:+.6f}({e['phi_T_R_per_h'][1]:.6f}); rho {rho}; A "
        f"{e['A'][0]:+.4f}({e['A'][1]:.4f})",
    )
smg = [e for e in R.values() if e["y"] == 3.0]
pc = [e for e in R.values() if e["y"] == 2.41]
c.item(
    "(i) SMG: rho > 4 (lower bounds where phi_R is unresolved) and phi_R/h <= 0.05 x free on all nine rows; min rho",
    min(e["rho"][0] for e in smg),
    len(smg) == 9 and C["8x8"]["i_smg"] and C["6x6"]["i_smg"] and C["6x12"]["i_smg"],
    "{:.1f}",
)
c.item(
    "(i) SMG: max phi_R/h over free",
    max(abs(e["phi_R_over_free"]) for e in smg),
    all(abs(e["phi_R_over_free"]) <= 0.012 for e in smg),
    "{:.3f}",
)
pr = {}
for e in smg:
    pr.setdefault((e["L"], e["Lt"]), {})[e["h"]] = e["phi_R_per_h"][0]
c.item(
    "(i) SMG: phi_R/h grows with h on every lattice (second-order response)",
    True,
    all(v[2.0] > v[1.0] > 0 for v in pr.values()),
)
c.item(
    "(ii) P_c (aaaa): rho < 0.2 on all nine rows; range",
    (min(e["rho"][0] for e in pc), max(e["rho"][0] for e in pc)),
    len(pc) == 9
    and C["8x8"]["ii_sym"]
    and C["6x6"]["ii_sym"]
    and C["6x12"]["ii_sym"]
    and max(e["rho"][0] for e in pc) < 0.013,
    "{0[0]:.3f}-{0[1]:.3f}",
)
p8 = [e["phi_T_R_per_h"][0] for e in smg if e["L"] == 8]
spread8 = (max(p8) - min(p8)) / abs(np.mean(p8))
vol = max(
    abs(e["phi_T_R_per_h"][0] - f["phi_T_R_per_h"][0]) / abs(f["phi_T_R_per_h"][0])
    for e in smg
    for f in smg
    if e["h"] == f["h"] and (e["L"], e["Lt"]) != (f["L"], f["Lt"])
)
c.item("(iii) SMG phi_T_R/h on 8^4 at h = 0.5, 1, 2", p8, True, "{}")
c.item("(iii) SMG phi_T_R/h: spread in h on 8^4", spread8, C["iii_linear_8"] and spread8 < 0.012, "{:.1%}")
c.item(
    "(iii) SMG phi_T_R/h: max difference between lattices at equal h", vol, C["iii_volume_6_8"] and vol < 0.02, "{:.1%}"
)
c.item(
    "(iii) SMG phi_T_R/h over free (6^4, 6^3x12), min",
    min(abs(e["phi_T_R_over_free"]) for e in smg if e["L"] == 6),
    all(abs(e["phi_T_R_over_free"]) >= 44 for e in smg if e["L"] == 6),
    "{:.0f}",
)
c.item(
    "(iv) the ordering does not reverse with the volume; the y-label swap fails (i)/(ii)",
    True,
    C["iv_no_reversal"] and C["control_swap_fails"],
)
pcT = {}
for e in pc:
    pcT.setdefault((e["L"], e["Lt"]), {})[e["h"]] = e["phi_T_R_per_h"][0]
c.item(
    "(v) P_c: phi_T_R/h falls from h = 0.5 to 2 (factor range); phi_R/h over free range",
    (
        min(v[0.5] / v[2.0] for v in pcT.values()),
        max(v[0.5] / v[2.0] for v in pcT.values()),
        min(e["phi_R_over_free"] for e in pc),
        max(e["phi_R_over_free"] for e in pc),
    ),
    all(2.0 < v[0.5] / v[2.0] < 3.5 for v in pcT.values()) and all(0.55 < e["phi_R_over_free"] < 0.9 for e in pc),
    "{0[0]:.1f}-{0[1]:.1f}x; {0[2]:.2f}-{0[3]:.2f}",
)
a8 = next(e for e in pc if e["L"] == 8 and e["h"] == 2.0)
fA = stage1.get(rows, 2.41, 8, 8, 2.0)["free"]["A"]
c.item(
    "(v) P_c 8^4 h = 2: on-configuration asymmetry A (free)",
    (a8["A"][0], a8["A"][1], fA),
    0.26 < a8["A"][0] < 0.28 and a8["A"][0] > fA + 5 * a8["A"][1],
    "{0[0]:.3f}({0[1]:.3f}) ({0[2]:.3f})",
)
c.item("the composite criterion holds at 8^4", True, C["holds_at_8"])
c.done()
