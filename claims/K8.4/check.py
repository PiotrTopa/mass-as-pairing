#!/usr/bin/env python3
"""K8.4 -- at P_c the light half approaches its FREE values on the same box, h and boundary conditions as h grows and
never overshoots them; the light pair-channel cosh mass does not distinguish the symmetric phase from SMG on 6^4.
Rows by the stage-1b reader (cut ts_cfg_traj >= 100) from data/derived/n1stage/{S1,S1b,S1c,TH} and the free baselines
(about 15 s)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import direction as D
from masspairing.claimcheck import Check

c = Check("K8.4")
rows = D.free_compare()
pc, ref = D.pc_rows(rows)
HS = D.HS
res = D.crit_no_overshoot(pc)

# 1. the pair-channel mass against free (6^4, 8^4)
for lat in ("L8", "L6"):
    R = res[lat + "_m"][1]
    c.item(
        f"{lat}: m/m_free at h = 0, 0.5, 1, 1.5, 2, 3 -- below free at every h (2 sigma), rising",
        [x[0] for x in R],
        res[lat + "_m"][0],
        "{0[0]:.3f} {0[1]:.3f} {0[2]:.3f} {0[3]:.3f} {0[4]:.3f} {0[5]:.3f}",
    )
for lat in ("L8", "L6"):
    c.record(
        f"{lat}: m (m_free) at h = 0, 1, 2, 3",
        ", ".join(
            f"{pc[(lat, h)]['m'][0]:.3f}({pc[(lat, h)]['m'][1]:.3f}) ({pc[(lat, h)]['m_free']:.3f})"
            for h in (0.0, 1.0, 2.0, 3.0)
        ),
    )

# 2. g and c on all three lattices
for lat in ("L8", "L6", "L6x12"):
    for O in ("g", "c"):
        a, b = res[f"{lat}_{O}"][1]
        c.item(
            f"{lat}: {O} from h = 0 to 3 -- toward free, never above 1 (2 sigma), > 0.8 at h = 3",
            (a, b),
            res[f"{lat}_{O}"][0],
            "{0[0][0]:.3f}({0[0][1]:.3f}) -> {0[1][0]:.3f}({0[1][1]:.3f})",
        )

# 3. 6^3x12: the midpoint ratio estimator (no free cosh solution there)
d = [pc[("L6x12", h)]["dmeff"] for h in HS]
c.record("6^3x12 dmeff = -ln[R(t+1)/R(t)], R = C/C_free, h = 0 ... 3", " ".join(f"{x[0]:+.3f}({x[1]:.3f})" for x in d))
c.item(
    "6^3x12: dmeff falls toward 0 (toward free) from h = 0 to 3; positive already at the critical h = 0 point "
    "(not read as a gap)",
    (d[0][0], d[-1][0]),
    d[-1][0] < d[0][0] - 2 * np.hypot(d[0][1], d[-1][1]) and d[0][0] > 0,
    "{0[0]:+.3f} -> {0[1]:+.3f}",
)

# 4. the pair-channel mass is not a phase discriminator (6^4, h = 0)
mS, mM, gS, gM = ref["SYM0"]["Rm"], ref["SMG0"]["Rm"], ref["SYM0"]["g"], ref["SMG0"]["g"]
c.item(
    "6^4 h = 0: m/m_free at y = 2.0 (SYM) vs y = 3.0 (SMG), differing by < 0.05, while g differs by > 0.8",
    (mS[0], mM[0], gS[0], gM[0]),
    abs(mS[0] - mM[0]) < 0.05 and gS[0] - gM[0] > 0.8,
    "{0[0]:.3f} vs {0[1]:.3f}; g {0[2]:.3f} vs {0[3]:.4f}",
)
c.record(
    "6^4 SYM (y = 2.0) h = 3: m/m_free, g, c",
    f"{ref['SYM3']['Rm'][0]:.3f}, {ref['SYM3']['g'][0]:.3f}, {ref['SYM3']['c'][0]:.3f}",
)
c.record(
    "6^4 SMG (y = 3.0) m/m_free at h = 0, 2, 3; g, c at h = 0 and 3",
    f"{ref['SMG0']['Rm'][0]:.3f}, {ref['SMG2']['Rm'][0]:.3f}, {ref['Y3H3']['Rm'][0]:.3f}; "
    f"g {ref['SMG0']['g'][0]:.4f} -> "
    f"{ref['Y3H3']['g'][0]:.4f}, c {ref['SMG0']['c'][0]:.4f} -> {ref['Y3H3']['c'][0]:.4f}",
)
c.item(
    "anchors: 8^4 m(0), m(3); 6^4 m(0), m(3) (stage 1 / 1b rows)",
    (pc[("L8", 0.0)]["m"][0], pc[("L8", 3.0)]["m"][0], pc[("L6", 0.0)]["m"][0], pc[("L6", 3.0)]["m"][0]),
    abs(pc[("L8", 0.0)]["m"][0] - 0.197) < 5e-4
    and abs(pc[("L8", 3.0)]["m"][0] - 0.764) < 5e-4
    and abs(pc[("L6", 0.0)]["m"][0] - 0.392) < 5e-4
    and abs(pc[("L6", 3.0)]["m"][0] - 0.645) < 5e-4,
    "{0[0]:.3f}, {0[1]:.3f}; {0[2]:.3f}, {0[3]:.3f}",
)

# 5. sabotage
f1, f2 = D.sabotage_free(pc)
c.item("sabotage: a synthetic 6^4 h = 3 row 10 % above free (a gap beyond free) is flagged", f1, f1)
c.item("sabotage: a synthetic 8^4 h = 3 row with the SMG value of g is flagged", f2, f2)
c.done()
