#!/usr/bin/env python3
"""K8.7 -- the epsilon-vertex density O4 at P_c read against h-MATCHED references: P_c moves toward the symmetric side
after h-matching, while the heavy doublet's mass alone removes a large part of O4 outside SMG; the light-doublet
restriction O4L carries no phase information. C0 errors, the stage-1b reader's selection; data
data/derived/n1stage/{S1,S1b,S1c,TH} and data/derived/n1stage/o4l_series.npz (about 10 s)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import direction as D
from masspairing.claimcheck import Check

c = Check("K8.7")
R = D.o4_rows()
pc6 = {h: R[(6, 6, h)] for h in (0.0, 1.0, 2.0, 3.0)}
S0, S3, M0, M3 = R["TH:SYM0"], R["TH:SYM3"], R["TH:SMG0"], R["TH:Y3H3"]

# 1. O4L carries no phase information
for L, Lt in ((8, 8), (6, 6), (6, 12)):
    fr = [R[(L, Lt, h)]["O4L"][0] / R[(L, Lt, h)]["O4"][0] for h in D.HS]
    c.record(f"P_c {L}^3x{Lt}: O4L/O4 at h = 0 ... 3", " ".join(f"{x:.3f}" for x in fr))
c.item(
    "6^4 h = 0: O4L at SYM vs SMG (within x2) while O4 differs by > x5 -- the stored observables cannot split the "
    "epsilon vertex into light and heavy parts",
    (S0["O4L"][0], M0["O4L"][0], S0["O4"][0], M0["O4"][0], M0["O4"][0] / S0["O4"][0]),
    abs(np.log(M0["O4L"][0] / S0["O4L"][0])) < np.log(2) and M0["O4"][0] / S0["O4"][0] > 5,
    "O4L {0[0]:.1e} vs {0[1]:.1e}; O4 {0[2]:.4f} vs {0[3]:.4f} (x{0[4]:.1f})",
)

# 2. h-matched position of P_c between SYM (y = 2.0) and SMG (y = 3.0) on 6^4
good, (f0, e0, f3, e3) = D.toward_sym_o4(R, pc6[0.0], pc6[3.0])
c.item(
    "f_O4 = (O4_Pc - O4_SYM(h)) / (O4_SMG(h) - O4_SYM(h)) at h = 0 -> 3 (h-matched references): toward SYM "
    "beyond 2 sigma",
    (f0, e0, f3, e3),
    good,
    "{0[0]:.3f}({0[1]:.3f}) -> {0[2]:.3f}({0[3]:.3f})",
)
r_u, r_m = pc6[3.0]["O4"][0] / S0["O4"][0], pc6[3.0]["O4"][0] / S3["O4"][0]
c.item(
    "'O4 falls to the SYM value' holds only against SYM(h = 0): O4_Pc(3)/O4_SYM(0), O4_Pc(3)/O4_SYM(3) (h-matched)",
    (r_u, r_m),
    r_m > 1.5 and abs(r_u - 1) < 0.1,
    "{0[0]:.2f}, {0[1]:.2f}",
)

# 3. the heavy doublet dropping out of the vertex: relative fall of O4 from h = 0 to 3 at fixed y (6^4)
dS = 1 - S3["O4"][0] / S0["O4"][0]
dM = 1 - M3["O4"][0] / M0["O4"][0]
dP = 1 - pc6[3.0]["O4"][0] / pc6[0.0]["O4"][0]
c.item(
    "relative fall of O4 from h = 0 to 3 at fixed y on 6^4: SYM (y = 2.0), SMG (y = 3.0), P_c -- phase-dependent, "
    "largest at P_c",
    (100 * dS, 100 * dM, 100 * dP),
    dP > dS > dM,
    "-{0[0]:.0f} %, -{0[1]:.0f} %, -{0[2]:.0f} %",
)

# 4. sabotage
fake = dict(pc6[3.0])
fake["O4"] = list(M3["O4"])
c.item(
    "sabotage: a synthetic P_c(h = 3) row at the h-matched SMG value does not read 'toward SYM'",
    True,
    not D.toward_sym_o4(R, pc6[0.0], fake)[0],
)
c.done()
