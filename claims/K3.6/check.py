"""K3.6: the columnar bond crystal (dimer channel) condenses on the +edge g6 = +g10.

Points (y, g10) = (2.41, 0.05), (3.0, 0.1) and the reference point (2.41, 0.1) at kappa = -0.01, lambda = 1, g6 = g10;
L = 4 (exact propagators, 400 trajectories), 6 (1000) and 8 (1500; 8 noise vectors every 2nd trajectory); on 6^4 at
the reference point three starts (hot+zero, afm+dimer, hot+hot).
Items: (1) at the reference point V<D_par^2> grows with exponent >= 0.65 on (4,6) and (6,8), the link-field dimer
length xi/L rises 4 -> 6 -> 8 and V<D_perp^2> stays below 0.05 (columnar order); (2) the eps-channel susceptibility
V<|phi_stag|^2> grows sub-linearly at every point (exponent on (6,8) < 0.5); (3) at (2.41, 0.05) the dimer exponent
on (6,8) exceeds the eps one by more than 1 sigma; (4) no hysteresis on 6^4: the three starts agree within 3.5 sigma.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import k3
from masspairing.claimcheck import Check

P = [(2.41, 0.05), (3.0, 0.1), (2.41, 0.1)]
REF = (2.41, 0.1)
LS = (4, 6, 8)
VSCALED = ("dimer_sq_par", "dimer_sq_perp", "phi_stag_sq")
KEYS = ("dimer_sq_par", "dimer_sq_perp", "phi_stag_sq", "O4", "xi2_stag", "xi2_s_dimer_1")


def mv(r, k):
    s = r["V"] if k in VSCALED else 1.0
    return s * r[k]["mean"], s * r[k]["err"]


c = Check("K3.6")
R = {(p, L): k3.ensemble_of([(k3.c073_rel(L, *p), 0)]) for p in P for L in LS}
for p in P:
    for L in LS:
        r = R[(p, L)]
        c.record(
            f"{p} L={L}: ntraj, acc, V<D_par^2>, V<D_perp^2>, V<|phi_st|^2>, O4, xi2_stag, xi2 link dimer",
            f"{r['ntraj']} {r['acceptance']:.2f} " + " ".join("{:.4g}({:.2g})".format(*mv(r, k)) for k in KEYS),
        )
ex = {}
for p in P:
    for name, k in (("dimer", "dimer_sq_par"), ("eps", "phi_stag_sq")):
        for a, b in ((4, 6), (6, 8)):
            ex[(p, name, a, b)] = k3.exponent(mv(R[(p, a)], k), mv(R[(p, b)], k), a, b)
    c.record(
        f"{p} exponents d ln X/d ln V (4,6) (6,8)",
        "  ".join(
            f"{n} {ex[(p, n, 4, 6)][0]:.2f}({ex[(p, n, 4, 6)][1]:.2f}) "
            f"{ex[(p, n, 6, 8)][0]:.2f}({ex[(p, n, 6, 8)][1]:.2f})"
            for n in ("dimer", "eps")
        ),
    )
e1, e2 = ex[(REF, "dimer", 4, 6)], ex[(REF, "dimer", 6, 8)]
c.item(
    "(1) reference point: V<D_par^2> exponents (4,6), (6,8) >= 0.65",
    (e1, e2),
    e1[0] >= 0.65 and e2[0] >= 0.65,
    fmt="{0[0][0]:.2f}({0[0][1]:.2f}), {0[1][0]:.2f}({0[1][1]:.2f})",
)
xl = [mv(R[(REF, L)], "xi2_s_dimer_1")[0] / L for L in LS]
c.item(
    "(1) link-field dimer correlation length xi/L at L = 4, 6, 8 rising",
    xl,
    xl[0] < xl[1] < xl[2],
    fmt="{0[0]:.3f}, {0[1]:.3f}, {0[2]:.3f}",
)
dp = max(mv(R[(REF, L)], "dimer_sq_perp")[0] for L in LS)
c.item("(1) V<D_perp^2> at the reference point (largest over L, < 0.05): columnar order", dp, dp < 0.05, fmt="{:.3f}")
e68 = [ex[(p, "eps", 6, 8)] for p in P]
c.item(
    "(2) eps exponent on (6,8) at (2.41, 0.05), (3.0, 0.1), (2.41, 0.1) (all < 0.5)",
    e68,
    all(e[0] < 0.5 for e in e68),
    fmt="{0[0][0]:.2f}, {0[1][0]:.2f}, {0[2][0]:.2f}",
)
(ed, sd), (ee, se) = ex[((2.41, 0.05), "dimer", 6, 8)], ex[((2.41, 0.05), "eps", 6, 8)]
z = (ed - ee) / np.hypot(sd, se)
c.item(
    "(3) (2.41, 0.05): dimer exponent vs eps exponent on (6,8), separation in sigma (> 1)",
    (ed, sd, ee, se, z),
    z > 1,
    fmt="{0[0]:.2f}({0[1]:.2f}) vs {0[2]:.2f}({0[3]:.2f}): {0[4]:.1f}",
)
H = {s: k3.ensemble_of([(k3.c073_rel(6, *REF, start=s), 0)]) for s in ("main", "afmdimer", "hothot")}
worst = 0.0
for s1, s2 in (("main", "afmdimer"), ("main", "hothot"), ("afmdimer", "hothot")):
    for k in KEYS:
        (a, ea), (b, eb) = mv(H[s1], k), mv(H[s2], k)
        worst = max(worst, abs(a - b) / np.hypot(ea, eb))
c.item(
    f"(4) 6^4 hysteresis hot+zero / afm+dimer / hot+hot: largest pull over {len(KEYS)} observables (< 3.5)",
    worst,
    worst < 3.5,
    fmt="{:.2f}",
)
c.done()
