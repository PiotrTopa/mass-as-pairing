"""K2.1: tau_2 K structure of the doublet operator: D_c anti-hermitian with a +-i mu paired spectrum, det D_c > 0
configuration by configuration; the on-site propagator blocks are i n.tau, so tr S(x, x) = 0 and
O4 = det S(x, x) = |n|^2.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.algebra.signfree import onsite_structure
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice
from masspairing.operator import DoubletOperator

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K2.1")
lat = Lattice((4, 4, 4, 4))  # periodic space, antiperiodic time
V = lat.V
rng = np.random.default_rng(40)

# free theory: the on-site block vanishes (no bilinear, O4 = 0)
D0 = DoubletOperator(lat, 0.0)
M0 = D0.dense(np.zeros(3 * V))
r0 = onsite_structure(M0, V)
c.item(
    "y = 0: D anti-hermitian exactly; max |n(x)| (free on-site block)",
    (np.abs(M0 + M0.conj().T).max(), float(np.abs(r0["n"]).max())),
    np.abs(M0 + M0.conj().T).max() == 0.0 and np.abs(r0["n"]).max() < 1e-12,
)

worst = {k: 0.0 for k in ("re_ev", "pairing", "trace_inv", "tr_onsite", "quaternion", "n_residual", "det_residual")}
signs = []
D = DoubletOperator(lat, 1.3)
for _ in range(3):
    sigma = rng.normal(size=3 * V)
    M = D.dense(sigma)
    assert np.abs(M + M.conj().T).max() == 0.0
    r = onsite_structure(M, V)
    for k in worst:
        worst[k] = max(worst[k], r[k])
    signs.append((np.sin(r["im_logdet"]), np.cos(r["im_logdet"]), r["re_logdet_defect"]))
c.item(
    "y = 1.3, 3 random sigma: D_c anti-hermitian; max |Re eigenvalue|",
    worst["re_ev"],
    worst["re_ev"] < 1e-12,
    fmt="{:.1e}",
)
c.item("  spectrum paired as +-i mu (max defect)", worst["pairing"], worst["pairing"] < 1e-10, fmt="{:.1e}")
s_max = max(abs(s[0]) for s in signs)
c_min = min(s[1] for s in signs)
re_def = max(abs(s[2]) for s in signs)
c.item(
    "  det D_c real positive: max |sin Im log det|, min cos Im log det, |Re log det - 1/2 log det D^+D|",
    (f"{s_max:.1e}", c_min, f"{re_def:.1e}"),
    s_max < 1e-10 and c_min > 0 and re_def < 1e-10,
)
c.item(
    "  |Tr D_c^-1| and max |tr S(x, x)|",
    (f"{worst['trace_inv']:.1e}", f"{worst['tr_onsite']:.1e}"),
    worst["trace_inv"] < 1e-10 and worst["tr_onsite"] < 1e-12,
)
c.item(
    "  quaternion structure tau_2 S* tau_2 = S of the on-site blocks (max defect)",
    worst["quaternion"],
    worst["quaternion"] < 1e-12,
    fmt="{:.1e}",
)
c.item(
    "  S(x, x) = i n(x).tau with n real (max residual)", worst["n_residual"], worst["n_residual"] < 1e-12, fmt="{:.1e}"
)
c.item(
    "  O4 = det S(x, x) = |n(x)|^2 >= 0 (max residual)",
    worst["det_residual"],
    worst["det_residual"] < 1e-12,
    fmt="{:.1e}",
)
c.done()
