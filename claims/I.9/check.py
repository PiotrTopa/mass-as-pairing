"""I.9: the Pfaffian sign of the flavour-selective source -- three routes agree, and the pencil gives the flip points.

(1) random antisymmetric matrices: Parlett-Reid = Householder in sign and log|Pf|; |Pf|^2 = |det|; the congruence
    Pf(B A B^T) = det(B) Pf(A); the sabotage "Parlett-Reid without the pivot-permutation sign" disagrees.
(2) the physical M(h) = M(0) - h C_0 (x) diag(0,0,0,1) on two stored 4^4 configurations and a hot one: Parlett-Reid =
    Householder = Schur route (sign and log|Pf M(h)/Pf M(0)|); flavour-blind mask: sign +1 and log ratio = log det ratio
    of the doublet operator; a negative Pfaffian occurs and all routes give -1; sign(h) on h = 0.1 ... 6 equals
    (-1)^#(pencil flip points below h); source on flavour 3 = source on flavour 4 (SU(2)_R); sabotage (C_0 without the
    boundary sign in the Schur route) disagrees with the direct route.
About 1 min.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.action import build_model
from masspairing.analysis.n1inst import stored_config, stored_model
from masspairing.claimcheck import Check
from masspairing.flavour import realify
from masspairing.lattice import Lattice
from masspairing.patterns import source_pattern
from masspairing.pfaffian import (
    flavour_block_G,
    pf_ratio_schur,
    pf_sign_config,
    pf_sign_direct,
    pfaffian_householder,
    pfaffian_logpr,
    sign_flips,
)

c = Check("I.9")
MASK = np.array([0.0, 0.0, 0.0, 1.0])
HS = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
rng = np.random.default_rng(141)
rngB = np.random.default_rng(1410)


def logpr_nopivotsign(A):
    """Sabotage: Parlett-Reid that forgets the sign of the row/column permutation."""
    A = np.array(A, float)
    n = A.shape[0]
    sign, logabs = 1.0, 0.0
    for k in range(0, n - 1, 2):
        col = np.abs(A[k + 1 :, k])
        p = k + 1 + int(np.argmax(col))
        if p != k + 1:
            A[[k + 1, p], :] = A[[p, k + 1], :]
            A[:, [k + 1, p]] = A[:, [p, k + 1]]
        piv = A[k, k + 1]
        sign *= np.sign(piv)
        logabs += np.log(abs(piv))
        if k + 2 < n:
            tau = A[k, k + 2 :] / piv
            A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1]) - np.outer(A[k + 2 :, k + 1], tau)
    return sign, logabs


# ---- (1) random matrices
neg = sab = 0
e_rand = e_det = e_cong = 0.0
for _ in range(24):
    n = 2 * rng.integers(4, 40)
    A = rng.normal(size=(n, n))
    A = A - A.T
    s1, l1 = pfaffian_logpr(A)
    s2, l2 = pfaffian_householder(A)
    e_rand = max(e_rand, abs(s1 - s2), abs(l1 - l2))
    e_det = max(e_det, abs(2 * l1 - np.linalg.slogdet(A)[1]))
    B = rngB.normal(size=(n, n))
    sB, lB = np.linalg.slogdet(B)
    s3, l3 = pfaffian_householder(B @ A @ B.T)
    e_cong = max(e_cong, abs(s3 - sB * s1), abs(l3 - (lB + l1)) / max(1.0, abs(l3)))
    neg += s1 < 0
    sab += logpr_nopivotsign(A)[0] != s1
c.item("24 random matrices (n = 8 ... 78): Parlett-Reid = Householder (sign, log|Pf|)", e_rand, e_rand < 1e-9, "{:.1e}")
c.item("2 log|Pf| = log|det|", e_det, e_det < 1e-9, "{:.1e}")
c.item("congruence Pf(B A B^T) = det B Pf A (sign, relative log)", e_cong, e_cong < 1e-10, "{:.1e}")
c.item("both signs occur (negative Pfaffians)", int(neg), neg > 0)
c.item("sabotage (no permutation sign) disagrees in sign on some matrices", int(sab), sab > 0)

# ---- (2) the physical operator
lat = Lattice((4, 4, 4, 4))
V = lat.V
C0 = source_pattern(lat).toarray()
cfgs = []
for name in ("L4_y2.41_g0.05_pppa", "L4_y2.41_eps_pppa"):
    latc, meta, F = stored_config(name)
    cfgs.append((stored_model(latc, meta), F, name))
mh = build_model(lat, 2.41, -0.01, 1.0, g10=0.1, g6=0.0)
cfgs.append((mh, np.asarray(mh.start(rng, "hot+hot")), "hot 4^4 (y 2.41, g10 0.1)"))

e_route = e_blind = 0.0
first_neg = None
for model, F, name in cfgs:
    Dc = model.D.dense(F)
    s0, ld0 = np.linalg.slogdet(Dc)
    M0 = realify(Dc, V)
    r = pf_sign_config(model, F, HS, mask=MASK)
    for k, h in enumerate(HS):
        Mh = M0 - h * np.kron(C0, np.diag(MASK))
        s1, l1 = pfaffian_logpr(Mh)
        s2, l2 = pfaffian_householder(Mh)
        e_route = max(e_route, abs(s1 - s2), abs(s1 - r["sign"][k]), abs(l1 - l2), abs(l1 - ld0 - r["logratio"][k]))
        if s1 < 0 and first_neg is None:
            first_neg = (name, h)
    rb = pf_sign_config(model, F, [0.05, 0.3], mask=np.ones(4))
    for k, h in enumerate((0.05, 0.3)):
        sb, ldb = np.linalg.slogdet(Dc - h * np.kron(C0, np.eye(2)))
        e_blind = max(e_blind, abs(rb["sign"][k] - 1.0), abs(rb["logratio"][k] - (ldb - ld0)))
c.item(
    f"{len(cfgs)} configurations x {len(HS)} h: Parlett-Reid = Householder = Schur route",
    e_route,
    e_route < 1e-8,
    "{:.1e}",
)
c.item("flavour-blind mask: sign +1 and log ratio = log det D_c(h)/det D_c(0)", e_blind, e_blind < 1e-8, "{:.1e}")
c.item("first negative Pfaffian on the h grid (all routes -1)", first_neg, first_neg is not None)

hscan = np.concatenate([np.arange(0.1, 1.0, 0.1), np.arange(1.0, 6.01, 0.25)])
e_flip = 0.0
flips = {}
for model, F, name in cfgs:
    S = np.linalg.inv(model.D.dense(F)).reshape(V, 2, V, 2)
    GF, Fl = flavour_block_G(S, MASK)
    CF = np.kron(C0, np.diag(MASK[Fl]))
    sg, _ = pf_ratio_schur(GF, CF, hscan)
    fl = sign_flips(GF, CF)
    flips[name] = fl
    e_flip = max(e_flip, float(np.abs(sg - np.array([(-1.0) ** int((fl < h).sum()) for h in hscan])).max()))
    c.record(f"{name}: sign(h), h = 0.1 ... 6", "".join("+" if v > 0 else "-" for v in sg))
    c.record(f"{name}: pencil flip points h = 1/lambda", np.round(fl[:4], 3).tolist())
c.item("sign(h) on h = 0.1 ... 6 = (-1)^#(pencil flips below h), three configurations", e_flip, e_flip == 0.0, "{:.0e}")
fl_hot, fl_wedge, fl_eps = flips[cfgs[2][2]], flips[cfgs[0][2]], flips[cfgs[1][2]]
c.item(
    "hot configuration: first flip 0.129, second 1.067",
    np.round(fl_hot[:2], 3).tolist(),
    np.allclose(fl_hot[:2], [0.129, 1.067], atol=5e-4),
)
c.item(
    "stored P_c wedge configuration: first flip h = 1.862",
    np.round(fl_wedge[:1], 3).tolist(),
    abs(fl_wedge[0] - 1.862) < 5e-4,
)
c.item(
    "stored P_c epsilon configuration: no flip below h = 6", np.round(fl_eps[:2], 3).tolist(), not np.any(fl_eps < 6.0)
)

model, F, name = cfgs[0]
same = pf_sign_config(model, F, [0.3], mask=np.array([0.0, 0.0, 1.0, 0.0]))
direct = pf_sign_direct(model, F, h=0.3, mask=MASK)
e_sym = abs(direct[1] - (same["logratio"][0] + same["logdet0"])) + abs(direct[0] - same["sign"][0])
c.item("source on flavour 3 = source on flavour 4 (SU(2)_R), h = 0.3", e_sym, e_sym < 1e-8, "{:.1e}")
C0_nb = source_pattern(Lattice((4, 4, 4, 4), bc=(1, 1, 1, 1))).toarray()
Dc = model.D.dense(F)
GF, Fl = flavour_block_G(np.linalg.inv(Dc).reshape(V, 2, V, 2), MASK)
_, lw = pf_ratio_schur(GF, np.kron(C0_nb, np.diag(MASK[Fl])), [0.3])
d_sab = abs(direct[1] - (lw[0] + np.linalg.slogdet(Dc)[1]))
c.item("sabotage (C_0 without the boundary sign, Schur route): log|Pf| off by", d_sab, d_sab > 1e-3, "{:.2f}")
c.done()
