"""K1.6: the four-flavour reduced staggered model (Euclidean and 2+1d Hamiltonian): channel algebra, Fierz identities,
symmetries; the lattice U(1)_eps is not U(1)_psi."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.linalg import expm

from masspairing.algebra.grassmann import Grass, substitute
from masspairing.algebra.hamiltonian import cluster_ops, h10_operator
from masspairing.algebra.spin10 import casimir, gammas, pati_salam, restrict, so10_generators
from masspairing.algebra.staggered import (
    FieldAlgebra,
    invariant_dim,
    levi_civita4,
    random_su4,
    rep_on_tensor,
    su4_generators,
    sym_antisym_projectors,
    tensor_gens,
)
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers

c = Check("K1.6")
rng = np.random.default_rng(20)

# ---- (1) SU(4) channel algebra
gens = su4_generators()
Ps, Pa = sym_antisym_projectors()
G44 = rep_on_tensor(gens, ["fund", "fund"])
c.item(
    "Sym^2(4) and Lambda^2(4) are invariant subspaces of 4 x 4 (dimensions)",
    (Ps.shape[1], Pa.shape[1]),
    Ps.shape[1] == 10 and Pa.shape[1] == 6 and max(np.abs(Pa.conj().T @ g @ Ps).max() for g in G44) < 1e-12,
)
g10 = [Ps.conj().T @ g @ Ps for g in G44]
g6 = [Pa.conj().T @ g @ Pa for g in G44]
bar = lambda gs: [-g.T for g in gs]
n44, n44b = invariant_dim(G44), invariant_dim(rep_on_tensor(gens, ["fund", "anti"]))
c.item("singlets in 4 x 4 and in 4 x 4bar (the kinetic term)", (n44, n44b), (n44, n44b) == (0, 1))
n66, n66b = invariant_dim(tensor_gens(g6, g6)), invariant_dim(tensor_gens(g6, bar(g6)))
c.item("singlets in 6 x 6 and 6 x 6bar (6 self-conjugate)", (n66, n66b), (n66, n66b) == (1, 1))
n1010, n1010b = invariant_dim(tensor_gens(g10, g10)), invariant_dim(tensor_gens(g10, bar(g10)))
c.item("singlets in 10 x 10 and 10 x 10bar: no holomorphic 10 quartic", (n1010, n1010b), (n1010, n1010b) == (0, 1))
A = np.vstack(tensor_gens(g6, g6))
v = np.linalg.svd(A)[2][-1].conj()
T = np.einsum("iA,jB,AB->ij", Pa, Pa, v.reshape(6, 6)).reshape(4, 4, 4, 4)
eps = levi_civita4()
cc = np.vdot(eps.ravel(), T.ravel()) / np.vdot(eps.ravel(), eps.ravel())
dev = np.linalg.norm(T - cc * eps) / np.linalg.norm(T)
c.item("the 6 x 6 singlet is epsilon_abcd (relative deviation)", dev, dev < 1e-9, fmt="{:.1e}")

# ---- (2) Grassmann statements on one quad (sites 0, 1 even; 2, 3 odd)
fa = FieldAlgebra(4)
par = [1, 1, -1, -1]
n = fa.ngen
onsite_sym = all(not fa.Phi(0, 0, a, b).t for a in range(4) for b in range(4))
onsite = [fa.Lam(0, 0, a, b) for a in range(4) for b in range(a + 1, 4)]
c.item(
    "on-site chi^a chi^b: symmetric part vanishes, antisymmetric part spans",
    len(onsite),
    onsite_sym and len(onsite) == 6 and all(o.t for o in onsite),
)
E = fa.eps_term(0)
q = (0, 1, 2, 3)
T10, T6 = fa.sym2_term(q), fa.lam2_term(q)
f10 = (T10 - fa.sym2_term(q, explicit=True)).maxabs()
f6 = (T6 - fa.lam2_term(q, explicit=True)).maxabs()
c.item(
    "Fierz forms: sum Phibar Phi = 2[K K - K K], sum Lambdabar Lambda = -2[K K + K K] (max coefficient error)",
    (f10, f6),
    f10 == 0 and f6 == 0,
)
kin = fa.K(0, 2) + fa.K(1, 3)
worst_inv, worst_u1, min_break = 0.0, 0.0, np.inf
for _ in range(3):
    V = fa.map_su4(random_su4(rng), par)
    for poly in (E, T10, T6, kin):
        worst_inv = max(worst_inv, (substitute(poly, V, n) - poly).maxabs())
    Vu = fa.map_u1eps(rng.uniform(0.3, 1.2), par)
    worst_u1 = max(worst_u1, (substitute(T10, Vu, n) - T10).maxabs(), (substitute(kin, Vu, n) - kin).maxabs())
    min_break = min(min_break, (substitute(E, Vu, n) - E).maxabs())
c.item(
    "eps-term, Sym^2 and Lambda^2 terms, kinetic bilinears invariant under chi_e -> U chi_e, chi_o -> U* chi_o",
    worst_inv,
    worst_inv < 1e-10,
    fmt="{:.1e}",
)
c.item("kinetic and Sym^2 terms U(1)_eps-invariant", worst_u1, worst_u1 < 1e-10, fmt="{:.1e}")
z4 = (substitute(E, fa.map_u1eps(np.pi / 2, par), n) - E).maxabs()
c.item(
    "eps-term breaks U(1)_eps (min change at random angles) exactly to Z_4 (change at pi/2)",
    (min_break, z4),
    min_break > 0.1 and z4 < 1e-10,
)
PP = Grass()
for a in range(4):
    for b in range(4):
        PP = PP + fa.Phi(0, 1, a, b) * fa.Phi(0, 1, a, b)
dsu4 = (substitute(PP, fa.map_su4(random_su4(rng), par), n) - PP).maxabs()
O = np.linalg.qr(rng.normal(size=(4, 4)))[0]
O *= np.sign(np.linalg.det(O))
dso4 = (substitute(PP, fa.map_su4(O, par), n) - PP).maxabs()
c.item(
    "the charge-4 candidate sum_ab Phi^ab Phi^ab: nonzero, changes under SU(4), invariant under SO(4)",
    (round(dsu4, 4), dso4),
    bool(PP.t) and dsu4 > 1e-3 and dso4 < 1e-10,
)

# ---- (3) the 2+1d Hamiltonian on a 2-site cluster
ops = cluster_ops(2, [(0, 1)])
xi = ops["xi"]
H10 = h10_operator(ops, g=1.0)
nop = lambda s: sum(xi(s, a).conj().T @ xi(s, a) for a in range(4))
K = sum(xi(0, a).conj().T @ xi(1, a) for a in range(4))
e1 = np.abs(H10 - (-2) * (nop(0) @ nop(1) - K.conj().T @ K + nop(1))).max()
c.item("sum_ab Phi^+ Phi = 2 (n n' - K^+ K + n')", e1, e1 < 1e-12, fmt="{:.1e}")
e2 = np.abs(ops["Dops"][0] @ ops["Dops"][0] - 2 * xi(0, 0) @ xi(0, 1) @ xi(0, 2) @ xi(0, 3)).max()
c.item("D_r^2 = 2 xi1 xi2 xi3 xi4 (eps-term = (V/2) sum (D^2 + h.c.))", e2, e2 < 1e-12, fmt="{:.1e}")
N = nop(0) + nop(1)
comm = lambda A, B: np.abs(A @ B - B @ A).max()
Z4 = expm(1j * np.pi / 2 * N)
vals = (comm(N, ops["Hband"]), comm(N, H10), comm(N, ops["Heps"]), comm(Z4, ops["Heps"]))
c.item(
    "[N, H_band], [N, H_10], [N, H_eps], [e^{i pi N/2}, H_eps]",
    np.round(vals, 12),
    vals[0] < 1e-12 and vals[1] < 1e-12 and vals[2] > 1e-3 and vals[3] < 1e-10,
)
worst = 0.0
for g in gens:
    Q = sum(g[a, b] * xi(s, a).conj().T @ xi(s, b) for s in range(2) for a in range(4) for b in range(4))
    worst = max(worst, comm(Q, ops["Hband"]), comm(Q, H10), comm(Q, ops["Heps"]))
c.item("H_band, H_10, H_eps commute with the 15 SU(4) charges", worst, worst < 1e-10, fmt="{:.1e}")

# ---- (4) U(1)_eps is not U(1)_psi (16 x 16 on the Weyl 16)
G = gammas()
ps = pati_salam(G)
P, PiL, PiR, so6 = ps["P"], ps["PiL"], ps["PiR"], ps["so6"]
combos = [
    (B, J, Jo)
    for B in (PiL, PiR)
    for J, Jo in ((ps["J1"], ps["J2"]), (ps["J2"], ps["J1"]))
    if abs(casimir(J, P, B)[0, 0] - 0.75) < 1e-9
]
B4, JL, JR = combos[0]  # the block that is a doublet of JL is the (4,2,1)
B4bar = PiR if B4 is PiL else PiL
c.item(
    "(4,2,1) and (4bar,1,2) blocks identified (two doublet assignments)",
    len(combos),
    len(combos) == 2 and abs(casimir(JL, P, B4bar)[0, 0]) < 1e-9 and abs(casimir(JR, P, B4bar)[0, 0] - 0.75) < 1e-9,
)
Q_eps = B4 @ B4.conj().T - B4bar @ B4bar.conj().T
Q_psi = np.eye(16)
gens16 = [restrict(X, P) for X in so10_generators(G)]
T4, TL, TR = [restrict(X, P) for X in so6], [restrict(X, P) for X in JL], [restrict(X, P) for X in JR]
c1 = max(np.abs(Q_eps @ X - X @ Q_eps).max() for X in T4 + TL + TR)
c.item("[Q_eps, su(4) + su(2)_L + su(2)_R]", c1, c1 < 1e-12, fmt="{:.1e}")
Amat = np.array([X.ravel() for X in gens16 + [Q_psi]]).T
resid = lambda Q: np.linalg.norm(Amat @ np.linalg.lstsq(Amat, Q.ravel(), rcond=None)[0] - Q.ravel()) / np.linalg.norm(Q)
r_eps, r_psi = resid(Q_eps), resid(Q_psi)
c.item(
    "relative residual outside spin(10) + u(1)_psi: Q_eps, Q_psi",
    (round(float(r_eps), 4), f"{r_psi:.1e}"),
    r_eps > 0.99 and r_psi < 1e-10,
)
tr = lambda *Ms: (np.trace(np.linalg.multi_dot(Ms)) if len(Ms) > 1 else np.trace(Ms[0])).real
a1, a3 = tr(Q_eps), tr(Q_eps, Q_eps, Q_eps)
a44 = max(abs(tr(Q_eps, X, Y)) for X in T4 for Y in T4)
aLL = max(abs(tr(Q_eps, X, Y)) for X in TL for Y in TL)
aRR = max(abs(tr(Q_eps, X, Y)) for X in TR for Y in TR)
p1, p3 = tr(Q_psi), tr(Q_psi, Q_psi, Q_psi)
c.item(
    "Tr Q_eps, Tr Q_eps^3, max Tr Q_eps T4 T4",
    (round(a1, 12), round(a3, 12), round(a44, 12)),
    abs(a1) < 1e-12 and abs(a3) < 1e-12 and a44 < 1e-12,
)
c.item(
    "max Tr Q_eps T_L T_L, max Tr Q_eps T_R T_R (mixed anomaly with the taste SU(2)s)",
    (round(aLL, 12), round(aRR, 12)),
    abs(aLL - 2) < 1e-12 and abs(aRR - 2) < 1e-12,
)
c.item("Tr Q_psi, Tr Q_psi^3", (round(p1, 12), round(p3, 12)), abs(p1 - 16) < 1e-12 and abs(p3 - 16) < 1e-12)
zeps, gL = expm(1j * np.pi / 2 * Q_eps), expm(2j * np.pi * TL[2])
d1 = np.abs(zeps - expm(-1j * np.pi / 2 * Q_psi) @ gL).max()
d2 = np.abs(zeps - expm(1j * np.pi / 2 * Q_psi) @ gL).max()
onL, onR = (B4.conj().T @ gL @ B4)[0, 0], (B4bar.conj().T @ gL @ B4bar)[0, 0]
c.item(
    "e^{i pi Q_eps/2} = e^{-i pi Q_psi/2} e^{2 pi i J_L3}: max difference (other sign)",
    (f"{d1:.1e}", round(d2, 3)),
    d1 < 1e-12 and d2 > 1,
)
c.item(
    "e^{2 pi i J_L3} on (4,2,1), on (4bar,1,2)",
    (np.round(onL, 12), np.round(onR, 12)),
    abs(onL + 1) < 1e-12 and abs(onR - 1) < 1e-12,
)
c.done()
