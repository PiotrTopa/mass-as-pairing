"""K2.4: the sign-free (tau_2 K) class is quaternion-linear, span{1, Gamma_a} (x) R^{V x V}, hence flavour-democratic:
every element commutes with the hidden SU(2)' (Gamma'), which acts transitively on flavour directions; sign-free
same-parity Majorana sources are flavour-blind; the 1-of-4 and 2+2 splits lie outside the class."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.algebra.signfree import class_distance, commutant, flavour_restriction, quaternion_units
from masspairing.claimcheck import Check
from masspairing.data import derived
from masspairing.flavour import GAMMA, GAMMA2, realify
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.operator import DoubletOperator
from masspairing.patterns import chiral_mass, singular_values, source_pattern
from masspairing.wedge import WedgeGeometry

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K2.4")

# ---- (1) quaternion structure and the commutant
Ji, Jt = quaternion_units()
c.item(
    "J_i^2 = J_tau^2 = -1, {J_i, J_tau} = 0",
    True,
    np.allclose(Ji @ Ji, -np.eye(4)) and np.allclose(Jt @ Jt, -np.eye(4)) and np.allclose(Ji @ Jt + Jt @ Ji, 0),
)
N4 = commutant([Ji, Jt], 4)
B = [np.eye(4)] + list(GAMMA)
proj = lambda N, v: N @ (N.T @ v)
e1 = max(np.linalg.norm(proj(N4, b.reshape(-1, order="F")) - b.reshape(-1, order="F")) for b in B)
c.item(
    "commutant of H in M_4(R): dimension, contains 1 and Gamma_1..3 (residual)",
    (N4.shape[1], f"{e1:.1e}"),
    N4.shape[1] == 4 and e1 < 1e-12,
)
lat = Lattice((2, 2, 2, 2))
V = lat.V
N = commutant([np.kron(np.eye(V), Ji), np.kron(np.eye(V), Jt)], 4 * V)
c.item(
    "2^4 (4V = 64): dimension of the commutant of {J_i (x) 1, J_tau (x) 1} (= 4V^2)",
    N.shape[1],
    N.shape[1] == 4 * V * V,
)
rng = np.random.default_rng(165)
worst = 0.0
for _ in range(20):
    A = rng.normal(size=(V, V))
    k = rng.integers(4)
    M = np.kron(A, B[k]).reshape(-1, order="F")
    worst = max(worst, np.linalg.norm(proj(N, M) - M) / np.linalg.norm(M))
K = kinetic_matrix(lat).toarray()
Kv = np.kron(K, np.eye(4)).reshape(-1, order="F")
worst = max(worst, np.linalg.norm(proj(N, Kv) - Kv))
surv = np.linalg.norm(proj(N, rng.normal(size=16 * V * V))) / np.sqrt(16 * V * V)
c.item(
    "random A (x) Gamma_k and K (x) 1 lie in the class (residual); norm fraction of a random matrix that survives",
    (f"{worst:.1e}", round(float(surv), 2)),
    worst < 1e-12,
)

# ---- (2) Gamma' and democracy
lat4 = Lattice((4,) * 4)
V4 = lat4.V
G2 = [np.kron(np.eye(V4), g) for g in GAMMA2]
geom = WedgeGeometry(lat4, "all", 0.1, 0.0)
fields = np.concatenate([rng.normal(size=3 * V4), rng.normal(size=geom.n_fields) * 0.3])
mats = {
    "random sigma, s, chiral mass h = 0.3": realify(
        DoubletOperator(lat4, 2.41, geom=geom, h=0.3, pattern="chiral").dense(fields), V4
    )
}
cfg = np.load(derived("algebra", "eps_L4_y2.41_k-0.01_final.npz"))["fields"]
mats["stored 4^4 P_c configuration"] = realify(DoubletOperator(lat4, 2.41).dense(cfg), V4)
for name, M in mats.items():
    e = max(abs(M @ g - g @ M).max() for g in G2)
    specs = []
    for u in (np.array([1, 0, 0, 0.0]), np.array([0, 0, 0, 1.0]), rng.normal(size=4)):
        specs.append(np.sort(np.linalg.svd(flavour_restriction(M, V4, u), compute_uv=False)))
    e2 = max(abs(s - specs[0]).max() for s in specs)
    c.item(
        f"realified D_c ({name}): [M, Gamma' (x) 1]; flavour-resolved spectra of flavour 1, 4, random direction "
        "(max difference)",
        (f"{e:.1e}", f"{e2:.1e}"),
        e < 1e-12 and e2 < 1e-10,
    )
e_1 = np.array([1, 0, 0, 0.0])
Q = np.stack([e_1] + [g @ e_1 for g in GAMMA2], 1)
e_tr = 0.0
for t in (np.array([0, 0, 0, 1.0]), np.array([0, 1, 0, 0.0]), rng.normal(size=4)):
    t = t / np.linalg.norm(t)
    cc = np.linalg.solve(Q, t)
    R = cc[0] * np.eye(4) + sum(ck * g for ck, g in zip(cc[1:], GAMMA2, strict=True))
    e_tr = max(e_tr, abs(np.linalg.norm(cc) - 1.0), abs(R @ e_1 - t).max(), abs(R.T @ R - np.eye(4)).max())
c.item(
    "SU(2)' (unit quaternions c0 + c.Gamma') maps e_1 to e_4, e_2 and a random unit vector (residual)",
    e_tr,
    abs(np.linalg.det(Q)) > 0.5 and e_tr < 1e-12,
    fmt="{:.1e}",
)

# ---- (3) Majorana sources inside / outside the class
C0 = source_pattern(lat4).toarray()
Cx = chiral_mass(lat4).toarray()
d_blind, d_c0 = class_distance(np.kron(Cx, np.eye(4)), V4), class_distance(np.kron(C0, np.eye(4)), V4)
c.item("1 (x) C_chi and 1 (x) C0 in the class (relative distance)", (d_blind, d_c0), d_blind < 1e-13 and d_c0 < 1e-13)
d_sel = class_distance(np.kron(C0, np.diag([0, 0, 0, 1.0])), V4)
d_sel_chi = class_distance(np.kron(Cx, np.diag([0, 0, 1, 1.0])), V4)
m = np.array([0.2, 0.5, 1.0, 1.3])
d_m, d_m_exp = class_distance(np.kron(Cx, np.diag(m)), V4), np.linalg.norm(m - m.mean()) / np.linalg.norm(m)
c.item(
    "1-of-4 split C0 (x) diag(0,0,0,1): distance (= sqrt(3)/2)",
    d_sel,
    abs(d_sel - np.sqrt(3 / 4)) < 1e-12,
    fmt="{:.4f}",
)
c.item(
    "2+2 chiral split C_chi (x) diag(0,0,1,1): distance (= 1/sqrt(2))",
    d_sel_chi,
    abs(d_sel_chi - np.sqrt(1 / 2)) < 1e-12,
    fmt="{:.4f}",
)
c.item(
    "diag(m) (x) C_chi, m = (0.2, 0.5, 1, 1.3): distance (= |m - mean m| / |m|)",
    d_m,
    abs(d_m - d_m_exp) < 1e-12,
    fmt="{:.4f}",
)
Msel = np.kron(C0, np.diag([0, 0, 0, 1.0]))
e_sel = max(abs(Msel @ g - g @ Msel).max() for g in G2)
c.item("the flavour-selective source does not commute with Gamma' (x) 1", e_sel, e_sel > 0.5, fmt="{:.2f}")
Sy = rng.normal(size=(V4, V4))
Sy = Sy + Sy.T
An = rng.normal(size=(V4, V4))
An = An - An.T
Mtest = np.kron(An, np.eye(4)) + np.kron(Sy, GAMMA[1])
blocks = [Mtest.reshape(V4, 4, V4, 4)[:, a, :, a] for a in range(4)]
c.item(
    "1 (x) A_antisym + Gamma (x) S_sym: antisymmetric class element whose four flavour blocks all equal A",
    (f"{class_distance(Mtest, V4):.1e}", f"{max(abs(b - An).max() for b in blocks):.1e}"),
    abs(Mtest + Mtest.T).max() < 1e-12
    and class_distance(Mtest, V4) < 1e-13
    and max(abs(b - An).max() for b in blocks) < 1e-12,
)

# ---- (4) the taste-chiral mass gives every flavour direction |m| = h
latp = Lattice((4,) * 4, bc=(1, 1, 1, 1))
Kp, Cp = kinetic_matrix(latp).toarray(), chiral_mass(latp).toarray()
Mfull = np.kron(Kp - 0.5 * Cp, np.eye(4))
e4 = 0.0
for u in ([1, 0, 0, 0], [0, 0, 0, 1], rng.normal(size=4)):
    s = singular_values(flavour_restriction(Mfull, latp.V, u), 16)
    e4 = max(e4, abs(s[:8]).max(), abs(s[8:16] - 0.5).max())
c.item(
    "K - h C_chi (periodic 4^4, h = 0.5) on flavour 1, 4 and a random direction: 8 zero modes + 8 at h (max dev.)",
    e4,
    e4 < 1e-6,
    fmt="{:.1e}",
)
c.done()
