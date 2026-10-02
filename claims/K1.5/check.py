"""K1.5: the Pati-Salam embedding SU(4) x SU(2)_1 x SU(2)_2 = Spin(6) x Spin(4) in Spin(10), the branching of
16, 10, 126, 120, 45, and the channel content of the lattice eps (Lambda^2(4)) and Sym^2(4) bilinears."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.linalg import null_space

from masspairing.algebra.characters import (
    branch_d5_to_ps,
    char126,
    decompose,
    ms_ext_powers,
    pretty,
    spinor16,
    vector10,
)
from masspairing.algebra.spin10 import (
    antisym_products,
    casimir,
    charge_conjugation,
    chirality,
    gammas,
    pati_salam,
    restrict,
)
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers

c = Check("K1.5")
G = gammas()
C = charge_conjugation(G)
ps = pati_salam(G)
P, c6, c4, PiL, PiR = ps["P"], ps["c6"], ps["c4"], ps["PiL"], ps["PiR"]
so6, J1, J2 = ps["so6"], ps["J1"], ps["J2"]

# ---- (1) Spin(6) x Spin(4) on the Weyl 16
ok = np.allclose(c6 @ c6, np.eye(32)) and np.allclose(c4 @ c4, np.eye(32))
ok &= np.allclose(c6 @ c4, chirality(G)) or np.allclose(c6 @ c4, -chirality(G))
c.item(
    "c6 = i G0..G5 and c4 = G6..G9 square to 1, c6 c4 = +-chirality; c6 eigenvalues on the 16",
    sorted(np.round(ps["w6"]).astype(int).tolist()),
    ok and sorted(np.round(ps["w6"]).astype(int)) == [-1] * 8 + [1] * 8,
)
su2 = True
for J in (J1, J2):
    for a in range(3):
        b, cc = (a + 1) % 3, (a + 2) % 3
        comm = J[a] @ J[b] - J[b] @ J[a]
        su2 &= np.allclose(comm, 1j * J[cc]) or np.allclose(comm, -1j * J[cc])
su2 &= all(np.allclose(A @ B, B @ A) for A in J1 for B in J2)
c.item("J1, J2 close into two commuting su(2) algebras", su2, su2)
blocks = {}
for name, B in (("c6=+1", PiL), ("c6=-1", PiR)):
    C6, C1, C2 = casimir(so6, P, B), casimir(J1, P, B), casimir(J2, P, B)
    A = np.vstack([np.kron(restrict(X, P, B).T, np.eye(8)) - np.kron(np.eye(8), restrict(X, P, B)) for X in so6])
    ncomm = null_space(A).shape[1]
    commute = all(
        np.allclose(restrict(X, P, B) @ restrict(J, P, B), restrict(J, P, B) @ restrict(X, P, B))
        for X in so6
        for J in J1 + J2
    )
    c1, c2 = np.round(C1[0, 0].real, 6), np.round(C2[0, 0].real, 6)
    ok = np.allclose(C6, 3.75 * np.eye(8)) and ncomm == 4 and commute
    ok &= np.allclose(C1, c1 * np.eye(8)) and np.allclose(C2, c2 * np.eye(8))
    blocks[name] = (float(c1), float(c2))
    c.item(
        f"block {name}: so(6) Casimir 15/4, commutant dimension, su(2)_1 / su(2)_2 Casimirs",
        (15 / 4, ncomm, float(c1), float(c2)),
        ok,
    )
c.item(
    "one block a doublet of SU(2)_1 only, the other of SU(2)_2 only",
    sorted(blocks.values()),
    sorted(blocks.values()) == [(0.0, 0.75), (0.75, 0.0)],
)

# ---- (2) exact branching D5 -> D3 x A1 x A1
s16 = spinor16(-1)
br = {
    "16": decompose(branch_d5_to_ps(s16), 2, 3),
    "10": decompose(branch_d5_to_ps(vector10()), 2, 3),
    "126": decompose(branch_d5_to_ps(char126()), 2, 3),
    "120": decompose(branch_d5_to_ps(ms_ext_powers(s16, 2)[2]), 2, 3),
    "45": decompose(branch_d5_to_ps(ms_ext_powers(vector10(), 2)[2]), 2, 3),
}
expected = {
    "16": {(1, 1, -1, 1, 0): 1, (1, 1, 1, 0, 1): 1},
    "10": {(2, 0, 0, 0, 0): 1, (0, 0, 0, 1, 1): 1},
    "126": {(2, 0, 0, 0, 0): 1, (2, 2, -2, 2, 0): 1, (2, 2, 2, 0, 2): 1, (2, 2, 0, 1, 1): 1},
    "120": {
        (0, 0, 0, 1, 1): 1,
        (2, 2, -2, 0, 0): 1,
        (2, 2, 2, 0, 0): 1,
        (2, 0, 0, 2, 0): 1,
        (2, 0, 0, 0, 2): 1,
        (2, 2, 0, 1, 1): 1,
    },
    "45": {(2, 2, 0, 0, 0): 1, (0, 0, 0, 2, 0): 1, (0, 0, 0, 0, 2): 1, (2, 0, 0, 1, 1): 1},
}
for rep, dec in br.items():
    c.item(
        f"{rep} -> (D3 x A1 x A1 highest weights, dimensions of the products)", pretty(dec, 2, 3), dec == expected[rep]
    )

# ---- (3) channel content of the one-block (LL, RR) bilinears inside Sym^2(16) = 10-span + 126-span
B10 = [P.T @ C @ g @ P for g in G]
B126 = [P.T @ C @ M @ P for M in antisym_products(G, 5)]


def onb(mats):
    A = np.array([m.ravel() for m in mats])
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    return vh[: int((s > 1e-8).sum())].conj().T


Q10, Q126 = onb(B10), onb(B126)
c.item(
    "10- and 126-spans orthogonal, dimensions",
    (Q10.shape[1], Q126.shape[1]),
    Q10.shape[1] == 10 and Q126.shape[1] == 126 and np.allclose(Q10.conj().T @ Q126, 0),
)


def frac(M):
    v = M.ravel()
    n2 = np.linalg.norm(M) ** 2
    return np.linalg.norm(Q10.conj().T @ v) ** 2 / n2, np.linalg.norm(Q126.conj().T @ v) ** 2 / n2


for name, B in (("LL", PiL), ("RR", PiR)):
    six = [B @ (B.T @ B10[i] @ B) @ B.T for i in range(6)]
    fr = [frac(M) for M in six]
    angle = np.degrees(np.arccos(np.sqrt(fr[0][0])))
    c.item(
        f"{name} Lambda^2(4) = (6,1,1) bilinears: fraction in 10, in 126, mixing angle",
        (round(float(fr[0][0]), 6), round(float(fr[0][1]), 6), round(float(angle), 2)),
        all(abs(a - 0.5) < 1e-9 and abs(b - 0.5) < 1e-9 for a, b in fr),
    )
    E = np.eye(8)
    sym = [B @ (E[:, [a]] @ E[[b]] + E[:, [b]] @ E[[a]]) @ B.T for a in range(8) for b in range(a, 8)]
    Qs, Q6 = onb(sym), onb(six)
    comp = Qs - Q6 @ (Q6.conj().T @ Qs)
    u, s, _ = np.linalg.svd(comp, full_matrices=False)
    comp = u[:, s > 1e-8]
    f10 = np.linalg.norm(Q10.conj().T @ comp) ** 2 / comp.shape[1]
    f126 = np.linalg.norm(Q126.conj().T @ comp) ** 2 / comp.shape[1]
    c.item(
        f"{name} Sym^2(4) = (10,3,1)/(10bar,1,3) bilinears (30 dims): fraction in 10, in 126",
        (f"{f10:.1e}", round(float(f126), 12)),
        Qs.shape[1] == 36 and Q6.shape[1] == 6 and comp.shape[1] == 30 and f10 < 1e-12 and abs(f126 - 1) < 1e-12,
    )
    no122 = max(np.linalg.norm(B.T @ B10[i] @ B) for i in range(6, 10))
    c.item(f"{name}: the (1,2,2) of the 10 has no {name} component", no122, no122 < 1e-12, fmt="{:.1e}")
X = [P.T @ C @ G[i] @ c4 @ P for i in range(6)]
fx = min(frac(M)[1] for M in X)
c.item(
    "the (6,1,1) of the 126 is Gamma_i Gamma_6789 (min fraction in the 126)",
    fx,
    all(abs(frac(M)[1] - 1) < 1e-12 for M in X),
    fmt="{:.12f}",
)
c.done()
