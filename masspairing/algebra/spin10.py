"""Spin(10) Clifford algebra, the Weyl 16 and its bilinears, Lie-algebra data and the Pati-Salam blocks.

Euclidean signature; the 32-dimensional Dirac representation is built from Pauli tensors:

    Gamma_{2k} = sz^{(x)k} (x) sx (x) 1^{(x)(4-k)},   Gamma_{2k+1} = sz^{(x)k} (x) sy (x) 1^{(x)(4-k)},   k = 0..4.

Gamma_{2k} are real symmetric, Gamma_{2k+1} imaginary antisymmetric, and the Cartan generators
H_k = -(i/2) Gamma_{2k} Gamma_{2k+1} = (1/2) sz^{(x)k} (x) sz (x) 1 are diagonal. The charge conjugation C with
C Gamma_i = +Gamma_i^T C is the product of the five symmetric gammas. A Weyl 16 is the +1 eigenspace of the chirality
i Gamma_0 ... Gamma_9; its bilinears psi^T C Gamma^{[k]} psi span 10 (k = 1), 120 (k = 3) and 126 (k = 5).
"""

from __future__ import annotations

import functools
import itertools

import numpy as np

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1, -1]).astype(complex)
I2 = np.eye(2)


def kron(*ms):
    out = np.eye(1)
    for m in ms:
        out = np.kron(out, m)
    return out


@functools.cache
def _gammas_cached():
    G = []
    for k in range(5):
        pre = [sz] * k
        post = [I2] * (4 - k)
        G.append(kron(*pre, sx, *post))
        G.append(kron(*pre, sy, *post))
    return tuple(g.copy() for g in G)


def gammas():
    """The ten hermitian anticommuting 32 x 32 gamma matrices (fresh copies)."""
    return [g.copy() for g in _gammas_cached()]


def chirality(G):
    return 1j * np.linalg.multi_dot(G)


@functools.cache
def _charge_conjugation_cached(sign):
    G = _gammas_cached()
    idx = range(0, 10, 2) if sign == +1 else range(1, 10, 2)
    C = np.linalg.multi_dot([G[i] for i in idx])
    assert all(np.allclose(C @ g, sign * g.T @ C) for g in G)
    return C


def charge_conjugation(G=None, sign=+1):
    """C with C Gamma_i = sign Gamma_i^T C for the standard gammas: C_+ = G0 G2 G4 G6 G8, C_- = G1 G3 G5 G7 G9.

    The product of the five symmetric (antisymmetric) gammas intertwines Gamma_i with +Gamma_i^T (-Gamma_i^T); the
    defining relation is asserted, uniqueness up to scale is Schur's lemma on the irreducible 32.
    """
    return _charge_conjugation_cached(sign).copy()


@functools.cache
def _weyl_basis_cached():
    G = _gammas_cached()
    w, v = np.linalg.eigh(chirality(G))
    return v[:, w > 0]


def weyl_basis(G=None):
    """32 x 16 matrix whose orthonormal columns span the +chirality subspace (the Weyl 16)."""
    return _weyl_basis_cached().copy()


def antisym_products(G, k):
    """Gamma^{[i1..ik]} = Gamma_{i1} ... Gamma_{ik} for i1 < ... < ik (antisymmetric automatically)."""
    out = []
    for idx in itertools.combinations(range(10), k):
        M = np.eye(32)
        for i in idx:
            M = M @ G[i]
        out.append(M)
    return out


def bilinear_blocks(k, sign=+1):
    """The 16 x 16 blocks P^T C Gamma^{[k]} P of all k-index products on the Weyl 16."""
    G = gammas()
    C = charge_conjugation(G, sign)
    P = weyl_basis(G)
    return [P.T @ C @ M @ P for M in antisym_products(G, k)]


def su5_singlet_126():
    """Mass matrix of the five-form Omega = wedge_k (e_{2k} + i e_{2k+1}), the SU(5)-singlet direction of the 126."""
    G = gammas()
    C = charge_conjugation(G, +1)
    P = weyl_basis(G)
    M = np.zeros((16, 16), complex)
    for bits in itertools.product([0, 1], repeat=5):
        idx = [2 * k + b for k, b in enumerate(bits)]
        coef = (1j) ** sum(bits)
        X = np.eye(32)
        for i in idx:
            X = X @ G[i]
        M += coef * (P.T @ C @ X @ P)
    return M


# ---- Lie algebra data ---------------------------------------------------------------------------------
def cartan_generators(G):
    """H_k = -(i/2) Gamma_{2k} Gamma_{2k+1}, k = 0..4: a Cartan subalgebra of so(10) (eigenvalues +-1/2)."""
    return [-0.5j * G[2 * k] @ G[2 * k + 1] for k in range(5)]


def so10_generators(G):
    """The 45 hermitian generators Sigma_ij = -(i/4)[Gamma_i, Gamma_j], i < j."""
    return [-0.25j * (G[i] @ G[j] - G[j] @ G[i]) for i in range(10) for j in range(i + 1, 10)]


def weyl_weight_basis(G=None):
    """(W, wts): 32 x 16 joint eigenvectors of the Cartan generators spanning the Weyl 16, and their 16 x 5 weights.

    The basis diagonalises a fixed generic combination of the H_k (seed 0); every column is asserted to be a joint
    eigenvector, the weights are (+-1/2)^5.
    """
    if G is None:
        G = gammas()
    P = weyl_basis(G)
    H = [P.conj().T @ h @ P for h in cartan_generators(G)]
    rng = np.random.default_rng(0)
    c = rng.normal(size=5)
    _, U = np.linalg.eigh(sum(ci * h for ci, h in zip(c, H, strict=True)))
    W = P @ U
    wts = np.array([[np.real(W[:, a].conj() @ h @ W[:, a]) for h in cartan_generators(G)] for a in range(16)])
    for a in range(16):
        for k, h in enumerate(cartan_generators(G)):
            assert np.allclose(h @ W[:, a], wts[a, k] * W[:, a])
    return W, wts


def root_vectors(G):
    """The 40 root vectors E_alpha of so(10) on the 32 with [H_k, E_alpha] = alpha_k E_alpha, alpha = +-e_i +- e_j.

    Returned as {alpha (tuple of ints): 32 x 32 matrix}.
    """
    Gp = [(G[2 * k] + 1j * G[2 * k + 1]) / 2 for k in range(5)]
    Gm = [(G[2 * k] - 1j * G[2 * k + 1]) / 2 for k in range(5)]
    H = cartan_generators(G)
    out = {}
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            out[tuple((1 if k == i else 0) - (1 if k == j else 0) for k in range(5))] = Gp[i] @ Gm[j]
    for i in range(5):
        for j in range(i + 1, 5):
            out[tuple((1 if k in (i, j) else 0) for k in range(5))] = Gp[i] @ Gp[j]
            out[tuple((-1 if k in (i, j) else 0) for k in range(5))] = Gm[i] @ Gm[j]
    for a, E in out.items():
        for k in range(5):
            assert np.allclose(H[k] @ E - E @ H[k], a[k] * E), (a, k)
    assert len(out) == 40
    return out


# ---- Pati-Salam: Spin(6) x Spin(4) = SU(4) x SU(2)_1 x SU(2)_2 ---------------------------------------
def pati_salam(G=None):
    """The 6 + 4 blocking of the gammas and the Pati-Salam data on the Weyl 16.

    Returns a dict with c6 = i Gamma_0..Gamma_5 and c4 = Gamma_6..Gamma_9 (32 x 32), the 16 x 8 blocks PiL / PiR of
    Spin(6) chirality +1 / -1 inside the Weyl basis, the 15 so(6) generators and the two commuting su(2) triplets
    J1, J2 of so(4) (all 32 x 32).
    """
    if G is None:
        G = gammas()
    P = weyl_basis(G)
    c6 = 1j * np.linalg.multi_dot(G[:6])
    c4 = np.linalg.multi_dot(G[6:])
    w, U = np.linalg.eigh(P.conj().T @ c6 @ P)
    so6 = [-0.25j * (G[i] @ G[j] - G[j] @ G[i]) for i in range(6) for j in range(i + 1, 6)]
    so4 = {(i, j): -0.25j * (G[i] @ G[j] - G[j] @ G[i]) for i in range(6, 10) for j in range(i + 1, 10)}
    J1 = [(so4[(6, 7)] + so4[(8, 9)]) / 2, (so4[(6, 8)] - so4[(7, 9)]) / 2, (so4[(6, 9)] + so4[(7, 8)]) / 2]
    J2 = [(so4[(6, 7)] - so4[(8, 9)]) / 2, (so4[(6, 8)] + so4[(7, 9)]) / 2, (so4[(6, 9)] - so4[(7, 8)]) / 2]
    return dict(P=P, c6=c6, c4=c4, w6=w, PiL=U[:, w > 0], PiR=U[:, w < 0], so6=so6, J1=J1, J2=J2)


def restrict(X, P, B=None):
    """B^dagger P^dagger X P B: a 32 x 32 operator restricted to the Weyl 16 (and to a block B of it)."""
    R = P.conj().T @ X @ P
    return R if B is None else B.conj().T @ R @ B


def casimir(gens, P, B):
    """sum_X (B^dagger P^dagger X P B)^2 for a list of generators."""
    return sum(restrict(X, P, B) @ restrict(X, P, B) for X in gens)
