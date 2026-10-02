"""The four real (reduced staggered) flavours and the complex doublet.

The doublet psi(x) in C^2 packs the four real flavours as chi^{2p} = Re psi_p, chi^{2p+1} = Im psi_p. With
u = (1, -i, 1, -i), p = (0, 0, 1, 1) and W[a, p(a)] = u_a, the real propagator is
G^{ab}(x, z) = <chi^a(x) chi^b(z)> = Re[u_a conj(u_b) S_{p(a) p(b)}(x, z)], S = D_c^{-1}; equivalently the real
4V x 4V operator is the realification M = Re[W D_c W^dagger].

GAMMA[a] = Re[W i tau_a W^dagger] (real antisymmetric 4 x 4): the Yukawa term is y sum_a sigma_a Gamma_a in the real
basis and n_a(x) = 1/4 tr[Gamma_a G(x, x)]. GAMMA2 is the second (anti-self-dual) triplet, commuting with GAMMA.
"""

from __future__ import annotations

import numpy as np

from .lattice import to_numpy

U = np.array([1.0, -1j, 1.0, -1j])
P = np.array([0, 0, 1, 1])
W = np.zeros((4, 2), complex)
for _a in range(4):
    W[_a, P[_a]] = U[_a]

TAU = [
    np.array([[0, 1], [1, 0]], complex),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]], complex),
]
GAMMA = np.stack([np.einsum("ap,pq,bq->ab", W, 1j * t, W.conj()).real for t in TAU])


def _other_triplet():
    """The three antisymmetric 4 x 4 real matrices orthogonal to GAMMA, normalised to Gamma'^2 = -1."""
    basis = []
    for i in range(4):
        for j in range(i + 1, 4):
            E = np.zeros((4, 4))
            E[i, j] = 1.0
            E[j, i] = -1.0
            basis.append(E)
    out = []
    for E in basis:
        R = E.copy()
        for Gm in list(GAMMA) + out:
            R = R - (R * Gm).sum() / (Gm * Gm).sum() * Gm
        if abs(R).max() > 1e-9:
            R = R / np.sqrt((R * R).sum() / 4.0)
            out.append(R)
    assert len(out) == 3
    return np.stack(out)


GAMMA2 = _other_triplet()
GAMMA6 = np.concatenate([GAMMA, GAMMA2])


def real_flavour_block(S):
    """G^{ab} (..., 4, 4), real, from doublet blocks S (..., 2, 2): G = Re[W S W^dagger]."""
    return np.einsum("ap,...pq,bq->...ab", W, S, W.conj()).real


def realify(Dc, V):
    """Dense 4V x 4V real form M = Re[W D_c W^dagger] of a dense 2V x 2V doublet matrix."""
    return np.einsum("ap,xpyq,bq->xayb", W, np.asarray(Dc).reshape(V, 2, V, 2), W.conj()).real.reshape(4 * V, 4 * V)


def pack(chi):
    """(..., 4) real -> (..., 2) complex packed doublet."""
    return chi[..., 0::2] + 1j * chi[..., 1::2]


def unpack(psi):
    """(..., 2) complex -> (..., 4) real."""
    psi = to_numpy(psi)
    out = np.empty(psi.shape[:-1] + (4,))
    out[..., 0::2] = psi.real
    out[..., 1::2] = psi.imag
    return out


def flavour_mask(spec):
    """(4,) mask from 1-based flavour indices ("4", [4], [1, 2]) or a 4-vector of weights."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.replace(",", " ").split()
    spec = [float(v) for v in spec]
    if len(spec) == 4 and not all(
        v in (1.0, 2.0, 3.0, 4.0) and float(v).is_integer() and spec.count(v) == 1 for v in spec
    ):
        return np.asarray(spec, float)
    m = np.zeros(4)
    for v in spec:
        assert float(v).is_integer() and 1 <= v <= 4, f"flavour index {v} not in 1..4"
        m[int(v) - 1] = 1.0
    return m
