"""Sign and magnitude of the Pfaffian of the real flavour-selective operator M(h) = M(0) - h C (x) diag(mask).

Routes: pivoted Parlett-Reid elimination (``pfaffian_logpr``), LAPACK Householder tridiagonalisation
(``pfaffian_householder``: T = Q^T A Q, Pf A = det Q prod_k T[2k, 2k+1]) and the Schur route through the sourced-flavour
block of the h = 0 propagator (``pf_sign_config``): Pf M(h) / Pf M(0) = Pf(G_F - h G_F C_F G_F) / Pf(G_F), one dense
inverse
per configuration for every h. The pencil (``sign_flips``): the sign changes at h = 1/lambda for the real positive
eigenvalues lambda of G_F C_F (each doubly degenerate, hence a simple root of the Pfaffian).
"""

from __future__ import annotations

import numpy as np

from .flavour import P, U, realify
from .patterns import source_pattern


def pfaffian_logpr(A):
    """(sign, log|Pf|) of a real antisymmetric matrix by pivoted Parlett–Reid elimination (reference; O(n³) numpy)."""
    A = np.array(A, float)
    n = A.shape[0]
    if n % 2:
        return 0.0, -np.inf
    sign, logabs = 1.0, 0.0
    for k in range(0, n - 1, 2):
        col = np.abs(A[k + 1 :, k])
        p = k + 1 + int(np.argmax(col))
        if col[p - k - 1] == 0.0:
            return 0.0, -np.inf
        if p != k + 1:
            A[[k + 1, p], :] = A[[p, k + 1], :]
            A[:, [k + 1, p]] = A[:, [p, k + 1]]
            sign = -sign
        piv = A[k, k + 1]
        sign *= np.sign(piv)
        logabs += np.log(abs(piv))
        if k + 2 < n:
            tau = A[k, k + 2 :] / piv
            A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1]) - np.outer(A[k + 2 :, k + 1], tau)
    return float(sign), float(logabs)


def pfaffian_householder(A, drop_detq=False):
    """(sign, log|Pf|) of a real antisymmetric matrix by LAPACK Householder tridiagonalisation (dgehrd):
    T = QᵀAQ tridiagonal antisymmetric, Pf A = det(Q) Π_k T[2k, 2k+1], det Q = Π (τ_i ≠ 0 ? −1 : 1).
    drop_detq=True omits the reflector signs (a control: the sign is then wrong)."""
    from scipy.linalg import lapack

    A = np.asarray(A, float)
    n = A.shape[0]
    if n % 2:
        return 0.0, -np.inf
    ht, tau, info = lapack.dgehrd(A, overwrite_a=False)
    assert info == 0, info
    detq = 1.0 if drop_detq else float(np.prod(np.where(tau != 0.0, -1.0, 1.0)))
    sup = 0.5 * (np.diag(ht, 1) - np.diag(ht, -1))
    vals = sup[0::2]
    if np.any(vals == 0.0):
        return 0.0, -np.inf
    return float(detq * np.prod(np.sign(vals))), float(np.sum(np.log(np.abs(vals))))


def pf_ratio_schur(GF, CF, hs, method="householder"):
    """Pf M(h) / Pf M(0) for the h values `hs` from the sourced-flavour block G_F of the h = 0 propagator and the source
    matrix C_F = C₀ ⊗ P_F on that block: returns (signs, logratios) with
        Pf M(h)/Pf M(0) = Pf(G_F − h G_F C_F G_F) / Pf(G_F)."""
    pf = pfaffian_householder if method == "householder" else pfaffian_logpr
    GF = np.asarray(GF, float)
    s0, l0 = pf(GF)
    GCG = GF @ (CF @ GF)
    signs, logs = [], []
    for h in hs:
        s, l = pf(GF - h * GCG)
        signs.append(s * s0)
        logs.append(l - l0)
    return np.asarray(signs), np.asarray(logs)


def flavour_block_G(S, mask):
    """G_F (|F|V × |F|V, (site, flavour) ordering) from the dense doublet inverse S (V,2,V,2), F = {a : mask_a ≠ 0}."""
    V = S.shape[0]
    F = [a for a in range(4) if mask[a] != 0.0]
    nf = len(F)
    GF = np.empty((V, nf, V, nf))
    for i, a in enumerate(F):
        for j, b in enumerate(F):  # G^{ab}(x,z) = Re[u_a ū_b S_{p(a)p(b)}(x,z)]
            GF[:, i, :, j] = (U[a] * np.conj(U[b]) * S[:, P[a], :, P[b]]).real
    return GF.reshape(V * nf, V * nf), F


def sign_flips(GF, CF, rel_tol=1e-8, lam_tol=1e-6):
    """The h > 0 at which Pf M(h) changes sign, from the pencil: Pf(G_F − hG_F C_F G_F) ∝ Pf(S₀ − hC_F) (S₀ = G_F⁻¹)
    vanishes iff
    1/h is an eigenvalue of G_F C_F; the eigenvalues of a product of two antisymmetric matrices are doubly
    degenerate, so every
    real positive eigenvalue λ (counted once) is a simple root of the Pfaffian: sign Pf M(h) = (−1)^{#{λ real > 0 :
    1/λ < h}}.
    Returns the sorted flip points 1/λ (each real positive λ once).  Real: |Im λ| ≤ rel_tol·max|λ|; λ > lam_tol·max|λ|.
    """
    w = np.linalg.eigvals(GF @ CF)
    scale = np.abs(w).max()
    real = np.sort(w[(np.abs(w.imag) <= rel_tol * scale) & (w.real > lam_tol * scale)].real)[::-1]
    return 1.0 / real[0::2] if len(real) else np.zeros(0)


def pf_sign_config(model, fields, hs, mask=None, method="householder", S=None, flips=False):
    """Sign and log-magnitude of Pf M(h) relative to Pf M(0) = det D_c(σ, s) > 0 for the h values `hs` on one
    configuration:
    dict(sign (n_h,), logratio (n_h,), logdet0, sign0 [, flips (the sign-change points in h, `sign_flips`)]): one
    dense inverse
    of the 2V complex D_c (numpy) — the Schur route."""
    D = model.D
    mask = D.mask if mask is None else np.asarray(mask, float)
    V = model.lat.V
    if S is None:
        Dc = D.dense(fields)
        s0, ld0 = np.linalg.slogdet(Dc)
        S = np.linalg.inv(Dc).reshape(V, 2, V, 2)
    else:
        s0, ld0 = 1.0, float("nan")
    GF, F = flavour_block_G(S, mask)
    C = D.C if D.selective else source_pattern(model.lat)
    CF = np.kron(C.toarray(), np.diag(mask[F]))
    signs, logs = pf_ratio_schur(GF, CF, hs, method=method)
    out = dict(sign=signs, logratio=logs, logdet0=float(ld0), sign0=float(np.real(s0)))
    if flips:
        out["flips"] = sign_flips(GF, CF)
    return out


def pf_sign_direct(model, fields, h=None, mask=None):
    """(sign, log|Pf|) of the dense 4V×4V M(h) by Householder tridiagonalisation (the independent route; L ≤ 6)."""
    D = model.D
    M = realify(D.dense(fields), model.lat.V)
    h = D.h_sel if h is None else float(h)
    mask = D.mask if mask is None else np.asarray(mask, float)
    if h and np.any(mask):
        C = D.C if D.selective else source_pattern(model.lat)
        M -= h * np.kron(C.toarray(), np.diag(mask))
    return pfaffian_householder(M)
