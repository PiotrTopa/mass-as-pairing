"""Krylov solvers: CG, multishift CG and Lanczos matrix functions.

``A`` is a callable v -> A v (hermitian positive definite). Vectors are arrays of any shape (numpy or cupy).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .lattice import xp_of


def _vdot(a, b):
    xp = xp_of(a)
    return complex(xp.vdot(a, b))


def cg(A, b, tol=1e-10, maxit=10000, x0=None):
    """Conjugate gradient; stops at |r| <= tol |b|. Returns (x, iterations)."""
    xp = xp_of(b)
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    p = r.copy()
    rr = _vdot(r, r).real
    bnorm = float(xp.linalg.norm(b))
    for it in range(1, maxit + 1):
        Ap = A(p)
        a = rr / _vdot(p, Ap).real
        x = x + a * p
        r = r - a * Ap
        rr_new = _vdot(r, r).real
        if np.sqrt(rr_new) <= tol * bnorm:
            return x, it
        p = r + (rr_new / rr) * p
        rr = rr_new
    raise RuntimeError(f"CG did not converge in {maxit} iterations (res {np.sqrt(rr) / bnorm:.2e})")


def multishift_cg(A, b, shifts, tol=1e-10, maxit=10000):
    """Solve (A + s_l) x_l = b for all shifts s_l >= 0 in one Krylov space (Jegerlehner, hep-lat/9612014).

    Returns (X, iterations), X of shape (n_shifts,) + b.shape. The shifted residuals are zeta_l r; the iteration
    stops once max_l |zeta_l| |r| <= tol |b|.
    """
    xp = xp_of(b)
    shifts = np.asarray(shifts, float)
    ns = len(shifts)
    N = b.size
    bc = (ns, 1)
    bf = b.reshape(N)
    X = xp.zeros((ns, N), dtype=b.dtype)
    P = xp.multiply.outer(xp.ones(ns), bf)
    tmp = xp.empty_like(P)
    r = bf.copy()
    p = bf.copy()
    zeta_old = np.ones(ns)
    zeta = np.ones(ns)
    active = np.ones(ns, bool)
    alpha_old = 1.0
    beta_old = 0.0
    rr = _vdot(r, r).real
    bnorm = np.sqrt(rr)
    for it in range(1, maxit + 1):
        Ap = A(p.reshape(b.shape)).reshape(N)
        alpha = rr / _vdot(p, Ap).real
        den = alpha_old * zeta_old * (1.0 + alpha * shifts) + alpha * beta_old * (zeta_old - zeta)
        zeta_new = np.zeros(ns)
        ratio = np.zeros(ns)
        zeta_new[active] = (zeta * zeta_old * alpha_old)[active] / den[active]
        ratio[active] = zeta_new[active] / zeta[active]
        xp.multiply(P, xp.asarray((alpha * ratio).reshape(bc)), out=tmp)
        X += tmp
        r -= alpha * Ap
        rr_new = _vdot(r, r).real
        rnorm = np.sqrt(rr_new)
        active &= np.abs(zeta_new) * rnorm > tol * bnorm
        if not active.any():
            return X.reshape((ns,) + b.shape), it
        beta = rr_new / rr
        p = r + beta * p
        xp.multiply(P, xp.asarray((beta * ratio**2).reshape(bc)), out=P)
        P += xp.multiply.outer(xp.asarray(zeta_new), r)
        zeta_old, zeta = zeta, zeta_new
        alpha_old, beta_old = alpha, beta
        rr = rr_new
    raise RuntimeError(f"multishift CG did not converge in {maxit} iterations")


def lanczos(A, v, f, m_max=2000, tol=1e-10, check_every=8, want_vector=True, check_grow=256):
    """f(A) v and the quadratic form v^dagger f(A) v by Lanczos (Gauss quadrature).

    Returns dict(value, vec (or None), m, ritz_min, ritz_max). ``f`` acts on eigenvalues. Convergence: the relative
    change of the value over one check spacing is below ``tol`` twice in a row; the spacing is
    check_every * max(1, m // check_grow). No reorthogonalisation: the quadrature value is accurate to the tolerance,
    the reconstructed vector to about sqrt(tol), adequate for the heat bath.
    """
    xp = xp_of(v)
    vnorm = float(xp.linalg.norm(v))
    q = v / vnorm
    q_old = None
    alphas, betas = [], []
    Q = [q] if want_vector else None
    beta = 0.0
    last = None
    nconv = 0

    def estimate(mm):
        w, U = eigh_tridiagonal(np.asarray(alphas[:mm]), np.asarray(betas[: mm - 1]))
        fw = f(np.clip(w, 1e-300, None))
        return U @ (fw * U[0]), w

    for j in range(m_max):
        w = A(q)
        if beta:
            w = w - beta * q_old
        alpha = _vdot(q, w).real
        w = w - alpha * q
        alphas.append(alpha)
        beta = float(xp.linalg.norm(w))
        m = j + 1
        step = check_every * max(1, m // check_grow)
        if (m % step == 0) or beta < 1e-14 * vnorm or m == m_max:
            coef, ritz = estimate(m)
            val = vnorm**2 * coef[0]
            if last is not None and abs(val - last) <= tol * abs(val):
                nconv += 1
            else:
                nconv = 0
            last = val
            if nconv >= 2 or beta < 1e-14 * vnorm or m == m_max:
                if m == m_max and nconv < 2 and beta >= 1e-14 * vnorm:
                    raise RuntimeError("Lanczos did not converge")
                vec = None
                if want_vector:
                    vec = xp.zeros_like(v)
                    for k in range(m):
                        vec = vec + (vnorm * coef[k]) * Q[k]
                return dict(value=val, vec=vec, m=m, ritz_min=float(ritz[0]), ritz_max=float(ritz[-1]))
        betas.append(beta)
        q_old, q = q, w / beta
        if Q is not None:
            Q.append(q)
