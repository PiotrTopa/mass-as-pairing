"""Fermion-bag (interaction-expansion) weights of the four-flavour model.

* Plaquette terms (``bag_weights``): Z = <exp(g sum_Q T_Q)>_0; the weight of an insertion configuration {k_Q} is
  w = <prod_Q T_Q^{k_Q} / k_Q!>_0, computed exactly by Wick's theorem (``staggered.WickCache``).
* Link monomials (``link_pfaffians``, ``kmonomial_table``): for one real flavour Pf(M - sum_l a_l A_l) / Pf(M) is
  multilinear in the link variables, with coefficients the oriented single-flavour dimer weights; for four identical
  flavours the generating function is its fourth power, so <prod_l Kt_l^{m_l} / m_l!>_0 is the coefficient of
  prod_l a_l^{m_l} in pf(a)^4.
* Site bags of the eps-model with flavour-dependent operators M^a = K - h_a C (``site_bag_weights``,
  ``order_coefficients``): Z(U) = int exp(-1/2 sum_a chi^a M^a chi^a) prod_x (1 + U chi^1 chi^2 chi^3 chi^4 (x));
  the term of a site set B is the Gaussian integral over the complement, Z = sum_B U^|B| prod_a Pf(M^a_{Bbar}), and
  by the Jacobi identity Pf(M_{Bbar}) = +- Pf(M) Pf(G[B]) (G = M^-1) with the same position sign for every flavour.
"""

from __future__ import annotations

import math

import numpy as np

from ..pfaffian import pfaffian_householder
from .grassmann import Grass, pfaffian
from .staggered import WickCache


# ---- plaquette-term bags --------------------------------------------------------------------------------
def enumerate_configs(nq, kmax, ksum):
    """All k-vectors of length nq with entries <= kmax and 1 <= sum <= ksum."""

    def rec(i, rem):
        if i == nq:
            yield ()
            return
        for k in range(0, min(kmax, rem) + 1):
            for rest in rec(i + 1, rem - k):
                yield (k,) + rest

    for c in rec(0, ksum):
        if sum(c) > 0:
            yield c


def bag_weights(M, quads, term, ksum, kmax=4, normalise=True):
    """[(config, weight)] for every insertion configuration with sum k <= ksum and |weight| >= 1e-9.

    ``term(q)`` gives T_Q as a Grass element; with ``normalise`` T_Q is multiplied by sign <T_Q>_0 (the
    shift-symmetric orientation)."""
    W = WickCache(M)
    T = {}
    for q in quads:
        t = term(q)
        if normalise:
            s = np.sign(W.expect(t))
            assert s != 0
            t = t * float(s)
        T[q] = t
    Tk = {(q, k): T[q] ** k * (1.0 / math.factorial(k)) for q in quads for k in range(1, kmax + 1)}
    out = []
    for c in enumerate_configs(len(quads), kmax, ksum):
        poly = None
        for q, k in zip(quads, c, strict=True):
            if k == 0:
                continue
            poly = Tk[(q, k)] if poly is None else poly * Tk[(q, k)]
            if not poly.t:
                break
        if poly is None or not poly.t:
            continue
        w = W.expect(poly)
        if abs(w) < 1e-9:
            continue
        out.append((c, w))
    return out


# ---- link monomials ------------------------------------------------------------------------------------
def kmonomial_weight(W, Kl, mdict):
    """<prod_l Kt_l^{m_l} / m_l!>_0 for oriented link bilinears Kt_l (Grass) by Wick's theorem."""
    poly = Grass.one()
    for link, m in mdict.items():
        poly = poly * (Kl[link] ** m) * (1.0 / math.factorial(m))
        if not poly.t:
            return 0.0
    return W.expect(poly)


def link_pfaffians(G, links, signs):
    """{mask: prod_{l in T} s_l Pf(G[sites(T)])} over the subsets T of ``links`` (0 for overlapping subsets)."""
    n = len(links)
    out = {0: 1.0}
    for mask in range(1, 1 << n):
        sel = [links[i] for i in range(n) if mask >> i & 1]
        sites_ = [s for link in sel for s in link]
        if len(set(sites_)) < len(sites_):
            out[mask] = 0.0
            continue
        v = pfaffian(G[np.ix_(sites_, sites_)]).real
        for i in range(n):
            if mask >> i & 1:
                v *= signs[i]
        out[mask] = v
    return out


def kmonomial_table(pf, n, power):
    """All coefficients of pf(a)^power for a multilinear pf given as {mask: coeff}: array (power+1,)*n by exponents."""
    shape = (power + 1,) * n
    P = np.zeros(shape)
    P[(0,) * n] = 1.0
    for _ in range(power):
        Q = np.zeros(shape)
        for mask, c in pf.items():
            if c == 0.0:
                continue
            src = tuple(slice(0, power + 1 - (mask >> i & 1)) for i in range(n))
            dst = tuple(slice(mask >> i & 1, power + 1) for i in range(n))
            Q[dst] += c * P[src]
        P = Q
    return P


# ---- site bags of the eps-model ------------------------------------------------------------------------------
def pf_small(A):
    """Pfaffian of a real antisymmetric matrix: closed forms for n <= 4, Householder sign * exp(log) above."""
    n = A.shape[0]
    if n == 0:
        return 1.0
    if n == 2:
        return A[0, 1]
    if n == 4:
        return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]
    s, lg = pfaffian_householder(A)
    return s * np.exp(lg)


def site_bag_weights(Ms, subsets):
    """(propagators G^a, {B: (prod_a Pf(G^a[B]), [Pf(G^a[B])]_a)})
    in the Jacobi form (the constant prod_a Pf M^a is dropped)."""
    Gs = [np.linalg.inv(M) for M in Ms]
    out = {}
    for B in subsets:
        idx = list(B)
        fac = [pf_small(G[np.ix_(idx, idx)]) for G in Gs]
        out[B] = (float(np.prod(fac)), fac)
    return Gs, out


def wick_full(Gs, B):
    """Pf of the 4|B| x 4|B| propagator in the vertex ordering (x1: a = 1..4, x2: a = 1..4, ...)."""
    n = 4 * len(B)
    A = np.zeros((n, n))
    for i, x in enumerate(B):
        for j, y in enumerate(B):
            for a in range(4):
                A[4 * i + a, 4 * j + a] = Gs[a][x, y]
    return pf_small(A)


def order_coefficients(Ms, Cprime_by_flavour, subsets, eps=1e-6):
    """(N_k, Z_k) summed over the site sets of each size k.

    Z_k = sum_{|B|=k} prod_a Pf(M^a_{Bbar}); N_k = sum_{|B|=k} d/dh' prod_a Pf(M^a_{Bbar} - h' C'^a_{Bbar}) at h' = 0,
    the numerator of <sum_a 1/2 chi^a C'^a chi^a> at order U^k. Per flavour p_a = Pf(M^a_{Bbar}) and
    d_a = -1/2 p_a tr[(M^a_{Bbar})^-1 C'] when the restriction is regular; a singular restriction has p_a = 0 and its
    derivative is the complex step Im Pf(M^a_{Bbar} - i eps C') / eps. N += sum_a d_a prod_{b != a} p_b.
    """
    V = Ms[0].shape[0]
    Nk, Zk = {}, {}
    all_idx = np.arange(V)
    for B in subsets:
        k = len(B)
        comp = np.setdiff1d(all_idx, B)
        p, d = [], []
        for a in range(4):
            A = Ms[a][np.ix_(comp, comp)]
            Cp = Cprime_by_flavour[a]
            sv_min = np.linalg.svd(A, compute_uv=False)[-1] if A.shape[0] else 1.0
            if sv_min > 1e-9:
                if A.shape[0] > 4:
                    s_, l_ = pfaffian_householder(A)
                else:
                    s_, l_ = (np.sign(pf_small(A)) or 1.0, np.log(abs(pf_small(A)) + 1e-300))
                pa = s_ * np.exp(l_)
                if Cp is None:
                    da = 0.0
                else:
                    Ainv = np.linalg.solve(A, np.eye(A.shape[0]))
                    da = -0.5 * pa * np.einsum("ij,ji->", Ainv, Cp[np.ix_(comp, comp)])
            else:
                pa = 0.0
                da = 0.0 if Cp is None else float(np.imag(pfaffian(A - 1j * eps * Cp[np.ix_(comp, comp)])) / eps)
            p.append(pa)
            d.append(da)
        Zk[k] = Zk.get(k, 0.0) + float(np.prod(p))
        val = 0.0
        for a in range(4):
            val += d[a] * float(np.prod([p[b] for b in range(4) if b != a]))
        Nk[k] = Nk.get(k, 0.0) + val
    return Nk, Zk
