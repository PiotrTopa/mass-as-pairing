"""Exact characters of D_n x A1^k (so(10) = D5, so(6) = D3, with sl(2) factors): weight multisets, tensor operations,
Schur functors and the Weyl-group multiplicity formula.

Weights are stored in doubled integer coordinates: the spinor 16 of D5 has weights (+-1, ..., +-1) (physically
(+-1/2)^5), a spin-1/2 sl(2) doublet has a = +-1. A character is a multiset {weight tuple: multiplicity}. The
multiplicity of the irreducible L(lambda) (lambda dominant) in a character chi_V is

    m_lambda(V) = sum_{w in W} eps(w) dim V_{lambda + rho - w(rho)},

with W = W(D_n) x (Z2)^k (permutations times even sign changes, and the sl(2) reflections) and eps = det w.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from fractions import Fraction

import numpy as np


# ---- root data ---------------------------------------------------------------------------------------
def dn_positive_roots(n=5):
    """Positive roots e_i +- e_j (i < j) of D_n in doubled coordinates (entries +-2)."""
    roots = []
    for i in range(n):
        for j in range(i + 1, n):
            for s in (+1, -1):
                r = [0] * n
                r[i] = 2
                r[j] = 2 * s
                roots.append(tuple(r))
    return roots


def positive_roots(k=0, n=5):
    """Positive roots of D_n x A1^k in doubled coordinates."""
    out = [r + (0,) * k for r in dn_positive_roots(n)]
    for a in range(k):
        r = [0] * (n + k)
        r[n + a] = 2
        out.append(tuple(r))
    return out


def rho(k=0, n=5):
    """Weyl vector, doubled: 2 (n-1, ..., 1, 0) for D_n and 1 for every A1."""
    return tuple(2 * (n - 1 - i) for i in range(n)) + (1,) * k


def weyl_group(k=0, n=5):
    """List of (matrix, sign) for W(D_n) x (Z2)^k acting on doubled weights (order n! 2^(n-1) 2^k)."""
    out = []
    for perm in itertools.permutations(range(n)):
        for signs in itertools.product((1, -1), repeat=n):
            if np.prod(signs) != 1:
                continue
            for a1 in itertools.product((1, -1), repeat=k):
                M = np.zeros((n + k, n + k), int)
                for i in range(n):
                    M[i, perm[i]] = signs[i]
                for a in range(k):
                    M[n + a, n + a] = a1[a]
                out.append((M, int(round(np.linalg.det(M)))))
    assert len(out) == math.factorial(n) * 2 ** (n - 1) * 2**k
    return out


def is_dominant(lam, k=0, n=5):
    lv = lam[:n]
    return (
        all(lv[i] >= lv[i + 1] for i in range(n - 2)) and lv[n - 2] >= abs(lv[n - 1]) and all(a >= 0 for a in lam[n:])
    )


def weyl_dim(lam, k=0, n=5):
    """Weyl dimension formula, exact (Fraction arithmetic)."""
    r = rho(k, n)
    num = Fraction(1)
    for a in positive_roots(k, n):
        num *= Fraction(
            sum((lv + rr) * aa for lv, rr, aa in zip(lam, r, a, strict=True)),
            sum(rr * aa for rr, aa in zip(r, a, strict=True)),
        )
    assert num.denominator == 1
    return int(num)


# ---- multisets ---------------------------------------------------------------------------------------
def ms_from_weights(wts):
    d = defaultdict(int)
    for w in wts:
        d[tuple(int(x) for x in w)] += 1
    return dict(d)


def ms_dim(A):
    return sum(A.values())


def ms_add(A, B):
    d = defaultdict(int, A)
    for w, m in B.items():
        d[w] += m
    return {w: m for w, m in d.items() if m != 0}


def ms_scale(A, c):
    return {w: c * m for w, m in A.items()}


def ms_tensor(A, B):
    d = defaultdict(int)
    for wa, ma in A.items():
        for wb, mb in B.items():
            d[tuple(x + y for x, y in zip(wa, wb, strict=True))] += ma * mb
    return dict(d)


def ms_dual(A):
    return {tuple(-x for x in w): m for w, m in A.items()}


def ms_adams(A, j):
    """Adams operation psi^j: weights scaled by j."""
    return {tuple(j * x for x in w): m for w, m in A.items()}


def ms_sym_powers(A, kmax):
    """h_k(A) = char Sym^k(A), k = 0..kmax (Newton: k h_k = sum_{j=1}^k p_j h_{k-j})."""
    zero = tuple(0 for _ in next(iter(A)))
    h = [{zero: 1}]
    for k in range(1, kmax + 1):
        acc = {}
        for j in range(1, k + 1):
            acc = ms_add(acc, ms_tensor(ms_adams(A, j), h[k - j]))
        assert all(m % k == 0 for m in acc.values())
        h.append({w: m // k for w, m in acc.items() if m != 0})
    return h


def ms_ext_powers(A, kmax):
    """e_k(A) = char Lambda^k(A), k = 0..kmax (Newton: k e_k = sum_{j=1}^k (-1)^(j-1) p_j e_{k-j})."""
    zero = tuple(0 for _ in next(iter(A)))
    e = [{zero: 1}]
    for k in range(1, kmax + 1):
        acc = {}
        for j in range(1, k + 1):
            acc = ms_add(acc, ms_scale(ms_tensor(ms_adams(A, j), e[k - j]), (-1) ** (j - 1)))
        assert all(m % k == 0 for m in acc.values())
        e.append({w: m // k for w, m in acc.items() if m != 0})
    return e


def ms_schur(A, lam):
    """Character of the Schur functor S_lambda(A) by Jacobi-Trudi: det(h_{lambda_i - i + j})."""
    n = len(lam)
    h = ms_sym_powers(A, max(lam) + n)
    zero = tuple(0 for _ in next(iter(A)))

    def H(k):
        return h[k] if k >= 0 else {}

    total = {}
    for perm in itertools.permutations(range(n)):
        sgn = np.linalg.det(np.eye(n)[list(perm)])
        term = {zero: 1}
        for i in range(n):
            term = ms_tensor(term, H(lam[i] - i + perm[i]))
            if not term:
                break
        total = ms_add(total, ms_scale(term, int(round(sgn))))
    return total


# ---- decomposition -----------------------------------------------------------------------------------
_WCACHE = {}


def _W(k, n):
    if (k, n) not in _WCACHE:
        _WCACHE[(k, n)] = weyl_group(k, n)
    return _WCACHE[(k, n)]


def multiplicity(A, lam, k=0, n=5):
    """m_lambda(A) = sum_w eps(w) dim A_{lambda + rho - w rho} for a dominant lambda (doubled coordinates)."""
    r = np.array(rho(k, n))
    lr = np.array(lam) + r
    tot = 0
    for M, sgn in _W(k, n):
        mu = tuple(int(x) for x in (lr - M @ r))
        tot += sgn * A.get(mu, 0)
    return tot


def decompose(A, k=0, n=5):
    """Full decomposition {lambda: m_lambda} of a character; asserts m >= 0 and sum m dim = dim A."""
    out = {}
    for w in A:
        if is_dominant(w, k, n):
            m = multiplicity(A, w, k, n)
            if m:
                out[w] = m
    assert all(m > 0 for m in out.values()), out
    assert sum(m * weyl_dim(lv, k, n) for lv, m in out.items()) == ms_dim(A), (out, ms_dim(A))
    return out


def invariants(A, k=0, n=5):
    """Multiplicity of the trivial representation."""
    return multiplicity(A, (0,) * (n + k), k, n)


def dims(dec, k=0, n=5):
    """Sorted list of the dimensions of the constituents (with multiplicity)."""
    return sorted(d for lv, m in dec.items() for d in [weyl_dim(lv, k, n)] * m)


def pretty(dec, k=0, n=5):
    """Readable decomposition: dimensions with highest weights (physical coordinates)."""
    items = sorted(dec.items(), key=lambda t: (weyl_dim(t[0], k, n), t[0]))
    return " ⊕ ".join(
        (f"{m}×" if m > 1 else "") + f"{weyl_dim(lv, k, n)}(" + ",".join(str(Fraction(x, 2)) for x in lv) + ")"
        for lv, m in items
    )


# ---- representations used by the claims ----------------------------------------------------------------
def spinor16(chirality=-1):
    """Weights of a Weyl 16 of D5: (+-1)^5 with product = chirality (doubled coordinates)."""
    return ms_from_weights([w for w in itertools.product((1, -1), repeat=5) if np.prod(w) == chirality])


def vector10():
    """Weights of the vector 10 of D5."""
    return ms_from_weights([tuple((2 * s if k == i else 0) for k in range(5)) for i in range(5) for s in (1, -1)])


def char126(chirality=-1):
    """Character of the 126 = Sym^2(16) - 10."""
    c = ms_add(ms_sym_powers(spinor16(chirality), 2)[2], ms_scale(vector10(), -1))
    assert ms_dim(c) == 126 and all(m > 0 for m in c.values())
    return c


def spinor4(chirality=+1):
    """Weights of a Weyl spinor of D3 = so(6) = su(4): the 4 (chirality +1) or the 4bar (-1)."""
    return ms_from_weights([w for w in itertools.product((1, -1), repeat=3) if np.prod(w) == chirality])


def branch_d5_to_ps(A):
    """Restrict a D5 character to D3 x A1_1 x A1_2 (Pati-Salam): (W1..W5) -> (W1, W2, W3; (W4+W5)/2, (W4-W5)/2)."""
    out = defaultdict(int)
    for w, m in A.items():
        assert (w[3] + w[4]) % 2 == 0
        out[(w[0], w[1], w[2], (w[3] + w[4]) // 2, (w[3] - w[4]) // 2)] += m
    return dict(out)


def weyl_lorentz(chirality=-1, dotted=False):
    """Character of psi in 16 (x) (1/2, 0) (or of psi-bar in 16bar (x) (0, 1/2) with ``dotted``) under D5 x A1 x A1."""
    s16 = spinor16(chirality)
    if not dotted:
        return ms_tensor({w + (0, 0): 1 for w in s16}, {(0,) * 5 + (1, 0): 1, (0,) * 5 + (-1, 0): 1})
    return ms_tensor({tuple(-x for x in w) + (0, 0): 1 for w in s16}, {(0,) * 5 + (0, 1): 1, (0,) * 5 + (0, -1): 1})
