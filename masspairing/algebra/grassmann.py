"""Grassmann algebra tools: sparse polynomials over up to 64 generators, the ``Grass`` element class, Pfaffians,
Berezin integration and Wick's theorem.

Two representations are used:

* dict polynomials {mask: coeff} (functions ``mul``, ``add``, ``bilinear``, ``derivation``, ...): a monomial is an int
  whose set bits are its generators in increasing order (the canonical normal form); products carry the exact
  reordering sign, so a polynomial vanishes identically iff every coefficient vanishes. ``mul`` is vectorised and
  is used for the Spin(10) quartics and octics on 32 or 64 generators.
* ``Grass``: the same normal form wrapped as a class with +, -, *, ** and the nilpotent exponential, used for the
  lattice polynomials of the four-flavour model (generators indexed flavour-major, see ``staggered.fidx``).

Conventions: int dchi_n ... dchi_1 chi_1 ... chi_n = 1; <chi_{i1} ... chi_{i2k}>_0 = Pf(G[i1..i2k]) with G = M^{-1}
for the Gaussian weight exp(-1/2 chi^T M chi).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps


def _bits(mask):
    out = []
    i = 0
    while mask:
        if mask & 1:
            out.append(i)
        mask >>= 1
        i += 1
    return out


# ---- dict polynomials ----------------------------------------------------------------------------------
def clean(p, tol=1e-13):
    """Drop coefficients below tol times the largest one."""
    scale = max((abs(c) for c in p.values()), default=0.0)
    return {m: c for m, c in p.items() if abs(c) > tol * max(scale, 1e-300)}


def add(p, q, cq=1.0):
    """p + cq q."""
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + cq * c
    return out


def scale(p, c):
    return {m: c * v for m, v in p.items()}


def norm(p):
    return float(np.sqrt(sum(abs(c) ** 2 for c in p.values())))


def mul(p, q, chunk=2_000_000):
    """Product p q (vectorised). The sign of psi_A psi_B for disjoint sorted A, B is (-1)^#{(a, b): a > b}."""
    if not p or not q:
        return {}
    pm = np.array(list(p.keys()), dtype=np.uint64)
    pc = np.array(list(p.values()), dtype=complex)
    qm = np.array(list(q.keys()), dtype=np.uint64)
    qc = np.array(list(q.values()), dtype=complex)
    qbits = [_bits(int(m)) for m in qm]
    nb = max(len(b) for b in qbits)
    qpos = np.full((len(qm), nb), 63, dtype=np.uint64)  # 63: shift by 64 below, emulated as 0
    for j, b in enumerate(qbits):
        qpos[j, : len(b)] = b
    out = {}
    step = max(1, chunk // len(qm))
    for s in range(0, len(pm), step):
        A = pm[s : s + step, None]
        C = pc[s : s + step, None] * qc[None, :]
        ok = (A & qm[None, :]) == 0
        inv = np.zeros(ok.shape, dtype=np.int64)
        for k in range(nb):
            sh = qpos[:, k][None, :] + np.uint64(1)  # count the elements of A above b
            shifted = np.where(sh >= 64, np.uint64(0), A >> np.minimum(sh, np.uint64(63)))
            inv += np.bitwise_count(shifted).astype(np.int64)
        sign = np.where(inv % 2 == 0, 1.0, -1.0)
        masks = (A | qm[None, :])[ok]
        vals = (C * sign)[ok]
        um, inv_idx = np.unique(masks, return_inverse=True)
        acc = np.zeros(len(um), dtype=complex)
        np.add.at(acc, inv_idx, vals)
        for m, v in zip(um.tolist(), acc.tolist(), strict=True):
            out[m] = out.get(m, 0) + v
    return clean(out)


def bilinear(M, eps=True, offset=0):
    """Lorentz-scalar bilinear psi^T M psi of a Weyl spinor psi_{a alpha} (generator 2a + alpha + offset):
    sum_ab M_ab eps^{alpha beta} psi_{a alpha} psi_{b beta}, eps^{01} = +1. With eps=False the four components
    (alpha, beta) -> sum_ab M_ab psi_{a alpha} psi_{b beta} are returned."""
    n = M.shape[0]
    polys = {(al, be): {} for al in (0, 1) for be in (0, 1)}
    for a in range(n):
        for b in range(n):
            if abs(M[a, b]) < 1e-14:
                continue
            for al in (0, 1):
                for be in (0, 1):
                    i, j = offset + 2 * a + al, offset + 2 * b + be
                    if i == j:
                        continue
                    sgn = 1.0 if i < j else -1.0
                    m = (1 << i) | (1 << j)
                    polys[(al, be)][m] = polys[(al, be)].get(m, 0) + sgn * M[a, b]
    if not eps:
        return {k: clean(v) for k, v in polys.items()}
    return clean(add(polys[(0, 1)], polys[(1, 0)], -1.0))


def rank_of(polys, tol=1e-9):
    """(rank, singular values) of a list of polynomials as coefficient vectors."""
    keys = sorted(set().union(*[p.keys() for p in polys]))
    idx = {k: i for i, k in enumerate(keys)}
    A = np.zeros((len(polys), len(keys)), complex)
    for r, p in enumerate(polys):
        for m, c in p.items():
            A[r, idx[m]] = c
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol * s[0]).sum()), s


def coefficient_vectors(polys):
    """The polynomials as rows of a dense coefficient matrix over the union of their monomials."""
    keys = sorted(set().union(*[p.keys() for p in polys]))
    idx = {k: i for i, k in enumerate(keys)}
    out = []
    for p in polys:
        v = np.zeros(len(keys), complex)
        for m, c in p.items():
            v[idx[m]] = c
        out.append(v)
    return out


def derivation(poly, X, tol=0.0):
    """D_X f for the derivation psi_i -> sum_j X_ij psi_j of the exterior algebra (X: ngen x ngen).
    f is invariant under the one-parameter group generated by X iff D_X f = 0."""
    ngen = X.shape[0]
    nz = {i: [(j, X[i, j]) for j in range(ngen) if abs(X[i, j]) > 1e-14] for i in range(ngen)}
    out = {}
    for mask, c in poly.items():
        S = _bits(mask)
        for i in S:
            others = [x for x in S if x != i]
            for j, v in nz[i]:
                if j in others:
                    continue
                between = sum(1 for x in others if (i < x < j) or (j < x < i))
                m = sum(1 << x for x in others) | (1 << j)
                out[m] = out.get(m, 0) + c * v * (-1.0 if between % 2 else 1.0)
    return clean(out, tol)


def derivation_matrix(X, monos):
    """Sparse matrix of D_X on the span of the monomials ``monos`` (index tuples); rows are the output monomials.

    Returns (matrix, {output mask: row})."""
    ngen = X.shape[0]
    nz = {i: [(j, X[i, j]) for j in range(ngen) if abs(X[i, j]) > 1e-14] for i in range(ngen)}
    rows, cols, vals, out_idx = [], [], [], {}
    for col, S in enumerate(monos):
        for i in S:
            others = [x for x in S if x != i]
            for j, v in nz[i]:
                if j in others:
                    continue
                between = sum(1 for x in others if (i < x < j) or (j < x < i))
                mask = sum(1 << x for x in others) | (1 << j)
                r = out_idx.setdefault(mask, len(out_idx))
                rows.append(r)
                cols.append(col)
                vals.append((-1.0 if between % 2 else 1.0) * v)
    return sps.csr_matrix((vals, (rows, cols)), shape=(len(out_idx), len(monos))), out_idx


def current_current(Ts, n=16, offset_bar=32):
    """Lorentz-scalar current-current quartic sum_A eps^{ad bd} eps^{a b} J^A_{ad a} J^{A dagger}_{bd b}.

    J^A_{ad a} = psibar_{a ad} T^A_ab psi_{b a}, J^{A dagger} built with (T^A)^dagger; {T^A} orthonormal under
    Tr(A^dagger B), so that sum_A T^A (x) (T^A)^dagger is the invariant projector.
    """
    out = {}
    eps = {(0, 1): 1.0, (1, 0): -1.0}
    for T in Ts:

        def cur(M):
            J = {}
            for ad in (0, 1):
                for al in (0, 1):
                    p = {}
                    for a in range(n):
                        for b in range(n):
                            if abs(M[a, b]) < 1e-14:
                                continue
                            i, j = offset_bar + 2 * a + ad, 2 * b + al
                            m = (1 << i) | (1 << j)
                            p[m] = p.get(m, 0) + (1.0 if i < j else -1.0) * M[a, b]
                    J[(ad, al)] = p
            return J

        J1, J2 = cur(T), cur(T.conj().T)
        for (ad, bd), e1 in eps.items():
            for (al, be), e2 in eps.items():
                out = add(out, mul(J1[(ad, al)], J2[(bd, be)]), e1 * e2)
    return clean(out)


# ---- Pfaffian ------------------------------------------------------------------------------------------
def pfaffian(A):
    """Pfaffian of an antisymmetric (complex) matrix by pivoted Parlett-Reid elimination."""
    A = np.array(A, dtype=complex)
    n = A.shape[0]
    if n == 0:
        return 1.0
    if n % 2:
        return 0.0
    pf = 1.0
    scl = max(np.abs(A).max(), 1e-300)
    for k in range(0, n - 1, 2):
        col = np.abs(A[k + 1 :, k])
        if col.max() < 1e-14 * scl:
            return 0.0
        p = k + 1 + int(np.argmax(col))
        if p != k + 1:
            A[[k + 1, p], :] = A[[p, k + 1], :]
            A[:, [k + 1, p]] = A[:, [p, k + 1]]
            pf = -pf
        pf *= A[k, k + 1]
        if k + 2 < n:
            tau = A[k, k + 2 :] / A[k, k + 1]
            A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1]) - np.outer(A[k + 2 :, k + 1], tau)
    return pf


# ---- Grass elements --------------------------------------------------------------------------------------
def _reorder_sign(mask_a, mask_b):
    """Sign of moving the sorted monomial b to the right of the sorted monomial a: (-1)^#{(i in a, j in b): i > j}."""
    s = 0
    b = mask_b
    while b:
        j = b & -b
        s += bin(mask_a & ~((j << 1) - 1)).count("1")
        b ^= j
    return -1 if s & 1 else 1


class Grass:
    """Element of the exterior algebra: {mask: coeff} in the canonical normal form."""

    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = dict(t) if t else {}

    @staticmethod
    def gen(i):
        return Grass({1 << i: 1.0})

    @staticmethod
    def one():
        return Grass({0: 1.0})

    def __add__(self, o):
        t = dict(self.t)
        for m, c in o.t.items():
            t[m] = t.get(m, 0.0) + c
        return Grass({m: c for m, c in t.items() if c != 0})

    def __sub__(self, o):
        return self + o * (-1.0)

    def __mul__(self, o):
        if not isinstance(o, Grass):
            return Grass({m: c * o for m, c in self.t.items()})
        t = {}
        for ma, ca in self.t.items():
            for mb, cb in o.t.items():
                if ma & mb:
                    continue
                s = _reorder_sign(ma, mb)
                m = ma | mb
                t[m] = t.get(m, 0.0) + s * ca * cb
        return Grass({m: c for m, c in t.items() if c != 0})

    __rmul__ = __mul__

    def __pow__(self, k):
        r = Grass.one()
        for _ in range(k):
            r = r * self
        return r

    def coeff(self, mask):
        return self.t.get(mask, 0.0)

    def maxabs(self):
        """Largest absolute coefficient (0 for the zero element)."""
        return max([abs(c) for c in self.t.values()] + [0.0])

    def exp_nilpotent(self, kmax):
        """exp(self) for a nilpotent element, summed to order kmax (or until the terms vanish)."""
        r = Grass.one()
        term = Grass.one()
        for k in range(1, kmax + 1):
            term = term * self * (1.0 / k)
            if not term.t:
                break
            r = r + term
        return r


def berezin_integral(poly, n):
    """int dchi_n ... dchi_1 poly over n generators (top coefficient)."""
    return poly.coeff((1 << n) - 1)


def wick(G, indices):
    """<chi_{i1} ... chi_{i2k}>_0 = Pf(G[I, I]) for the ordered index list."""
    if len(indices) % 2:
        return 0.0
    if len(set(indices)) < len(indices):
        return 0.0
    idx = list(indices)
    return pfaffian(G[np.ix_(idx, idx)])


def substitute(poly, V, n):
    """The linear substitution chi_i -> sum_j V[i, j] chi_j applied to a Grass over n generators."""
    images = []
    for i in range(n):
        images.append(Grass({1 << j: V[i, j] for j in range(n) if abs(V[i, j]) > 1e-15}))
    out = Grass()
    for m, c in poly.t.items():
        term = Grass.one() * c
        for i in range(n):
            if m >> i & 1:
                term = term * images[i]
        out = out + term
    return out
