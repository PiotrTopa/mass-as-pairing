"""The four-flavour reduced staggered model as exact Grassmann algebra on small lattices.

Euclidean action on an L^d hypercubic lattice (L even), one real Grassmann variable chi^a(x) per site and flavour
(a = 1..4):

    S_0 = 1/2 sum_{x,y,a} chi^a(x) M_xy chi^a(y),
    M_xy = 1/2 sum_mu eta_mu(x) [s_+ delta_{y,x+mu} - s_- delta_{y,x-mu}],

eta_mu(x) = (-1)^(x_1 + ... + x_{mu-1}), s_+- = -1 when the hop crosses an antiperiodic boundary. M is real
antisymmetric and connects opposite parities. eps(x) = (-1)^(sum x); even sites carry the 4 of SU(4), odd sites the
4bar (chi_e -> U chi_e, chi_o -> U* chi_o); U(1)_eps: chi(x) -> exp(i alpha eps(x)) chi(x).

Interactions on a plaquette quad Q = (x, x'; y, y') (x even, x' = x + mu + nu, y = x + mu, y' = x + nu):

    eps-term   chi^1 chi^2 chi^3 chi^4 (x)                              (Lambda^2 = 6 channel, on site)
    Sym^2 term T_Q = sum_ab Phibar^{ab}(y, y') Phi^{ab}(x, x')         (10 channel)
    Lambda^2   T_6 = sum_ab Lambdabar^{ab}(y, y') Lambda^{ab}(x, x')   (its two-site 6 partner)

Phi^{ab}(x, x') = chi^a(x) chi^b(x') + chi^b(x) chi^a(x') (Lambda with the minus sign). Fierz form, with the
SU(4)-singlet link bilinear K_uv = sum_a chi^a(u) chi^a(v):

    T_Q = 2 [K_{xy'} K_{x'y} - K_{xy} K_{x'y'}],   T_6 = -2 [K_{xy'} K_{x'y} + K_{xy} K_{x'y'}].

Generators are indexed flavour-major: chi^a(site) is generator a * nsites + site.
"""

from __future__ import annotations

import itertools

import numpy as np

from .grassmann import Grass, pfaffian

NF = 4


# ---- lattice -------------------------------------------------------------------------------------------
def sites(L, d):
    return list(itertools.product(range(L), repeat=d))


def parity(x):
    return (-1) ** (sum(x) % 2)


def eta(x, mu):
    return (-1) ** (sum(x[:mu]) % 2)


def staggered_matrix(L, d, bc="antiperiodic"):
    """(M, sites, index) for the real antisymmetric N x N single-flavour operator, N = L^d (bc in every direction)."""
    S = sites(L, d)
    idx = {x: i for i, x in enumerate(S)}
    N = len(S)
    M = np.zeros((N, N))
    for x in S:
        for mu in range(d):
            for sgn in (+1, -1):
                y = list(x)
                y[mu] += sgn
                cross = y[mu] < 0 or y[mu] >= L
                y[mu] %= L
                s = -1.0 if (cross and bc == "antiperiodic") else 1.0
                M[idx[x], idx[tuple(y)]] += 0.5 * eta(x, mu) * sgn * s
    assert np.allclose(M, -M.T)
    return M, S, idx


def _lattice(L, d, S, idx):
    if S is None:
        S = sites(L, d)
        idx = {x: i for i, x in enumerate(S)}
    return S, idx


def plaquette_quads(L, d, S=None, idx=None):
    """Distinct quads (x, x', y, y') with x even, x' = x + mu + nu, y = x + mu, y' = x + nu (mu < nu), as site
    indices."""
    S, idx = _lattice(L, d, S, idx)
    seen = set()
    quads = []
    for x in S:
        if parity(x) != 1:
            continue
        for mu, nu in itertools.combinations(range(d), 2):

            def sh(z, *ds):
                z = list(z)
                for dd in ds:
                    z[dd] = (z[dd] + 1) % L
                return tuple(z)

            q = (idx[x], idx[sh(x, mu, nu)], idx[sh(x, mu)], idx[sh(x, nu)])
            key = frozenset(q)
            if key in seen or len(key) < 4:
                continue
            seen.add(key)
            quads.append(q)
    return quads


def line_quads(L, d, S=None, idx=None):
    """The collinear geometry: x, x + mu, x + 2mu, x + 3mu with the even pair (x, x + 2mu) and the odd pair
    (x + mu, x + 3mu)."""
    S, idx = _lattice(L, d, S, idx)
    seen = set()
    quads = []
    for x in S:
        if parity(x) != 1:
            continue
        for mu in range(d):

            def sh(z, k):
                z = list(z)
                z[mu] = (z[mu] + k) % L
                return tuple(z)

            q = (idx[x], idx[sh(x, 2)], idx[sh(x, 1)], idx[sh(x, 3)])
            key = frozenset(q)
            if key in seen or len(key) < 4:
                continue
            seen.add(key)
            quads.append(q)
    return quads


def oriented_links(M, S, idx, L, d):
    """{(u even, v odd): s_l} over the nearest-neighbour links, s_l = sign G_uv of the free propagator G = M^-1, so that
    Kt_l = s_l K_l has <Kt_l>_0 > 0."""
    G1 = np.linalg.inv(M)
    ls = {}
    for x in S:
        if parity(x) != 1:
            continue
        for mu in range(d):
            for sg in (1, -1):
                y = list(x)
                y[mu] = (y[mu] + sg) % L
                link = (idx[x], idx[tuple(y)])
                ls[link] = float(np.sign(G1[link]))
    return ls


# ---- field algebra -----------------------------------------------------------------------------------------
class FieldAlgebra:
    """Grassmann polynomials of the four-flavour fields on ``nsites`` sites (flavour-major generator index)."""

    def __init__(self, nsites):
        self.n = int(nsites)
        self.ngen = NF * self.n

    def fidx(self, site, a):
        return a * self.n + site

    def chi(self, site, a):
        return Grass.gen(self.fidx(site, a))

    def K(self, u, v):
        """K_uv = sum_a chi^a(u) chi^a(v) (antisymmetric under u <-> v)."""
        r = Grass()
        for a in range(NF):
            r = r + self.chi(u, a) * self.chi(v, a)
        return r

    def eps_term(self, site):
        """chi^1 chi^2 chi^3 chi^4 at a site."""
        r = Grass.one()
        for a in range(NF):
            r = r * self.chi(site, a)
        return r

    def Phi(self, x, xp, a, b):
        return self.chi(x, a) * self.chi(xp, b) + self.chi(x, b) * self.chi(xp, a)

    def Lam(self, x, xp, a, b):
        return self.chi(x, a) * self.chi(xp, b) - self.chi(x, b) * self.chi(xp, a)

    def sym2_term(self, q, explicit=False):
        """T_Q = sum_ab Phibar^{ab}(y, y') Phi^{ab}(x, x') on q = (x, x', y, y') (Fierz form unless ``explicit``)."""
        x, xp, y, yp = q
        if explicit:
            r = Grass()
            for a in range(NF):
                for b in range(NF):
                    r = r + self.Phi(y, yp, a, b) * self.Phi(x, xp, a, b)
            return r
        return (self.K(x, yp) * self.K(xp, y) - self.K(x, y) * self.K(xp, yp)) * 2.0

    def lam2_term(self, q, explicit=False):
        """T_6 = sum_ab Lambdabar^{ab}(y, y') Lambda^{ab}(x, x') on the quad."""
        x, xp, y, yp = q
        if explicit:
            r = Grass()
            for a in range(NF):
                for b in range(NF):
                    r = r + self.Lam(y, yp, a, b) * self.Lam(x, xp, a, b)
            return r
        return (self.K(x, yp) * self.K(xp, y) + self.K(x, y) * self.K(xp, yp)) * (-2.0)

    def kinetic(self, M):
        """S_0 = 1/2 sum chi^a(x) M_xy chi^a(y) as a Grass element."""
        S0 = Grass()
        for i in range(self.n):
            for j in range(self.n):
                if abs(M[i, j]) > 0:
                    for a in range(NF):
                        S0 = S0 + self.chi(i, a) * self.chi(j, a) * (0.5 * M[i, j])
        return S0

    def map_su4(self, U, parities):
        """Substitution matrix of chi_e -> U chi_e (4), chi_o -> U* chi_o (4bar); parities[site] = +-1."""
        V = np.zeros((self.ngen, self.ngen), complex)
        for s in range(self.n):
            Us = U if parities[s] == 1 else np.conj(U)
            for a in range(NF):
                for b in range(NF):
                    V[self.fidx(s, a), self.fidx(s, b)] = Us[a, b]
        return V

    def map_u1eps(self, alpha, parities):
        """Substitution matrix of chi(x) -> exp(i alpha eps(x)) chi(x)."""
        V = np.zeros((self.ngen, self.ngen), complex)
        for s in range(self.n):
            for a in range(NF):
                V[self.fidx(s, a), self.fidx(s, a)] = np.exp(1j * alpha * parities[s])
        return V


def random_su4(rng):
    """A Haar-random SU(4) matrix (QR of a complex Gaussian, phases fixed, det normalised)."""
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.diag(R) / np.abs(np.diag(R)))
    Q = Q / np.linalg.det(Q) ** 0.25
    assert np.allclose(np.linalg.det(Q), 1) and np.allclose(Q @ Q.conj().T, np.eye(4))
    return Q


# ---- Wick's theorem, factorised over flavours ----------------------------------------------------------
class WickCache:
    """<monomial>_0 for flavour-major monomials: the propagator 1_4 (x) M^-1 is flavour-block-diagonal, so the
    Pfaffian factorises over flavours; each flavour block Pfaffian is cached by its site subset."""

    def __init__(self, M):
        self.n = M.shape[0]
        self.Ginv = np.linalg.inv(M)
        self.cache = {}

    def block(self, mask):
        v = self.cache.get(mask)
        if v is None:
            idx = [i for i in range(self.n) if mask >> i & 1]
            v = pfaffian(self.Ginv[np.ix_(idx, idx)]).real if len(idx) % 2 == 0 else 0.0
            self.cache[mask] = v
        return v

    def mono(self, mask):
        full = (1 << self.n) - 1
        r = 1.0
        for a in range(NF):
            r *= self.block((mask >> (a * self.n)) & full)
            if r == 0.0:
                return 0.0
        return r

    def expect(self, poly):
        return sum(c * self.mono(m) for m, c in poly.t.items())


# ---- 10-plet Yukawa (Hubbard-Stratonovich) operator ------------------------------------------------------
def sym2_yukawa_matrix(M, quads, phi):
    """Antisymmetric 4N x 4N D with 1/2 chi^T D chi = S_0 - sum_Q sum_ab [conj(phi_Q^ab) Phi^ab(x, x') +
    phi_Q^ab Phibar^ab(y, y')]; phi[Q] is a complex symmetric 4 x 4 matrix (the 10 of SU(4))."""
    N = M.shape[0]
    fa = FieldAlgebra(N)
    D = np.kron(np.eye(NF), M).astype(complex)
    for q, ph in zip(quads, phi, strict=True):
        x, xp, y, yp = q
        ph = np.asarray(ph)
        assert np.allclose(ph, ph.T)
        for a in range(NF):
            for b in range(NF):
                D[fa.fidx(x, a), fa.fidx(xp, b)] += -2.0 * np.conj(ph[a, b])
                D[fa.fidx(xp, b), fa.fidx(x, a)] += +2.0 * np.conj(ph[a, b])
                D[fa.fidx(y, a), fa.fidx(yp, b)] += -2.0 * ph[a, b]
                D[fa.fidx(yp, b), fa.fidx(y, a)] += +2.0 * ph[a, b]
    assert np.allclose(D, -D.T)
    return D


# ---- SU(4) representation theory (numerical) --------------------------------------------------------------
def su4_generators():
    """15 traceless hermitian 4 x 4 generators (off-diagonal real/imaginary pairs, then the diagonal ones)."""
    gens = []
    for i in range(4):
        for j in range(i + 1, 4):
            E = np.zeros((4, 4), complex)
            E[i, j] = 1
            gens.append(E + E.T)
            gens.append(-1j * E + 1j * E.T)
    for k in range(1, 4):
        D = np.zeros((4, 4), complex)
        D[:k, :k] = np.eye(k)
        D[k, k] = -k
        gens.append(D)
    return gens


def rep_on_tensor(gens, reps):
    """Generators on a tensor product of fundamentals ("fund") and antifundamentals ("anti")."""
    out = []
    for g in gens:
        mats = [g if r == "fund" else -g.T for r in reps]
        tot = np.zeros((4 ** len(reps),) * 2, complex)
        for k, m in enumerate(mats):
            f = [np.eye(4)] * len(reps)
            f[k] = m
            t = f[0]
            for ff in f[1:]:
                t = np.kron(t, ff)
            tot += t
        out.append(tot)
    return out


def tensor_gens(gA, gB):
    """Generators of a tensor product of two representations given by their generator lists."""
    return [np.kron(a, np.eye(b.shape[0])) + np.kron(np.eye(a.shape[0]), b) for a, b in zip(gA, gB, strict=True)]


def invariant_dim(gen_mats, P=None):
    """Dimension of the joint null space of a set of generators (optionally restricted to the columns of P)."""
    mats = [P.conj().T @ g @ P for g in gen_mats] if P is not None else gen_mats
    A = np.vstack(mats)
    s = np.linalg.svd(A, compute_uv=False)
    n = A.shape[1]
    return int(np.sum(s < 1e-9)) + (n - len(s) if len(s) < n else 0)


def sym_antisym_projectors():
    """Orthonormal bases (16 x 10, 16 x 6) of Sym^2 and Lambda^2 inside 4 (x) 4."""
    P = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            P[i * 4 + j, j * 4 + i] = 1
    w, v = np.linalg.eigh((P + P.T) / 2)
    return v[:, w > 0.5], v[:, w < -0.5]


def levi_civita4():
    eps = np.zeros((4, 4, 4, 4))
    for p in itertools.permutations(range(4)):
        eps[p] = np.linalg.det(np.eye(4)[list(p)])
    return eps
