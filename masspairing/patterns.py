"""Same-parity pair patterns of the four-flavour reduced staggered model.

A pattern is a real antisymmetric V x V matrix C; flavour-blind it enters the doublet operator as
K -> K - h C (weight e^{+h O}, O = 1/2 sum_{xz} C[x, z] sum_a chi^a(x) chi^a(z)).

* ``source_pattern`` -- the nodal same-parity pairing field C0: the pair operators chi(x) chi(x + delta) on the
  diagonal displacements delta = mu +- nu (even sites) and -mu +- nu (odd sites), boundary signs included.
  It carries no staggered phase and annihilates the 16 corner zero modes: it never opens a gap.
* ``plane_op`` / ``chiral_mass`` -- the phased two-link plane operators and the taste-chiral Majorana mass
  C_chi = [C_plane(mu nu) + dual_sign eps_{mu nu rho sigma} C_plane(rho sigma)] / 8: a mass h for one taste
  doublet of every flavour, the other doublet exactly massless at the corners.
* ``TasteProjector`` -- the light/heavy taste projectors P_L, P_R of C_chi, exact at every momentum where a real
  split exists.

Corner space: near the 16 Brillouin-zone corners p in {0, pi}^4 the free operator is K ~ i sum_mu k_mu Gamma_mu
with Gamma_mu = X_{S_mu} Z_{mu} (S_mu = {0, ..., mu - 1}), real "Pauli strings" on the four corner bits:
X_S flips the bits in S (a momentum shift by pi, i.e. the phase (-1)^{sum_{rho in S} x_rho}), Z_D multiplies by
(-1)^{|A cap D|}. A translation-covariant bilinear with phase S and displacement support D has corner block
proportional to X_S Z_D; it is a scalar mass iff it anticommutes with every Gamma_mu.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from .lattice import Lattice, shift_table, to_numpy

# the unique phase set S (0-based axes) making the two-link pattern on plane (mu, nu) a scalar mass
PH = {(0, 1): (0, 2, 3), (0, 2): (0, 3), (0, 3): (0,), (1, 2): (0, 1, 3), (1, 3): (0, 1), (2, 3): (0, 1, 2)}
DUAL = {(0, 1): (2, 3), (2, 3): (0, 1), (0, 2): (1, 3), (1, 3): (0, 2), (0, 3): (1, 2), (1, 2): (0, 3)}
PATTERNS = ("c0", "chiral", "full")


# ---- nodal same-parity pattern --------------------------------------------------------------------
def pair_planes(d):
    """(mu, nu, sg, delta_even, delta_odd) for mu < nu and sg = +-1.

    The even pair is (x, x + delta), delta = mu + sg nu; the odd pair is (y, y + delta'), delta' = -mu + sg nu,
    i.e. the two pairs of the plaquette quad (x, x' = x + delta; y = x + mu, y' = y + delta' = x + sg nu).
    """
    out = []
    for mu in range(d):
        for nu in range(mu + 1, d):
            for sg in (+1, -1):
                de = np.zeros(d, int)
                de[mu] = 1
                de[nu] = sg
                do = np.zeros(d, int)
                do[mu] = -1
                do[nu] = sg
                out.append((mu, nu, sg, tuple(de), tuple(do)))
    return out


def source_pattern(lat: Lattice) -> sps.csr_matrix:
    """The flavour-blind nodal same-parity pattern C0 (real antisymmetric CSR, entries +-1 on L >= 3)."""
    V = lat.V
    eps = lat.eps_np.reshape(V)
    rows, cols, vals = [], [], []
    x = np.arange(V)
    for _mu, _nu, _sg, de, do in pair_planes(lat.d):
        for par, delta in ((+1, de), (-1, do)):
            sel = eps == par
            j, w = shift_table(lat, delta)
            rows += [x[sel], j[sel]]
            cols += [j[sel], x[sel]]
            vals += [w[sel], -w[sel]]
    C = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, V))
    C.sum_duplicates()
    C.eliminate_zeros()
    C.sort_indices()
    return C


# ---- corner-space strings ------------------------------------------------------------------------
def pauli_string(S: int, D: int, n: int = 4) -> np.ndarray:
    """Real 2^n x 2^n matrix of X_S Z_D on corner labels A (bit vectors): (X_S Z_D)_{A^S, A} = (-1)^{|A & D|}."""
    N = 1 << n
    M = np.zeros((N, N))
    for A in range(N):
        M[A ^ S, A] = (-1.0) ** bin(A & D).count("1")
    return M


def kinetic_gammas(n: int = 4):
    """Corner-space Gamma_mu = X_{S_mu} Z_{mu}, S_mu = {0, ..., mu - 1}."""
    return [pauli_string(sum(1 << r for r in range(mu)), 1 << mu, n) for mu in range(n)]


def scalar_mass_strings(n: int = 4):
    """All (S, D, antisymmetric) with X_S Z_D anticommuting with every Gamma_mu (the corner-space scalar masses)."""
    G = kinetic_gammas(n)
    out = []
    for S in range(1 << n):
        for D in range(1 << n):
            P = pauli_string(S, D, n)
            if all(np.allclose(P @ g + g @ P, 0.0) for g in G):
                out.append(
                    (
                        tuple(r for r in range(n) if S >> r & 1),
                        tuple(r for r in range(n) if D >> r & 1),
                        bool(np.allclose(P.T, -P)),
                    )
                )
    return out


# ---- plane operators and the taste-chiral mass -----------------------------------------------------
def plane_op(lat: Lattice, mu: int, nu: int, parity=None, phase=None) -> sps.csr_matrix:
    """Real antisymmetric C with 1/2 chi^T C chi = sum_x zeta(x) chi(x) chi(x + mu + nu) + (same for mu - nu).

    zeta = (-1)^{sum_{rho in S} x_rho} with S = PH[(mu, nu)] (or ``phase``, a tuple of axes; () = no phase);
    boundary signs included; both orientations. ``parity`` = +1/-1 keeps the terms starting on even/odd x only.
    """
    V, d = lat.V, lat.d
    co = lat.coords.reshape(d, V)
    eps = lat.eps_np.reshape(V)
    S = PH[(mu, nu)] if phase is None else tuple(phase)
    zeta = (-1.0) ** (sum(co[r] for r in S) % 2) if S else np.ones(V)
    rows, cols, vals = [], [], []
    x = np.arange(V)
    sel = np.ones(V, bool) if parity is None else (eps == parity)
    for sg in (+1, -1):
        de = np.zeros(d, int)
        de[mu] = 1
        de[nu] = sg
        j, w = shift_table(lat, de)
        v = (w * zeta)[sel]
        rows += [x[sel], j[sel]]
        cols += [j[sel], x[sel]]
        vals += [v, -v]
    C = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, V))
    C.sum_duplicates()
    C.eliminate_zeros()
    C.sort_indices()
    return C


def eps_sign(comp) -> int:
    """epsilon_{mu nu rho sigma} for (rho, sigma) = DUAL[(mu, nu)]."""
    perm = tuple(comp) + tuple(DUAL[tuple(comp)])
    inv = sum(1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j])
    return -1 if inv % 2 else +1


def chiral_mass(lat: Lattice, comp=(0, 3), dual_sign=+1, parity=None) -> sps.csr_matrix:
    """The taste-chiral Majorana mass C_chi = [C_plane(comp) + dual_sign eps C_plane(dual(comp))] / 8.

    Normalised so that K - h C_chi has corner gap exactly h on one taste doublet; the other doublet keeps 8 exact
    corner zero modes. ``dual_sign = +1`` (self-dual) gaps the same doublet for comp = (0,3), (0,1), (0,2);
    ``dual_sign = -1`` (anti-self-dual) gaps the other one.
    """
    c1 = plane_op(lat, *comp, parity=parity)
    c2 = plane_op(lat, *DUAL[tuple(comp)], parity=parity)
    C = (c1 + dual_sign * eps_sign(comp) * c2) / 8.0
    C.sum_duplicates()
    C.eliminate_zeros()
    C.sort_indices()
    return C


def all_planes_mass(lat: Lattice, phase="scalar") -> sps.csr_matrix:
    """Sum over the six planes of ``plane_op`` / 8 (a non-chiral mass for all four Weyl fermions)."""
    V = lat.V
    C = sps.csr_matrix((V, V))
    for mu, nu in PH:
        if phase == "scalar":
            C = C + plane_op(lat, mu, nu)
        else:
            C = C + plane_op(lat, mu, nu, phase=tuple(range(nu)))
    return C / 8.0


def source_matrix(lat: Lattice, pattern="c0", comp=(0, 3), dual_sign=+1) -> sps.csr_matrix:
    """The flavour-blind pattern by name: "c0" (nodal), "chiral" (C_chi), "full" (all six planes).

    A scipy sparse matrix passed as ``pattern`` is returned as CSR unchanged (custom patterns, e.g. controls).
    """
    if sps.issparse(pattern):
        return sps.csr_matrix(pattern)
    if pattern == "c0":
        return source_pattern(lat)
    if pattern == "chiral":
        return chiral_mass(lat, comp, dual_sign).tocsr()
    if pattern == "full":
        return all_planes_mass(lat).tocsr()
    raise ValueError(f"unknown source pattern {pattern!r} (choose from {PATTERNS})")


def singular_values(M, k=None):
    """Sorted singular values of a dense real matrix (all, or the lowest k)."""
    M = np.asarray(M, float)
    s = np.sqrt(np.maximum(np.linalg.eigvalsh(M.T @ M), 0.0))
    return s if k is None else s[:k]


# ---- taste projectors -------------------------------------------------------------------------------
class TasteProjector:
    """Taste chirality Q and the projectors P_R (heavy doublet of C_chi) and P_L (light doublet).

    In the 16-block of a reduced momentum p the string Q = eps J_D J_dual = X_{13} Z_{0123} commutes with K(p)
    and every plane operator; in position space Q = zeta_13(x) F^-1 [prod_mu sign cos p_mu] F with twisted
    momenta. At momenta with cos p_mu = 0 a real field has no light/heavy split; there sign(0) := 0, Q^2 = P_nb
    projects onto the momenta where the split exists, P_R = (Q^2 + Q)/2, P_L = (Q^2 - Q)/2. On all-antiperiodic
    boxes with every L = 0 mod 4 there are no such modes and P_L + P_R = 1.

    Applied by FFT to arrays whose first axis is the site index (V, ...); ``dense_*`` give V x V matrices.
    """

    def __init__(self, lat: Lattice, comp=(0, 3), dual_sign=+1, tol=1e-12):
        self.lat = lat
        self.comp, self.dual_sign = tuple(comp), int(dual_sign)
        d, V = lat.d, lat.V
        assert d == 4, "the taste projector is a 4d corner-space object"
        co = lat.coords.reshape(d, V)
        S = (1, 3)
        self.zeta = ((-1.0) ** (sum(co[r] for r in S) % 2)).reshape(lat.shape)
        self.twist = np.ones(lat.shape, complex)
        zs = np.ones(lat.shape)
        for mu, L in enumerate(lat.shape):
            th = 0.5 if lat.bc[mu] == -1 else 0.0
            n = np.arange(L)
            c = np.cos(2.0 * np.pi * (n + th) / L)
            sgn = np.where(np.abs(c) < tol, 0.0, np.sign(c))
            shp = [1] * d
            shp[mu] = L
            zs = zs * sgn.reshape(shp)
            self.twist = self.twist * np.exp(-1j * 2.0 * np.pi * th * n / L).reshape(shp)
        self.zsign = zs
        self.n_boundary = int((zs == 0.0).sum())
        self.sign = -int(dual_sign)
        self._dense = None

    def _Zhat(self, v, power=1):
        lat = self.lat
        shp = v.shape
        f = v.reshape(lat.shape + shp[1:])
        ax = tuple(range(lat.d))
        ex = (Ellipsis,) + (None,) * (v.ndim - 1)
        g = np.fft.fftn(f * self.twist[ex], axes=ax)
        g = np.fft.ifftn(g * (self.zsign**power)[ex], axes=ax) * np.conj(self.twist)[ex]
        out = g.reshape(shp)
        return out.real if np.isrealobj(v) else out

    def Q(self, v):
        """+1 on the doublet gapped by chiral_mass(lat, comp, dual_sign), -1 on the other, 0 on boundary modes."""
        ex = (Ellipsis,) + (None,) * (v.ndim - 1)
        return self.sign * self.zeta.reshape(-1)[ex] * self._Zhat(v)

    def Pnb(self, v):
        """Projector onto the momenta where the real light/heavy split exists."""
        return self._Zhat(v, power=2)

    def PR(self, v):
        return 0.5 * (self.Pnb(v) + self.Q(v))

    def PL(self, v):
        return 0.5 * (self.Pnb(v) - self.Q(v))

    def dense(self):
        if self._dense is None:
            self._dense = self.Q(np.eye(self.lat.V))
        return self._dense

    def dense_Pnb(self):
        return self.Pnb(np.eye(self.lat.V))

    def dense_PL(self):
        return 0.5 * (self.dense_Pnb() - self.dense())

    def dense_PR(self):
        return 0.5 * (self.dense_Pnb() + self.dense())

    def project(self, G, which="L"):
        """P G P for a dense propagator array with the site indices on axes 0 and -2."""
        P = self.dense_PL() if which == "L" else self.dense_PR()
        return np.einsum("xy,y...zb->x...zb", P, np.einsum("y...zb,zw->y...wb", G, P))


def light_pattern(lat: Lattice, comp=(0, 3), dual_sign=+1, proj=None) -> np.ndarray:
    """C_L = P_L C_asd P_L: the Majorana-mass direction of the light doublet (dense V x V)."""
    tp = proj if proj is not None else TasteProjector(lat, comp, dual_sign)
    Casd = chiral_mass(lat, comp, -dual_sign).toarray()
    PL = tp.dense_PL()
    return PL @ Casd @ PL


__all__ = [
    "PH",
    "DUAL",
    "PATTERNS",
    "pair_planes",
    "source_pattern",
    "pauli_string",
    "kinetic_gammas",
    "scalar_mass_strings",
    "plane_op",
    "eps_sign",
    "chiral_mass",
    "all_planes_mass",
    "source_matrix",
    "singular_values",
    "TasteProjector",
    "light_pattern",
    "to_numpy",
]
