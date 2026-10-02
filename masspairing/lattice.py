"""Hypercubic lattice, staggered phases, fermionic shifts and the reduced staggered kinetic matrix.

Conventions (used throughout the package):

* A lattice of shape (L_1, ..., L_d); the last axis is Euclidean time. Fields carry the lattice as their
  last d axes; fermion vectors are site-major with the site index the C-order flattening of the shape.
* ``bc[mu] = +1`` (periodic) or ``-1`` (antiperiodic) is the fermion boundary sign; bosons are periodic.
  The default is periodic space, antiperiodic time.
* Staggered phase eta_mu(x) = (-1)^(x_1 + ... + x_{mu-1}); site parity eps(x) = (-1)^(x_1 + ... + x_d).
* The kinetic matrix K has (K psi)(x) = sum_mu eta_mu(x) [psi(x + mu) - psi(x - mu)] with the boundary
  signs; it is real antisymmetric and connects opposite parities only.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sps


def get_xp(device: str = "cpu"):
    """Array module for a device name: numpy for "cpu", cupy for "cuda"."""
    if device in ("cpu", None):
        return np
    import cupy

    return cupy


def xp_of(a):
    """Array module of an array (numpy or cupy)."""
    if isinstance(a, np.ndarray):
        return np
    try:
        import cupy
    except ImportError:
        return np
    return cupy.get_array_module(a)


def to_numpy(a):
    """A numpy copy/view of a numpy or cupy array."""
    if isinstance(a, np.ndarray):
        return a
    return a.get() if hasattr(a, "get") else np.asarray(a)


class Lattice:
    """Hypercubic lattice of any dimension d with fermion boundary signs ``bc``."""

    def __init__(self, shape, bc=None, xp=np):
        self.shape = tuple(int(s) for s in shape)
        self.d = len(self.shape)
        self.V = math.prod(self.shape)
        default = (1,) * (self.d - 1) + (-1,)
        self.bc = tuple(int(b) for b in (bc if bc is not None else default))
        assert len(self.bc) == self.d
        self.xp = xp
        coords = np.indices(self.shape)
        cum = np.cumsum(coords, axis=0)
        eta = np.ones((self.d,) + self.shape)
        for mu in range(1, self.d):
            eta[mu] = (-1.0) ** cum[mu - 1]
        self.coords = coords
        self.eps_np = (-1.0) ** cum[self.d - 1]
        self.eta = xp.asarray(eta)
        self.eps = xp.asarray(self.eps_np)
        self.eps_spatial = xp.asarray((-1.0) ** cum[self.d - 2])

    # ---- shifts -------------------------------------------------------------------------------
    def fwd(self, f, mu, dist=1, bc=True):
        """g(x) = f(x + dist mu), with the fermionic boundary sign when ``bc``."""
        ax = f.ndim - self.d + mu
        g = self.xp.roll(f, -dist, axis=ax)
        if bc and self.bc[mu] != 1:
            L = self.shape[mu]
            idx = [slice(None)] * f.ndim
            idx[ax] = slice(L - dist, L)
            g[tuple(idx)] = g[tuple(idx)] * self.bc[mu]
        return g

    def bwd(self, f, mu, dist=1, bc=True):
        """g(x) = f(x - dist mu), with the fermionic boundary sign when ``bc``."""
        ax = f.ndim - self.d + mu
        g = self.xp.roll(f, dist, axis=ax)
        if bc and self.bc[mu] != 1:
            idx = [slice(None)] * f.ndim
            idx[ax] = slice(0, dist)
            g[tuple(idx)] = g[tuple(idx)] * self.bc[mu]
        return g

    def box(self, s):
        """Block-lattice Laplacian sum_mu [s(x + 2mu) + s(x - 2mu) - 2 s(x)] (periodic, bosonic)."""
        out = -2.0 * self.d * s
        for mu in range(self.d):
            out = out + self.fwd(s, mu, 2, bc=False) + self.bwd(s, mu, 2, bc=False)
        return out

    # ---- helpers ------------------------------------------------------------------------------
    def momenta(self, twisted=True):
        """Fermionic momenta p_mu = 2 pi (n_mu + theta_mu) / L_mu, theta = 1/2 on antiperiodic axes."""
        ps = []
        for mu, L in enumerate(self.shape):
            th = 0.5 if (twisted and self.bc[mu] == -1) else 0.0
            ps.append(2 * np.pi * (np.arange(L) + th) / L)
        return np.meshgrid(*ps, indexing="ij")

    def random_sigma(self, rng, scale=1.0):
        """Gaussian sigma triplet of shape (3,) + shape."""
        return self.xp.asarray(rng.normal(size=(3,) + self.shape) * scale)

    def random_psi(self, rng, n=None):
        """Complex Gaussian doublet(s) (V, 2) [or (n, V, 2)] with <psi psi^dagger> = 1."""
        shape = (self.V, 2) if n is None else (n, self.V, 2)
        z = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        return self.xp.asarray(z / math.sqrt(2.0))


def kinetic_matrix(lat: Lattice) -> sps.csr_matrix:
    """Real antisymmetric CSR matrix K of the reduced staggered hopping term (boundary signs included)."""
    V = lat.V
    idx = np.arange(V).reshape(lat.shape)
    eta = to_numpy(lat.eta)
    rows, cols, vals = [], [], []
    for mu in range(lat.d):
        L = lat.shape[mu]
        for sgn, dist in ((+1.0, 1), (-1.0, -1)):
            j = np.roll(idx, -dist, axis=mu)
            s = np.ones(lat.shape)
            if lat.bc[mu] != 1:
                sl = [slice(None)] * lat.d
                sl[mu] = slice(L - 1, L) if dist == 1 else slice(0, 1)
                s[tuple(sl)] = lat.bc[mu]
            rows.append(idx.reshape(-1))
            cols.append(j.reshape(-1))
            vals.append((sgn * eta[mu] * s).reshape(-1))
    K = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(V, V))
    K.sum_duplicates()
    K.sort_indices()
    return K


def shift_table(lat: Lattice, delta):
    """(index of x + delta, fermionic boundary sign of the displacement) as two (V,) arrays.

    The sign is the product of bc[mu] over the boundaries crossed, so that w(x, delta) chi(x) chi(x + delta)
    is the translation-covariant bilinear.
    """
    V = lat.V
    j = np.arange(V).reshape(lat.shape)
    w = np.ones(lat.shape)
    for mu, dm in enumerate(delta):
        if not dm:
            continue
        j = np.roll(j, -dm, axis=mu)
        if lat.bc[mu] != 1:
            c = lat.coords[mu] + dm
            w = w * np.where((c < 0) | (c >= lat.shape[mu]), float(lat.bc[mu]), 1.0)
    return j.reshape(V), w.reshape(V)


def parse_bc(spec: str) -> tuple:
    """Boundary conditions from a string of 'p' (periodic) and 'a' (antiperiodic), e.g. 'pppa'."""
    assert set(spec) <= {"p", "a"}, spec
    return tuple(1 if c == "p" else -1 for c in spec)
