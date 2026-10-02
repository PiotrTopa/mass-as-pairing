"""The doublet fermion operator of the model.

    D_c(sigma, s) = (K - A(a) - h C) (x) 1_2 + i y sigma.tau  [+ m]

on the complex doublet psi (V, 2): K the reduced staggered kinetic matrix, A(a) the link term of the
Hubbard-Stratonovich fields s (a = I^T s, ``wedge.WedgeGeometry``), C a flavour-blind same-parity pattern
(``patterns.source_matrix``) with strength h, y the Yukawa coupling of the sigma triplet, m an optional Dirac mass.
K - A - h C is real antisymmetric, so for m = 0 the operator is anti-hermitian and commutes with the antiunitary
tau_2 K: det D_c = det(D_c^dagger D_c)^{1/2} > 0 on every configuration.

Flavour-selective source (``h_sel``, ``mask``): in the real 4V basis M(h) = Re[W D_c W^dagger] - h_sel C (x) diag(mask).
It is real-linear but not complex-linear on the packed doublet; the operator then applies
M psi = D_c psi - h_sel C (P_re Re psi + i P_im Im psi), P_re = (m_0, m_2), P_im = (m_1, m_3), and the weight of the
RHMC is |Pf M| = det(M^T M)^{1/4} (the sign is not in the weight).

Configurations are flat arrays ``fields = [sigma (3V), s (n_fields)]``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from .flavour import realify
from .lattice import Lattice, kinetic_matrix, to_numpy, xp_of
from .patterns import source_matrix, source_pattern


class Yukawa:
    """On-site term i y sigma.tau as three (V,) arrays: block [[a, b], [c, -a]], plus the kinetic CSR ``K``."""

    __slots__ = ("a", "b", "c", "K")

    def __init__(self, sigma, y):
        s = sigma.reshape(3, -1)
        self.a = 1j * y * s[2]
        self.b = 1j * y * (s[0] - 1j * s[1])
        self.c = 1j * y * (s[0] + 1j * s[1])
        self.K = None

    def blocks(self):
        """(V, 2, 2) complex blocks (numpy)."""
        a, b, c = (to_numpy(v) for v in (self.a, self.b, self.c))
        Y = np.empty((a.size, 2, 2), complex)
        Y[:, 0, 0], Y[:, 0, 1], Y[:, 1, 0], Y[:, 1, 1] = a, b, c, -a
        return Y


def _csr(xp, data, indices, indptr, V):
    if xp is np:
        return sps.csr_matrix((data, indices, indptr), shape=(V, V))
    import cupyx.scipy.sparse as csp

    return csp.csr_matrix((data, xp.asarray(indices), xp.asarray(indptr)), shape=(V, V))


class DoubletOperator:
    """D_c(sigma, s) with optional link fields, flavour-blind pattern, Dirac mass or flavour-selective source.

    Parameters
    ----------
    lat : Lattice
    y : Yukawa coupling
    geom : WedgeGeometry or None (no link fields)
    h, pattern : flavour-blind source strength and pattern ("c0", "chiral", "full" or a sparse matrix)
    mass : Dirac mass m (adds m psi; only without link fields and sources)
    h_sel, mask : flavour-selective source strength and (4,) flavour mask (uses ``pattern`` as C)
    """

    def __init__(
        self,
        lat: Lattice,
        y: float,
        geom=None,
        h: float = 0.0,
        pattern="c0",
        mass: float = 0.0,
        h_sel: float = 0.0,
        mask=(0.0, 0.0, 0.0, 1.0),
    ):
        self.lat, self.y, self.geom = lat, float(y), geom
        self.xp = lat.xp
        self.h, self.pattern, self.mass = float(h), pattern, float(mass)
        self.h_sel = float(h_sel)
        self.mask = np.asarray(mask, float).reshape(4)
        self.selective = bool(self.h_sel) and bool(np.any(self.mask != 0.0))
        assert not (self.h and self.selective), "flavour-blind and flavour-selective sources are exclusive"
        assert not (self.mass and (self.h or self.selective)), "a Dirac mass is not combined with sources"
        self.K0 = geom.K0 if geom is not None else kinetic_matrix(lat)
        self.n_fields = geom.n_fields if geom is not None else 0
        self._tables = None
        self._merged = None
        self._csel = None
        if self.selective:
            self.C = (
                source_pattern(lat) if isinstance(pattern, str) and pattern == "c0" else source_matrix(lat, pattern)
            )
            self.p_re = self.mask[0::2].copy()
            self.p_im = self.mask[1::2].copy()
        elif self.h:
            self.C = source_matrix(lat, pattern)
        else:
            self.C = None

    # ---- tables -------------------------------------------------------------------------------
    def merged_tables(self):
        """Merged CSR structure of K0 and the flavour-blind pattern (used when h != 0)."""
        if self._merged is None:
            V = self.lat.V
            K0, C = self.K0, self.C
            Pm = (abs(K0) + abs(C)).tocsr()
            Pm.sum_duplicates()
            Pm.sort_indices()
            rows = np.repeat(np.arange(V), np.diff(Pm.indptr))
            cols = Pm.indices
            k0 = np.asarray(K0[rows, cols]).reshape(-1)
            c0 = np.asarray(C[rows, cols]).reshape(-1)
            isK = k0 != 0
            link = np.zeros(len(rows), np.int64)
            if self.geom is not None:
                pid = self.geom.pair_id
                link[isK] = [pid[(min(a, b), max(a, b))] for a, b in zip(rows[isK], cols[isK], strict=False)]
            self._merged = dict(
                indices=Pm.indices.astype(np.int32),
                indptr=Pm.indptr.astype(np.int32),
                data0=k0,
                ent_link=link,
                ent_eps=self.lat.eps_np.reshape(V)[rows] * isK,
                ent_c=c0,
                _dev={},
            )
        return self._merged

    def _merged_dev(self, xp):
        t = self.merged_tables()
        d = t["_dev"].get(xp.__name__)
        if d is None:
            d = t["_dev"][xp.__name__] = {k: xp.asarray(t[k]) for k in ("data0", "ent_link", "ent_eps", "ent_c")}
        return d

    # ---- fields -> operator -------------------------------------------------------------------
    def split(self, fields):
        """(sigma (3,) + shape, s (n_fields,)) views of a flat configuration."""
        V = self.lat.V
        return fields[: 3 * V].reshape((3,) + self.lat.shape), fields[3 * V :]

    def link_coefficients(self, s):
        return self.geom.dev(xp_of(s))["IT"] @ s

    def prepare(self, fields) -> Yukawa:
        """Yukawa blocks of sigma and the field-dependent real kinetic part (complex CSR in ``Y.K``)."""
        xp = xp_of(fields)
        V = self.lat.V
        sigma, s = self.split(fields)
        Y = Yukawa(sigma, self.y)
        has_links = self.geom is not None
        a = self.geom.dev(xp)["IT"] @ s if has_links else None
        if self.h:
            m = self._merged_dev(xp)
            t = self.merged_tables()
            if has_links:
                data = m["data0"] - m["ent_eps"] * a[m["ent_link"]] - self.h * m["ent_c"]
            else:
                data = m["data0"] - self.h * m["ent_c"]
            Y.K = _csr(xp, data.astype(complex), t["indices"], t["indptr"], V)
            return Y
        if has_links:
            t = self.geom.dev(xp)
            data = t["data0"] - t["ent_eps"] * a[t["ent_link"]]
        else:
            data = xp.asarray(self.K0.data)
        Y.K = _csr(xp, data.astype(complex), self.K0.indices, self.K0.indptr, V)
        return Y

    # ---- application ---------------------------------------------------------------------------
    def _apply_csr(self, psi, Y):
        V = self.lat.V
        shp = psi.shape
        Kp = (Y.K @ psi.reshape(V, -1)).reshape(shp)
        if psi.ndim == 3:
            Kp[..., 0] += Y.a[:, None] * psi[..., 0] + Y.b[:, None] * psi[..., 1]
            Kp[..., 1] += Y.c[:, None] * psi[..., 0] - Y.a[:, None] * psi[..., 1]
        else:
            Kp[:, 0] += Y.a * psi[:, 0] + Y.b * psi[:, 1]
            Kp[:, 1] += Y.c * psi[:, 0] - Y.a * psi[:, 1]
        return Kp

    def _C_sel(self, xp):
        if self._csel is None or self._csel[0] is not xp:
            if xp is np:
                c = self.C.astype(complex)
            else:
                import cupyx.scipy.sparse as csp

                c = csp.csr_matrix(self.C.astype(complex))
            self._csel = (xp, c)
        return self._csel[1]

    def source_apply(self, psi):
        """-h_sel C (P_re Re psi + i P_im Im psi) on packed vectors (V, 2) or batches (V, n, 2)."""
        xp = xp_of(psi)
        pr, pi = xp.asarray(self.p_re), xp.asarray(self.p_im)
        proj = psi.real * pr + 1j * (psi.imag * pi)
        shp = psi.shape
        return (-self.h_sel) * (self._C_sel(xp) @ proj.reshape(self.lat.V, -1)).reshape(shp)

    def apply(self, psi, fields=None, Y=None):
        """D psi for psi of shape (V, 2) or a batch (V, n, 2)."""
        if Y is None:
            Y = self.prepare(fields)
        out = self._apply_csr(psi, Y)
        if self.selective:
            return out + self.source_apply(psi)
        if self.mass:
            out = out + self.mass * psi
        return out

    def apply_dag(self, psi, fields=None, Y=None):
        """D^dagger psi (= -D psi without a Dirac mass; M^T = -M with the selective source)."""
        if Y is None:
            Y = self.prepare(fields)
        if self.selective:
            return -(self._apply_csr(psi, Y) + self.source_apply(psi))
        out = -self._apply_csr(psi, Y)
        if self.mass:
            out = out + self.mass * psi
        return out

    def AA(self, psi, fields=None, Y=None):
        """D^dagger D psi (hermitian positive definite)."""
        if Y is None:
            Y = self.prepare(fields)
        return self.apply_dag(self.apply(psi, Y=Y), Y=Y)

    # ---- dense forms (numpy, small lattices) ---------------------------------------------------
    def dense_real(self, fields):
        """K - A(a) - h C as a dense real antisymmetric V x V matrix."""
        V = self.lat.V
        if self.geom is not None:
            s = to_numpy(self.split(fields)[1])
            a = self.geom.I.T @ s
            data = self.K0.data - self.geom.ent_eps * a[self.geom.ent_link]
        else:
            data = self.K0.data
        R = sps.csr_matrix((data, self.K0.indices, self.K0.indptr), shape=(V, V)).toarray()
        if self.h:
            R = R - self.h * self.C.toarray()
        return R

    def dense(self, fields):
        """Dense 2V x 2V complex matrix of the complex-linear part (no selective source)."""
        sigma = to_numpy(self.split(fields)[0])
        Yb = Yukawa(sigma, self.y).blocks()
        V = self.lat.V
        M = np.kron(self.dense_real(fields), np.eye(2)).astype(complex)
        M += sps.block_diag([sps.coo_matrix(Yb[0])] + list(Yb[1:])).toarray()
        if self.mass:
            M += self.mass * np.eye(2 * V)
        return M

    def dense_real4(self, fields):
        """M(h) as a dense real antisymmetric 4V x 4V matrix in the (site, flavour) basis."""
        M = realify(self.dense(fields), self.lat.V)
        if self.selective:
            M -= self.h_sel * np.kron(self.C.toarray(), np.diag(self.mask))
        return M

    def dense_G(self, fields):
        """The real propagator G^{ab}(x, z) = (M^-1)_{(x,a),(z,b)} as (V, 4, V, 4)."""
        V = self.lat.V
        return np.linalg.inv(self.dense_real4(fields)).reshape(V, 4, V, 4)

    def dense_S(self, fields):
        """The doublet propagator S = D_c^-1 as (V, 2, V, 2) (complex-linear operators only)."""
        assert not self.selective
        V = self.lat.V
        return np.linalg.inv(self.dense(fields)).reshape(V, 2, V, 2)
