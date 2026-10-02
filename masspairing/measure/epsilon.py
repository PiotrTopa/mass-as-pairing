"""Epsilon-channel fermion observables of the doublet model (no link fields, no source).

The on-site propagator block S(x, x) = [D^-1](x, x) is i n(x).tau with n real (anti-hermiticity and tau_2 K), so
tr S(x, x) = 0 on every configuration and the bilinear order parameter is the SU(2) triplet
n_a(x) = -(i/2) tr[tau_a S(x, x)]; O4(x) = det S(x, x) = |n(x)|^2. phi = (1/V) sum_x n(x), phi_stag = (1/V) sum_x
eps(x) n(x),
O4 = (1/V) sum_x |n(x)|^2. Stochastic estimator: Z2 noise diluted over the doublet index; the quadratic quantities use
U-statistics over distinct noise samples. Fermion correlator from a point source x0: G(t) = sum_xvec 1/2 eta_4(x0)
tr S((xvec, t0 + t), x0) (antiperiodic wrap sign included) and the positive channel C_f(t) = sum_xvec |S|_F^2.
"""

from __future__ import annotations

import numpy as np

from ..lattice import to_numpy
from .common import n_from_blocks


class FermionMeasure:
    def __init__(self, model, n_noise=4, exact=False, seed=0, cg_tol=1e-9):
        self.model, self.lat = model, model.lat
        self.n_noise, self.exact, self.cg_tol = int(n_noise), bool(exact), cg_tol
        self.rng = np.random.default_rng(seed)

    def _blocks_exact(self, fields):
        V = self.lat.V
        return self.model.D.dense_S(fields)[np.arange(V), :, np.arange(V), :]

    def _n_samples(self, fields, Y):
        if self.exact:
            return n_from_blocks(self._blocks_exact(fields))[None]
        lat, m = self.lat, self.model
        xp = lat.xp
        V = lat.V
        out = np.empty((self.n_noise, V, 3))
        for k in range(self.n_noise):
            S = np.zeros((V, 2, 2), complex)
            for b in range(2):
                eta = xp.zeros((V, 2), complex)
                eta[:, b] = xp.asarray(self.rng.choice([-1.0, 1.0], size=V))
                z = m.solve_Dinv(fields, eta, tol=self.cg_tol, Y=Y)
                S[:, :, b] = to_numpy(z * eta[:, b].conj()[:, None])
            out[k] = n_from_blocks(S)
        return out

    def bilinears(self, fields, Y=None) -> dict:
        lat = self.lat
        if Y is None:
            Y = self.model.D.prepare(fields)
        eps = to_numpy(lat.eps).reshape(-1)
        ns = self._n_samples(fields, Y)
        k = ns.shape[0]
        nbar = ns.mean(0)
        phi = nbar.mean(0)
        phi_stag = (nbar * eps[:, None]).mean(0)
        out = dict(
            phi=phi,
            phi_stag=phi_stag,
            phi_abs=float(np.linalg.norm(phi)),
            phi_stag_abs=float(np.linalg.norm(phi_stag)),
            O4_biased=float((nbar**2).sum(1).mean()),
        )
        if k >= 2:
            pk = ns.mean(1)
            psk = (ns * eps[None, :, None]).mean(1)
            out["phi_sq"] = float((np.linalg.norm(pk.sum(0)) ** 2 - (pk**2).sum()) / (k * (k - 1)))
            out["phi_stag_sq"] = float((np.linalg.norm(psk.sum(0)) ** 2 - (psk**2).sum()) / (k * (k - 1)))
            tot = ns.sum(0)
            out["O4"] = float((((tot**2).sum(1) - (ns**2).sum((0, 2))) / (k * (k - 1))).mean())
        else:
            out["phi_sq"] = out["phi_abs"] ** 2
            out["phi_stag_sq"] = out["phi_stag_abs"] ** 2
            out["O4"] = out["O4_biased"]
        return out

    def correlator(self, fields, Y=None, x0=None) -> dict:
        """Point-source propagator from x0 (a random site if None): G(t) (complex) and C_f(t)."""
        lat, m = self.lat, self.model
        xp = lat.xp
        if Y is None:
            Y = m.D.prepare(fields)
        L = lat.shape
        if x0 is None:
            x0 = tuple(int(self.rng.integers(n)) for n in L)
        i0 = int(np.ravel_multi_index(x0, L))
        S = np.zeros((lat.V, 2, 2), complex)
        for b in range(2):
            e = xp.zeros((lat.V, 2), complex)
            e[i0, b] = 1.0
            S[:, :, b] = to_numpy(m.solve_Dinv(fields, e, tol=self.cg_tol, Y=Y))
        S = S.reshape(L + (2, 2))
        eta4 = float(to_numpy(lat.eta[3])[x0])
        tr = 0.5 * eta4 * (S[..., 0, 0] + S[..., 1, 1])
        fro = (np.abs(S) ** 2).sum((-1, -2))
        Lt = L[3]
        G = np.zeros(Lt, complex)
        C = np.zeros(Lt)
        for dt in range(Lt):
            t = x0[3] + dt
            sign = 1.0
            if t >= Lt:
                t -= Lt
                sign = lat.bc[3]
            G[dt] = sign * tr[:, :, :, t].sum()
            C[dt] = fro[:, :, :, t].sum()
        return dict(Gf=G, Cf=C, x0=np.array(x0))
