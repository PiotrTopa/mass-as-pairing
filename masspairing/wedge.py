"""Link fields of the sum-of-squares completion of the 10-channel plaquette term ("wedge" model).

The fermionic content of the completed term on a plaquette quad Q = (x, x'; y, y') (x even, x' = x + mu + sg nu,
y = x + mu, y' = x + sg nu) is

    g1 (Kt_nu + Kt_nu')^2 + g2 (Kt_mu + Kt_mu')^2,   g1 = g10 - g6,  g2 = g10 + g6,  |g6| <= g10,

with K_l = sum_a chi^a(u) chi^a(v) the SU(4)-singlet link bilinear (u even, v odd) and Kt_l = s_l K_l oriented by
the sign of the free propagator. A real Gaussian Hubbard-Stratonovich field per (quad, pair), prior s^2 / (4 g_j),
gives link coefficients a = I^T s and the fermion operator (K - A(a)) (x) 1_2 + i y sigma.tau, with
A(a) = sum_l a_l (e_u e_v^T - e_v e_u^T); K - A is real antisymmetric and flavour-blind, so the weight stays
positive (tau_2 K argument).

Quad sets: "all" (both orientations per even site: every square of the lattice; shift symmetric), "checkerboard"
(one orientation; breaks the one-site shift) and "links" (the per-link pure completion model: one field per link with
prior s^2 / (4 c_l g), c_l the number of "all" squares through link l, inducing c_l g K_l^2 without the plaquette term).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sps

from .lattice import Lattice, kinetic_matrix, to_numpy


class WedgeGeometry:
    """Links, quads and the field-to-link incidence on a lattice.

    links: distinct nearest-neighbour pairs {u even, v odd} with (lu, lv, ldir, lpos) and the orientation sign
    lsign = -sign K[u, v]. quads: (x, x', y, y') with links [nu, nu', mu, mu'] = [{x,y'}, {x',y}, {x,y}, {x',y'}].
    fields: (quad, j) with j = 0 the nu pair (g1) and j = 1 the mu pair (g2); a pair with zero coupling has no field.
    Incidence I (n_fields x n_links) with entries lsign: a = I^T s.
    """

    def __init__(self, lat: Lattice, quads="all", g10=0.0, g6=0.0):
        self.lat = lat
        self.quad_set = quads
        self.g10, self.g6 = float(g10), float(g6)
        assert abs(self.g6) <= self.g10 + 1e-15, "the sign-free wedge requires |g6| <= g10"
        self.g = (self.g10 - self.g6, self.g10 + self.g6)
        shape, d, V = lat.shape, lat.d, lat.V
        idx = np.arange(V).reshape(shape)
        coords = lat.coords.reshape(d, V)
        eps = lat.eps_np.reshape(V)
        K0 = kinetic_matrix(lat)
        self.K0 = K0
        # ---- links
        pair_id = {}
        lu, lv, ldir, lpos = [], [], [], []
        self.link_of = np.full((d, V), -1, np.int64)
        for mu in range(d):
            j = np.roll(idx, -1, axis=mu).reshape(V)
            for x in range(V):
                key = (min(x, j[x]), max(x, j[x]))
                link = pair_id.get(key)
                if link is None:
                    link = len(lu)
                    pair_id[key] = link
                    u, v = (x, j[x]) if eps[x] > 0 else (j[x], x)
                    lu.append(u)
                    lv.append(v)
                    ldir.append(mu)
                    lpos.append(x)
                self.link_of[mu, x] = link
        self.n_links = len(lu)
        self.lu, self.lv, self.ldir, self.lpos = (np.asarray(a) for a in (lu, lv, ldir, lpos))
        self.lsign = -np.sign(np.asarray(K0[self.lu, self.lv]).reshape(-1))
        assert np.all(self.lsign != 0)
        self.pair_id = pair_id
        # ---- quads
        seen = set()
        qsites, qlinks, qorient = [], [], []
        for x in range(V):
            if eps[x] < 0:
                continue
            c = coords[:, x]
            for mu in range(d):
                for nu in range(mu + 1, d):
                    for sg in (+1,) if quads == "checkerboard" else (+1, -1):

                        def sh(dmu, dnu, mu=mu, nu=nu, c=c):
                            cc = c.copy()
                            cc[mu] = (cc[mu] + dmu) % shape[mu]
                            cc[nu] = (cc[nu] + dnu) % shape[nu]
                            return int(idx[tuple(cc)])

                        xp_, y, yp = sh(1, sg), sh(1, 0), sh(0, sg)
                        key = frozenset((x, xp_, y, yp))
                        if len(key) < 4 or key in seen:
                            continue
                        seen.add(key)
                        links = [pair_id[(min(a, b), max(a, b))] for a, b in ((x, yp), (xp_, y), (x, y), (xp_, yp))]
                        qsites.append((x, xp_, y, yp))
                        qlinks.append(links)
                        qorient.append((mu, nu, sg))
        self.n_quads = len(qsites)
        self.qsites, self.qlinks, self.qorient = np.asarray(qsites), np.asarray(qlinks), qorient
        # ---- fields and incidence
        rows, cols, vals, f_quad, f_pair = [], [], [], [], []
        for q in range(self.n_quads if quads != "links" else 0):
            for jj in (0, 1):
                if self.g[jj] <= 0.0:
                    continue
                f = len(f_quad)
                f_quad.append(q)
                f_pair.append(jj)
                for link in self.qlinks[q, 2 * jj : 2 * jj + 2]:
                    rows.append(f)
                    cols.append(link)
                    vals.append(self.lsign[link])
        self.f_link = None
        if quads == "links":
            assert self.g6 == 0.0, "the per-link completion model has no g6"
            self.link_count = np.bincount(self.qlinks.reshape(-1), minlength=self.n_links)
            nl = self.n_links if self.g10 > 0.0 else 0
            self.n_fields = nl
            self.f_quad, self.f_pair, self.f_link = np.full(nl, -1), np.zeros(nl, int), np.arange(nl)
            self.f_g = self.link_count[:nl] * self.g10
            self.I = sps.csr_matrix(
                (self.lsign[:nl].astype(float), (np.arange(nl), np.arange(nl))),
                shape=(nl, self.n_links),
            )
            self.inv4g = 1.0 / (4.0 * self.f_g)
        else:
            self.n_fields = len(f_quad)
            self.f_quad, self.f_pair = np.asarray(f_quad, int), np.asarray(f_pair, int)
            self.I = sps.csr_matrix((np.asarray(vals, float), (rows, cols)), shape=(self.n_fields, self.n_links))
            self.inv4g = np.asarray([1.0 / (4.0 * self.g[jj]) for jj in self.f_pair]) if self.n_fields else np.zeros(0)
            self.f_g = np.asarray([self.g[jj] for jj in self.f_pair], float)
        # ---- entry tables of K0 - A(a): entry (x, y) has value K0[x, y] - eps(x) a[link{x, y}]
        rows_e = np.repeat(np.arange(V), np.diff(K0.indptr))
        cols_e = K0.indices
        self.ent_link = np.asarray([pair_id[(min(a, b), max(a, b))] for a, b in zip(rows_e, cols_e, strict=False)])
        self.ent_eps = eps[rows_e]
        self.IT = sps.csr_matrix(self.I.T)
        self._dev = {}

    def dev(self, xp):
        """Tables on the array module xp (cached)."""
        key = xp.__name__
        t = self._dev.get(key)
        if t is None:
            t = dict(
                lu=xp.asarray(self.lu),
                lv=xp.asarray(self.lv),
                ent_link=xp.asarray(self.ent_link),
                ent_eps=xp.asarray(self.ent_eps),
                inv4g=xp.asarray(self.inv4g),
                data0=xp.asarray(self.K0.data),
            )
            if xp is np:
                t["I"], t["IT"] = self.I, self.IT
            else:
                import cupyx.scipy.sparse as csp

                t["I"], t["IT"] = csp.csr_matrix(self.I), csp.csr_matrix(self.IT)
            self._dev[key] = t
        return t

    def start(self, rng, how="hot"):
        """Initial link fields: "zero", "hot" (Gaussian of the prior) or "dimer" (columnar pattern)."""
        g = self
        if how == "zero":
            return np.zeros(g.n_fields)
        if g.f_link is not None:
            if how == "dimer":
                co = g.lat.coords.reshape(g.lat.d, -1)
                return np.sqrt(2.0 * g.f_g) * (-1.0) ** co[g.ldir[g.f_link], g.lpos[g.f_link]]
            return rng.normal(size=g.n_fields) * np.sqrt(2.0 * g.f_g)
        if how == "dimer":
            s = np.zeros(g.n_fields)
            for f in range(g.n_fields):
                q = g.f_quad[f]
                mu = g.qorient[q][0]
                x = g.qsites[q, 0]
                s[f] = math.sqrt(2.0 * g.g[g.f_pair[f]]) * (-1.0) ** g.lat.coords[mu].reshape(-1)[x]
            return s
        return rng.normal(size=g.n_fields) * np.sqrt(np.asarray([2.0 * g.g[j] for j in g.f_pair]))


def link_field_observables(geom: WedgeGeometry, s) -> dict:
    """Mean s^2, per-pair means and structure factors of the link fields.

    Structure factors over the even sublattice per quad orientation (or per link direction for the per-link model)
    at p = 0, pi e_rho, p_min e_rho and pi e_rho + p_min e_rho' (averaged over rho, rho' != rho).
    """
    g = geom
    lat = g.lat
    s = to_numpy(s)
    out = dict(s2=float(np.mean(s**2)) if s.size else 0.0)
    if not s.size:
        return out
    d, V = lat.d, lat.V
    L = lat.shape

    def spectra(P, Sp):
        Sp["0"].append(P[(0,) * d])
        for rho in range(d):
            i = [0] * d
            i[rho] = L[rho] // 2
            Sp["pi"].append(P[tuple(i)])
            i = [0] * d
            i[rho] = 1
            Sp["pmin"].append(P[tuple(i)])
            for rho2 in range(d):
                if rho2 != rho:
                    i = [0] * d
                    i[rho] = L[rho] // 2
                    i[rho2] = 1
                    Sp["pi_pmin"].append(P[tuple(i)])

    if g.f_link is not None:
        out["s_mean_0"] = float(s.mean())
        out["s2_0"] = float((s**2).mean())
        Sp = {"0": [], "pi": [], "pmin": [], "pi_pmin": []}
        for mu in range(d):
            fsel = g.ldir[g.f_link] == mu
            field = np.zeros(V)
            field[g.lpos[g.f_link[fsel]]] = s[fsel]
            P = np.abs(np.fft.fftn(field.reshape(L))) ** 2 / fsel.sum()
            spectra(P, Sp)
        for key, v in Sp.items():
            out[f"Ss_{key}_0"] = float(np.mean(v))
        return out
    for j in (0, 1):
        sel = g.f_pair == j
        if not sel.any():
            continue
        out[f"s_mean_{j}"] = float(s[sel].mean())
        out[f"s2_{j}"] = float((s[sel] ** 2).mean())
        Sp = {"0": [], "pi": [], "pmin": [], "pi_pmin": []}
        for o in sorted(set(g.qorient)):
            fsel = sel & np.array([g.qorient[q] == o for q in g.f_quad])
            if not fsel.any():
                continue
            field = np.zeros(V)
            field[g.qsites[g.f_quad[fsel], 0]] = s[fsel]
            F = np.fft.fftn(field.reshape(L))
            P = np.abs(F) ** 2 / fsel.sum()
            spectra(P, Sp)
        for key, v in Sp.items():
            out[f"Ss_{key}_{j}"] = float(np.mean(v))
    return out
