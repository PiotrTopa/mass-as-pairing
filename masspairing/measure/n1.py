"""Fermion observables in the real flavour basis: per-flavour pair channels and the taste channels of model N1.

The real propagator G = M^-1 (dense, ``exact``) or Z2 noise diluted over the four real flavours (4 solves per sample;
G^{ab}(u, v) ~ Z[u, a, b] E[v, b]); every product of two propagators uses distinct noise samples.

* epsilon channel: n_a(x) = 1/4 tr[Gamma_a G(x, x)] -> phi, phi_stag, phi_sq, phi_stag_sq, O4.
* links: Kt_l = s_l tr G(u, v) -> E_bond, dimer, dimer_sq(_par/_perp).
* pair channel of the pattern C per real flavour, O^(a)(x) = 1/2 sum_z C[x, z] chi^a(x) chi^a(z): phi_f (4,) =
<O^(a)>/V,
  sublattice parts, corner momenta; chi_f (4,) = (1/V) sum <O^(a) O^(a)> with all Wick terms at p = 0, its connected
  part
  chi_f_conn, corners and p_min; with ``summed`` the flavour-summed susceptibility chi_f_sum (all 16 flavour blocks).
* flavour-resolved point-source correlators Gf_f, Cf_f (exact: averaged over all sources).
* six-fermion composite (exact): T^F(x) = prod_{b in F} chi^b(x) over flavour triples F -> phi_T = (1/V) 1/2 sum C <T
T>,
  and its time-slice correlator C_T.
* taste channels (model N1, pattern C_chi): heavy pattern C_R = C_chi (keys phi_f, chi_f, ...), light pattern
  C_L = P_L C_asd P_L (phi_L, chi_L, chi_L_sum = V<phi_L^2>, ...), the light (6,1,1) channel phi6L (sigma triplet on
  G_L = P_L G P_L), the momentum readouts GL_p0, GR_p0, G_p0 = 4 sum sin^2 p0 |G(p0)|_F^2 / 32 (/64 full) on the
  16-corner x 4-flavour block at the smallest twisted momentum (1 for a free massless doublet); exact only: the
  all-source time-slice correlators CL_t, GL_t, CR_t, GR_t and the composites phi_T_L, C_T_L, phi_T_R, C_T_R.
"""

from __future__ import annotations

import numpy as np

from ..flavour import GAMMA, pack, unpack
from ..lattice import to_numpy
from ..patterns import TasteProjector, chiral_mass, source_matrix
from .common import ustat_pairs as _ustat_pairs
from .epsilon import FermionMeasure


def _pf_small(A):
    """Pfaffian of antisymmetric matrices A (..., n, n), n ≤ 6 even, by expansion along the first row (vectorised)."""
    n = A.shape[-1]
    if n == 0:
        return np.ones(A.shape[:-2])
    if n == 2:
        return A[..., 0, 1]
    out = 0.0
    for j in range(1, n):
        keep = [i for i in range(n) if i not in (0, j)]
        sub = A[..., keep, :][..., :, keep]
        out = out + (-1.0) ** (j + 1) * A[..., 0, j] * _pf_small(sub)
    return out


class N1Measure(FermionMeasure):
    """Real-basis fermion measure (see the module docstring). ``exact`` uses the dense inverse of M; otherwise
    ``n_noise`` >= 2 Z2 samples diluted over the four real flavours. ``taste`` (default: pattern "chiral") adds the
    light/heavy taste channels; ``six_fermion`` the composite channels (exact only)."""

    def __init__(
        self,
        model,
        n_noise=4,
        exact=False,
        seed=0,
        cg_tol=1e-9,
        six_fermion=True,
        taste=None,
        comp=(0, 3),
        dual_sign=+1,
    ):
        super().__init__(model, n_noise=n_noise, exact=exact, seed=seed, cg_tol=cg_tol)
        self.D = model.D
        if model.geom is not None:
            self.geom = model.geom
        else:  # link tables only (bond energies of models without link fields)
            from ..wedge import WedgeGeometry

            self.geom = WedgeGeometry(model.lat, "all", 0.0, 0.0)
        lat = self.lat
        d, V = lat.d, lat.V
        self.eps = to_numpy(lat.eps).reshape(V)
        self.even = np.where(self.eps > 0)[0]
        self.odd = np.where(self.eps < 0)[0]
        self.coords = lat.coords.reshape(d, V)
        pattern = model.D.pattern
        self.C0 = model.D.C if model.D.selective else source_matrix(lat, pattern)
        self.C0d = None
        self.taste = (isinstance(pattern, str) and pattern == "chiral") if taste is None else bool(taste)
        if self.taste:
            self.tp = TasteProjector(lat, comp, dual_sign)
            self.CR = chiral_mass(lat, comp, dual_sign).tocsr()  # the heavy (sourced) pattern C_χ
            self.CL = (
                self.tp.dense_PL() @ chiral_mass(lat, comp, -dual_sign).toarray() @ self.tp.dense_PL()
            )  # C_L = P_L C_asd P_L
            self.CL = 0.5 * (self.CL - self.CL.T)
            self.CL[np.abs(self.CL) < 1e-14] = 0.0
            th = [0.5 if lat.bc[mu] == -1 else 0.0 for mu in range(d)]
            self.p0 = np.array(
                [2.0 * np.pi * (th[mu] if th[mu] else 1.0) / lat.shape[mu] for mu in range(d)]
            )  # smallest twisted momentum
            self.p0_signs = np.array(list(np.ndindex(*(2,) * d))) * -2 + 1  # (16, d) sign choices
            corners = [np.array(c, float) * np.pi for c in np.ndindex(*(2,) * d)]
            self.corner_shifts = np.asarray(corners)
        corners = [np.array(c, float) * np.pi for c in np.ndindex(*(2,) * d)]
        self.corners = np.asarray(corners)
        self.Pc = np.cos(self.corners @ self.coords)  # (16, V) real ±1
        pm = [np.array([2.0 * np.pi / lat.shape[r] if k == r else 0.0 for k in range(d)]) for r in range(d)]
        self.Ppm = np.exp(1j * (np.asarray(pm) @ self.coords))  # (d, V)
        mask = model.D.mask if model.D.selective else np.ones(4)
        self.light = [a for a in range(4) if mask[a] == 0.0] if np.any(mask) else [0, 1, 2]
        self.six_fermion = bool(six_fermion) and len(self.light) == 3
        self.six_fermion_taste = bool(six_fermion)

    # ---- propagator samples
    # ----------------------------------------------------------------------------------------
    def _samples(self, fields, Y):
        """[(Z (V,4,4) real with Z[u,a,b] = (M⁻¹ξ_b)_a(u), E (V,4) with E[v,b] = ξ_b(v))]."""
        lat, m = self.lat, self.model
        xp = lat.xp
        V = lat.V
        out = []
        for _k in range(self.n_noise):
            Z = np.zeros((V, 4, 4))
            E = np.zeros((V, 4))
            for b in range(4):
                e = self.rng.choice([-1.0, 1.0], size=V)
                eta = np.zeros((V, 2), complex)
                if b % 2 == 0:
                    eta[:, b // 2] = e
                else:
                    eta[:, b // 2] = 1j * e
                z = m.solve_Dinv(fields, xp.asarray(eta), tol=self.cg_tol, Y=Y)
                Z[:, :, b] = unpack(z)
                E[:, b] = e
            out.append((Z, E))
        return out

    # ---- the observables
    # -------------------------------------------------------------------------------------------
    def bilinears(self, fields, Y=None):
        lat, g = self.lat, self.geom
        d, V = lat.d, lat.V
        if Y is None:
            Y = self.model.D.prepare(fields)
        out = {}
        if self.exact:
            G = self.D.dense_G(fields)
            samples = [None]
        else:
            samples = self._samples(fields, Y)
        k = len(samples)
        idx = np.arange(V)
        # (a) on-site: n_a(x) = ¼ tr[Γ_a G(x,x)]
        if self.exact:
            Gxx = [G[idx, :, idx, :]]
        else:
            Gxx = [np.einsum("uab,ub->uab", Z, E) for Z, E in samples]  # G^{ab}(u,u) ≈ Z[u,a,b]E[u,b]
        ns = np.stack([0.25 * np.einsum("cab,uab->uc", GAMMA, gx) for gx in Gxx])  # (k, V, 3)
        nbar = ns.mean(0)
        phi = nbar.mean(0)
        phi_stag = (nbar * self.eps[:, None]).mean(0)
        out.update(
            phi=phi,
            phi_stag=phi_stag,
            phi_abs=float(np.linalg.norm(phi)),
            phi_stag_abs=float(np.linalg.norm(phi_stag)),
            O4_biased=float((nbar**2).sum(1).mean()),
        )
        if k >= 2:
            pk = ns.mean(1)
            psk = (ns * self.eps[None, :, None]).mean(1)
            out["phi_sq"] = float(_ustat_pairs(pk))
            out["phi_stag_sq"] = float(_ustat_pairs(psk))
            tot = ns.sum(0)
            out["O4"] = float((((tot**2).sum(1) - (ns**2).sum((0, 2))) / (k * (k - 1))).mean())
        else:
            out["phi_sq"] = out["phi_abs"] ** 2
            out["phi_stag_sq"] = out["phi_stag_abs"] ** 2
            out["O4"] = out["O4_biased"]
        # (b) links: K̃_l = s_l tr G(u,v)
        if self.exact:
            Kl = np.stack([g.lsign * np.einsum("naa->n", G[g.lu, :, g.lv, :])])
        else:
            Kl = np.stack([g.lsign * np.einsum("nbb,nb->n", Z[g.lu], E[g.lv]) for Z, E in samples])
        Kbar = Kl.mean(0)
        out["E_bond"] = np.array([Kbar[g.ldir == mu].mean() for mu in range(d)])
        Dk = np.zeros((k, d, d))
        for mu in range(d):
            sel = g.ldir == mu
            for nu in range(d):
                sgn = (-1.0) ** self.coords[nu, g.lpos[sel]]
                Dk[:, mu, nu] = (Kl[:, sel] * sgn).mean(1)
        out["dimer"] = Dk.mean(0)
        D2 = np.array([[_ustat_pairs(Dk[:, mu, nu]) for nu in range(d)] for mu in range(d)]) if k >= 2 else Dk[0] ** 2
        out["dimer_sq"] = D2
        out["dimer_sq_par"] = float(np.trace(D2) / d)
        out["dimer_sq_perp"] = float((D2.sum() - np.trace(D2)) / (d * (d - 1)))
        # (c) per-flavour pair channel
        out.update(self.pair_channel(G if self.exact else None, None if self.exact else samples))
        # (d) six-fermion (exact)
        if self.exact and self.six_fermion:
            out.update(self.six_fermion_channel(G))
        # (e) the taste channels (light/heavy doublets of the chiral mass)
        if self.taste:
            out.update(self.taste_channels(G if self.exact else None, None if self.exact else samples))
        return out

    def pair_channel(self, G=None, samples=None, C=None, prefix="f", summed=False):
        """Per-flavour pair channel of the pattern C (default: the model's pattern): phi_<prefix>, chi_<prefix>, … as
        in the
        class docstring.  summed=True adds the flavour-SUMMED susceptibility of O = Σ_a O^{(a)} with the cross-flavour
        Wick terms, chi_<prefix>_sum (p = 0; = V⟨φ²⟩ of the summed amplitude), _sum_conn, _sum_corners, _sum_pmin.
        """
        V, d = self.lat.V, self.lat.d
        C = self.C0 if C is None else C
        nc = len(self.corners)
        phi = np.zeros((4, V))
        phi_c = np.zeros((4, nc))
        chi_c = np.zeros((4, nc))
        chi_conn = np.zeros(4)
        chi_pm = np.zeros(4)
        s_c = np.zeros(nc)
        s_conn = 0.0
        s_pm = 0.0
        pairs = [(a, a) for a in range(4)] + ([(a, b) for a in range(4) for b in range(4) if a != b] if summed else [])
        if G is not None:
            Cd = C if isinstance(C, np.ndarray) else C.toarray()
            if C is self.C0:
                self.C0d = Cd
            conn_sum = np.zeros((V, V)) if summed else None
            for a, b in pairs:
                A = G[:, a, :, b]  # (V,V); antisymmetric for a = b
                CA = Cd @ A
                AC = A @ Cd
                CAC = CA @ Cd
                conn = 0.25 * A * CAC - 0.25 * CA * AC  # (V,V): the two exchange terms
                if a == b:
                    phi[a] = 0.5 * (Cd * A).sum(1)  # φ(x) = ½ Σ_z C[x,z] A[x,z]
                    full = phi[a][:, None] * phi[a][None, :] + conn
                    phi_c[a] = (self.Pc @ phi[a]) / V
                    chi_c[a] = np.einsum("kx,xz,kz->k", self.Pc, full, self.Pc) / V
                    chi_conn[a] = conn.sum() / V
                    chi_pm[a] = np.mean([np.einsum("x,xz,z->", p, full, p.conj()).real for p in self.Ppm]) / V
                if summed:
                    conn_sum += conn
            if summed:
                ps = phi.sum(0)
                full = ps[:, None] * ps[None, :] + conn_sum
                s_c = np.einsum("kx,xz,kz->k", self.Pc, full, self.Pc) / V
                s_conn = conn_sum.sum() / V
                s_pm = np.mean([np.einsum("x,xz,z->", p, full, p.conj()).real for p in self.Ppm]) / V
        else:
            k = len(samples)
            assert k >= 2, "the pair channel needs n_noise >= 2"
            npair = k * (k - 1)
            z = [Z[:, np.arange(4), np.arange(4)] for Z, E in samples]  # (V,4): z_a = Z[:,a,a]
            e = [E for Z, E in samples]
            cz = [C @ zi for zi in z]
            ce = [C @ ei for ei in e]  # (V,4)
            ph = [0.5 * zi * cei for zi, cei in zip(z, ce, strict=False)]  # φ_i(x) per flavour (V,4)
            phi = np.mean(ph, 0).T  # (4,V)
            phi_c = (self.Pc @ phi.T).T / V
            if summed:
                zab = [Z.reshape(V, 16) for Z, E in samples]  # Z[:,a,b] as (V, 16), index 4a + b
                czab = [C @ zi for zi in zab]
                eb = [np.repeat(E, 4, axis=1) for Z, E in samples]  # E[:, b] at index 4a + b
                ceb = [C @ ei for ei in eb]
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    # disc: φ_i(x)φ_j(z);  term2: −¼ [z_i(Cz_j)](x) [E_i(CE_j)](z);  term3: +¼ [(Cz_i) z_j](x)
                    # [E_i(CE_j)](z)
                    F2 = z[i] * cz[j]
                    F3 = cz[i] * z[j]
                    H = e[i] * ce[j]  # (V,4)
                    Fx = ph[i]
                    Hx = ph[j]
                    for kc in range(nc):
                        p = self.Pc[kc]
                        chi_c[:, kc] += (
                            ((p @ Fx) * (p @ Hx) - 0.25 * (p @ F2) * (p @ H) + 0.25 * (p @ F3) * (p @ H)) / V / npair
                        )
                    chi_conn += ((-0.25 * F2.sum(0) + 0.25 * F3.sum(0)) * H.sum(0)) / V / npair
                    for p in self.Ppm:
                        chi_pm += (
                            (
                                ((p @ Fx) * (p.conj() @ Hx)).real
                                - 0.25 * ((p @ F2) * (p.conj() @ H)).real
                                + 0.25 * ((p @ F3) * (p.conj() @ H)).real
                            )
                            / V
                            / npair
                            / d
                        )
                    if summed:  # all 16 flavour blocks (a,b) of the connected part
                        F2s = zab[i] * czab[j]
                        F3s = czab[i] * zab[j]
                        Hs = eb[i] * ceb[j]
                        Fxs = Fx.sum(1)
                        Hxs = Hx.sum(1)
                        for kc in range(nc):
                            p = self.Pc[kc]
                            s_c[kc] += (
                                (
                                    (p @ Fxs) * (p @ Hxs)
                                    + (-0.25 * (p @ F2s) * (p @ Hs) + 0.25 * (p @ F3s) * (p @ Hs)).sum()
                                )
                                / V
                                / npair
                            )
                        s_conn += ((-0.25 * F2s.sum(0) + 0.25 * F3s.sum(0)) * Hs.sum(0)).sum() / V / npair
                        for p in self.Ppm:
                            s_pm += (
                                (
                                    ((p @ Fxs) * (p.conj() @ Hxs)).real
                                    + (
                                        -0.25 * ((p @ F2s) * (p.conj() @ Hs)).real
                                        + 0.25 * ((p @ F3s) * (p.conj() @ Hs)).real
                                    ).sum()
                                )
                                / V
                                / npair
                                / d
                            )
        f = prefix
        out = {
            f"phi_{f}": phi.sum(1) / V,
            f"phi_{f}_even": phi[:, self.even].sum(1) / V,
            f"phi_{f}_odd": phi[:, self.odd].sum(1) / V,
            f"phi_{f}_corners": phi_c,
            f"chi_{f}": chi_c[:, 0].copy(),
            f"chi_{f}_corners": chi_c,
            f"chi_{f}_conn": chi_conn,
            f"chi_{f}_pmin": chi_pm,
        }
        if summed:
            out.update(
                {
                    f"chi_{f}_sum": float(s_c[0]),
                    f"chi_{f}_sum_corners": s_c,
                    f"chi_{f}_sum_conn": float(s_conn),
                    f"chi_{f}_sum_pmin": float(s_pm),
                }
            )
        return out

    def six_fermion_channel(self, G, flavour_sets=None, C=None, keys=("phi_T", "C_T")):
        """phi_T and C_T(t) from the dense G: T^F(x) = Π_{b∈F} χ^b(x) over the flavour triples F in `flavour_sets`
        (default:
        the one triple self.light), phi_T = (1/V) ½ Σ_F Σ_{xz} C[x,z] ⟨T^F(x)T^F(z)⟩ with the pattern C (default
        self.C0; dense or
        sparse) and C_T(t) = (1/V) Σ_F Σ_{x0,x⃗} wrap ⟨T^F(x⃗, t0+t) T^F(x0)⟩."""
        lat = self.lat
        V, Lt = lat.V, lat.shape[-1]
        sets = [self.light] if flavour_sets is None else flavour_sets
        if C is None:
            if self.C0d is None:
                self.C0d = self.C0.toarray()
            Cd = self.C0d
        else:
            Cd = C if isinstance(C, np.ndarray) else C.toarray()
        xs, zs = np.nonzero(Cd)
        t_of = self.coords[-1]
        phi_T = 0.0
        CT = np.zeros(Lt)
        for F in sets:
            GF = G[:, F][:, :, :, F]  # (V,3,V,3)
            Gd = GF[np.arange(V), :, np.arange(V), :]  # (V,3,3)

            def pf6(xs, zs):
                n = len(xs)
                A = np.zeros((n, 6, 6))
                A[:, :3, :3] = Gd[xs]
                A[:, 3:, 3:] = Gd[zs]
                A[:, :3, 3:] = GF[xs, :, zs, :]
                A[:, 3:, :3] = -A[:, :3, 3:].transpose(0, 2, 1)
                return _pf_small(A)

            chunk = max(1, 200000 // V)
            for i0 in range(0, len(xs), 4 * 200000):
                phi_T += (
                    0.5
                    * (
                        Cd[xs[i0 : i0 + 4 * 200000], zs[i0 : i0 + 4 * 200000]]
                        * pf6(xs[i0 : i0 + 4 * 200000], zs[i0 : i0 + 4 * 200000])
                    ).sum()
                    / V
                )
            # time-slice correlator, all sources: C_T(t) = (1/V) Σ_{x0} Σ_x⃗ sign ⟨T(x⃗, t0+t) T(x0)⟩
            for x0 in range(0, V, chunk):
                src = np.repeat(np.arange(x0, min(V, x0 + chunk)), V)
                dst = np.tile(np.arange(V), min(chunk, V - x0))
                vals = pf6(dst, src)
                dt = (t_of[dst] - t_of[src]) % Lt
                wrap = np.where(t_of[dst] < t_of[src], float(lat.bc[-1]), 1.0)
                np.add.at(CT, dt, wrap * vals)
        return {keys[0]: float(phi_T), keys[1]: CT / V}

    def correlator(self, fields, Y=None, x0=None):
        """Flavour-resolved point-source correlators Gf_f (4, Lt) and Cf_f (4, Lt); exact: averaged over all sources."""
        lat, m = self.lat, self.model
        L = lat.shape
        Lt = L[-1]
        V = lat.V
        eta4 = to_numpy(lat.eta[-1]).reshape(V)
        t_of = self.coords[-1]
        if Y is None:
            Y = m.D.prepare(fields)
        if self.exact:
            G = self.D.dense_G(fields)  # (V,4,V,4)
            Gf = np.zeros((4, Lt))
            Cf = np.zeros((4, Lt))
            for x0 in range(V):
                dt = (t_of - t_of[x0]) % Lt
                wrap = np.where(t_of < t_of[x0], float(lat.bc[-1]), 1.0)
                Ga = G[:, :, x0, :]  # (V, b, a)
                for a in range(4):
                    np.add.at(Gf[a], dt, wrap * eta4[x0] * Ga[:, a, a])
                    np.add.at(Cf[a], dt, (Ga[:, :, a] ** 2).sum(1))
            return dict(Gf_f=Gf / V, Cf_f=Cf / V, x0=np.array([-1]))
        if x0 is None:
            x0 = tuple(int(self.rng.integers(l)) for l in L)
        i0 = int(np.ravel_multi_index(x0, L))
        xp = lat.xp
        Gf = np.zeros((4, Lt))
        Cf = np.zeros((4, Lt))
        dt = (t_of - t_of[i0]) % Lt
        wrap = np.where(t_of < t_of[i0], float(lat.bc[-1]), 1.0)
        for a in range(4):
            e = np.zeros((V, 2), complex)
            e[i0, a // 2] = 1.0 if a % 2 == 0 else 1j
            z = unpack(m.solve_Dinv(fields, xp.asarray(e), tol=self.cg_tol, Y=Y))  # (V,4): G^{ba}(x, x0)
            np.add.at(Gf[a], dt, wrap * eta4[i0] * z[:, a])
            np.add.at(Cf[a], dt, (z**2).sum(1))
        return dict(Gf_f=Gf, Cf_f=Cf, x0=np.array(x0))

    # ---- taste channels of the chiral mass
    # ----------------------------------------------------------------------
    def _project_G(self, G, which):
        """P G P on the dense (V,4,V,4) propagator by FFT (P flavour-blind)."""
        P = self.tp.PL if which == "L" else self.tp.PR
        V = self.lat.V
        H = P(np.ascontiguousarray(G.transpose(2, 3, 0, 1)).reshape(V, -1)).reshape(V, 4, V, 4)  # G P (z first)
        return P(np.ascontiguousarray(H.transpose(2, 3, 0, 1)).reshape(V, -1)).reshape(V, 4, V, 4)  # P (G P)

    def _timeslice(self, G, key, wrap_sign=True):
        """Positive channel C(t) = (1/V) Σ_{x0,x⃗} Σ_ab G^{ab}(x,x0)² and the linear channel G(t) = (1/V) Σ_{x0,x⃗}
        wrap η₄(x0)
        Σ_a G^{aa}(x,x0) from a dense (V,4,V,4) propagator, all sources (numpy)."""
        lat = self.lat
        Lt, V = lat.shape[-1], lat.V
        ns = V // Lt
        Gt = G.reshape(ns, Lt, 4, ns, Lt, 4)
        S2 = np.einsum("xtaysb,xtaysb->ts", Gt, Gt)  # (Lt, Lt) Σ_{x⃗,x⃗0,ab} G²
        eta4 = to_numpy(lat.eta[-1]).reshape(ns, Lt)
        S1 = np.einsum("xtays,ys->ts", np.einsum("xtaysa->xtays", Gt), eta4)  # Σ_{x⃗,x⃗0,a} η₄(x0) G^{aa}
        C = np.zeros(Lt)
        Gl = np.zeros(Lt)
        for dt in range(Lt):
            for t0 in range(Lt):
                t = (t0 + dt) % Lt
                w = float(lat.bc[-1]) if (wrap_sign and t < t0) else 1.0
                C[dt] += S2[t, t0]
                Gl[dt] += w * S1[t, t0]
        return {f"C{key}_t": C / V, f"G{key}_t": Gl / V}

    def _p0_blocks(self, G, key):
        """‖G(p₀)‖²_F over the 16-corner × 4-flavour block at the smallest twisted momentum p₀ (averaged over its 2⁴
        sign
        images), times 4Σ_μ sin² p₀_μ / 64: = 1 for the free massless doublet (4 flavours), → 0 for a pole (×
        p²/(p²+m²)) or a
        Luttinger zero.  G dense (V,4,V,4)."""
        lat = self.lat
        V = lat.V
        co = self.coords
        vals = []
        for sg in self.p0_signs:
            p = sg * self.p0
            mom = self.corner_shifts + p[None, :]  # (16, d)
            E = np.exp(1j * (mom @ co)) / np.sqrt(V)  # (16, V)
            T = np.tensordot(E.conj(), G, axes=([1], [0]))  # (16,4,V,4)
            B = np.tensordot(T, E.T, axes=([2], [0]))  # (16,4,4,16)
            vals.append((np.abs(B) ** 2).sum())
        norm = (
            4.0 * np.sum(np.sin(self.p0) ** 2) / (64.0 if key == "" else 32.0)
        )  # per doublet: 1 for a free massless doublet
        return {f"G{key}_p0": float(np.mean(vals) * norm)}

    def _p0_blocks_stochastic(self, blocks, key):
        """The momentum readout from noise samples [(Z (V,4,4), E (V,4))] (already projected for L/R): G(p₀)_{Aa,Bb} ≈
        ẑ_i[A,a,b] ê_i[B,b] with ẑ = E_p^† Z, ê = E_pᵀ E; ‖G(p₀)‖² by the U-statistic over ordered pairs of distinct
        samples.
        """
        lat = self.lat
        V = lat.V
        co = self.coords
        k = len(blocks)
        assert k >= 2
        vals = []
        for sg in self.p0_signs:
            mom = self.corner_shifts + (sg * self.p0)[None, :]
            Ep = np.exp(1j * (mom @ co)) / np.sqrt(V)  # (16, V)
            g = []
            for Z, E in blocks:
                zh = np.tensordot(Ep.conj(), Z, axes=([1], [0]))  # (16,4,4)
                eh = Ep @ E  # (16,4)
                g.append(np.einsum("Aab,Bb->AaBb", zh, eh))
            tot = sum(g)
            vals.append(((np.abs(tot) ** 2).sum() - sum((np.abs(x) ** 2).sum() for x in g)) / (k * (k - 1)))
        norm = (
            4.0 * np.sum(np.sin(self.p0) ** 2) / (64.0 if key == "" else 32.0)
        )  # per doublet: 1 for a free massless doublet
        return {f"G{key}_p0": float(np.mean(vals) * norm)}

    def taste_channels(self, G=None, samples=None):
        """Light/heavy-doublet observables (exact from the dense G, or stochastic from the noise samples):
        heavy pattern C_χ (= the model's pattern, keys phi_f… above) plus its flavour-summed susceptibility
        chi_f_sum(_conn/_corners/_pmin);
        light pattern C_L = P_L C_asd P_L: phi_L (4,), chi_L (4,), chi_L_conn, chi_L_corners (4,16), chi_L_pmin, and
        the flavour-summed
            chi_L_sum (= V⟨φ_L²⟩, the light-doublet SSB susceptibility), chi_L_sum_conn/_corners/_pmin;
        light (6,1,1) channel: n^L_a(x) = ¼ tr[Γ_a G_L(x,x)], the σ-triplet of the on-site 6 restricted to the light
        doublet
            (the other triplet Γ′ of GAMMA2 vanishes identically for every propagator of the τ₂K class, which lives in
            span{1, Γ_a}): phi6L (3,), phi6L_stag (3,), phi6L_sq, phi6L_stag_sq (U-statistics), O4L = (1/V)Σ|n^L|²;
        momentum readout GL_p0, GR_p0 (G_p0): 4Σ sin²p₀ ‖G_{L,R}(p₀)‖²_F/32 (‖G(p₀)‖²/64 for the full propagator) on the
            16-corner × 4-flavour block at the smallest twisted momentum p₀, averaged over its 2⁴ sign images: exactly
            4Σsin²p₀/(4Σsin²p₀ + m(p₀)²) for a free doublet of mass m(p₀) (1 when massless; m_R(p₀) = ½h(c_D + c_dual),
            m_L(p₀) = ½h(c_D − c_dual)); → 0 for a pole (× p²/(p²+m²)) or a Luttinger zero; stochastic by U-statistics;
        exact only: light/heavy timeslice correlators CL_t, GL_t, CR_t, GR_t (all sources); the light-doublet composite
            phi_T_L = (1/V)½ΣC_χ⟨Ψ_L Ψ_L⟩ with Ψ_L^a = Π_{b≠a}(P_Lχ)^b (all four a) and its correlator C_T_L(t), and
            the full
            composite phi_T_R, C_T_R (unprojected Ψ, pattern C_χ)."""
        lat = self.lat
        V = lat.V
        out = {}
        if G is not None:
            GL = self._project_G(G, "L")
            out.update(self.pair_channel(G=G, C=self.CR, prefix="f", summed=True))
            out.update(self.pair_channel(G=G, C=self.CL, prefix="L", summed=True))
            Gxx = GL[np.arange(V), :, np.arange(V), :]
            ns = [0.25 * np.einsum("cab,uab->uc", GAMMA, Gxx)]
        else:
            out.update(self.pair_channel(samples=samples, C=self.CR, prefix="f", summed=True))
            out.update(self.pair_channel(samples=samples, C=self.CL, prefix="L", summed=True))
            ns = []
            blocks = {"L": [], "R": [], "": []}
            for Z, E in samples:
                Zf = Z.reshape(V, 16)
                PZ = self.tp.PL(Zf).reshape(V, 4, 4)
                PE = self.tp.PL(E)
                ns.append(0.25 * np.einsum("cab,uab,ub->uc", GAMMA, PZ, PE))
                blocks["L"].append((PZ, PE))
                blocks["R"].append((self.tp.PR(Zf).reshape(V, 4, 4), self.tp.PR(E)))
                blocks[""].append((Z, E))
            for key, bl in blocks.items():
                out.update(self._p0_blocks_stochastic(bl, key))
        ns = np.stack(ns)  # (k, V, 3)
        k = ns.shape[0]
        nbar = ns.mean(0)
        phi = nbar.mean(0)
        phi_stag = (nbar * self.eps[:, None]).mean(0)
        out.update(phi6L=phi, phi6L_stag=phi_stag, O4L_biased=float((nbar**2).sum(1).mean()))
        if k >= 2:
            pk = ns.mean(1)
            psk = (ns * self.eps[None, :, None]).mean(1)
            out["phi6L_sq"] = float(_ustat_pairs(pk))
            out["phi6L_stag_sq"] = float(_ustat_pairs(psk))
            tot = ns.sum(0)
            out["O4L"] = float((((tot**2).sum(1) - (ns**2).sum((0, 2))) / (k * (k - 1))).mean())
        else:
            out["phi6L_sq"] = float((phi**2).sum())
            out["phi6L_stag_sq"] = float((phi_stag**2).sum())
            out["O4L"] = out["O4L_biased"]
        if G is not None:
            GR = self._project_G(G, "R")
            out.update(self._timeslice(GL, "L"))
            out.update(self._timeslice(GR, "R"))
            out.update(self._p0_blocks(GL, "L"))
            out.update(self._p0_blocks(GR, "R"))
            out.update(self._p0_blocks(G, ""))
            if self.six_fermion_taste:
                sets = [[b for b in range(4) if b != a] for a in range(4)]
                out.update(self.six_fermion_channel(GL, sets, C=self.CR, keys=("phi_T_L", "C_T_L")))
                out.update(self.six_fermion_channel(G, sets, C=self.CR, keys=("phi_T_R", "C_T_R")))
        return out

    def taste_correlator(self, fields, Y=None, x0=None):
        """Point-source light/heavy correlators (stochastic path; the exact path has them from `taste_channels`):
        Cf_L, Gf_L, Cf_R, Gf_R (4, Lt) from the columns (P G P) e_{x0,a} = P G (P e_{x0,a}), 8 solves."""
        lat, m = self.lat, self.model
        L = lat.shape
        Lt = L[-1]
        V = lat.V
        eta4 = to_numpy(lat.eta[-1]).reshape(V)
        t_of = self.coords[-1]
        if Y is None:
            Y = m.D.prepare(fields)
        if x0 is None:
            x0 = tuple(int(self.rng.integers(l)) for l in L)
        i0 = int(np.ravel_multi_index(x0, L))
        xp = lat.xp
        out = {}
        for which, P in (("L", self.tp.PL), ("R", self.tp.PR)):
            Gf = np.zeros((4, Lt))
            Cf = np.zeros((4, Lt))
            dt = (t_of - t_of[i0]) % Lt
            wrap = np.where(t_of < t_of[i0], float(lat.bc[-1]), 1.0)
            for a in range(4):
                src = np.zeros((V, 4))
                src[i0, a] = 1.0
                src = P(src)  # P e_{x0,a} (real, (V,4))
                z = unpack(m.solve_Dinv(fields, xp.asarray(pack(src)), tol=self.cg_tol, Y=Y))  # (V,4): G (P e)
                z = P(z)  # P G P e
                np.add.at(Gf[a], dt, wrap * eta4[i0] * z[:, a])
                np.add.at(Cf[a], dt, (z**2).sum(1))
            out[f"Gf_{which}"] = Gf
            out[f"Cf_{which}"] = Cf
        out["x0_taste"] = np.array(x0)
        return out
