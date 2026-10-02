"""Fermion observables of the doublet model with link fields: epsilon channel, bonds and dimers, and the complete
SU(4)-invariant 10- and 6-channel pair correlators.

Real flavours from the doublet: G^{ab}(x, z) = Re[u_a conj(u_b) S_{p(a) p(b)}(x, z)], S = D_c^-1. Wick's theorem for
four
real Grassmann fields gives, for the pair operators of a plaquette quad (G_1 = G(y, x), G_2 = G(y', x'), G_3 = G(y, x'),
G_4 = G(y', x), K = tr G):

    sum_ab <Phibar^{ab}(y, y') Phi^{ab}(x, x')>       = disc_10 - 2 K1 K2 - 2 tr(G1 G2) + 2 tr(G3 G4) + 2 K3 K4,
    sum_ab <Lambdabar^{ab}(y, y') Lambda^{ab}(x, x')> = disc_6  - 2 K1 K2 + 2 tr(G1 G2) + 2 tr(G3 G4) - 2 K3 K4,

reported as structure factors S(p) = (1/V) sum_{x even, y odd} cos(p.(x - y)) w(y) w(x) <pair(y, y') pair(x, x')>,
class-averaged over the plane orientations, at the 16 corner momenta (chi10_corners, chi6_corners; chi10, chi6 = the
p = 0 entries) and at corner + p_min e_rho averaged over rho (chi10_pmin, chi6_pmin); the local energies E10 = <T_Q>,
E6 = <T_6> per quad; and the source-direction one-point function phi_src = <O_h>/V (even/odd parts) with its
Wick-connected h-derivative phi_src_dh (O_h of the operator's pattern). w is the fermionic boundary sign of the pair.
Every term is a product of exactly two propagators, so the U-statistic over ordered pairs of distinct noise samples
is unbiased.

Also: epsilon-channel bilinears (phi, phi_stag, O4 as in ``epsilon``), the oriented link bilinears Kt_l (bond energies
E_bond per direction, dimer structure factors D_{mu nu} = (1/n_mu) sum (-1)^{x_nu} Kt_{(x, x + mu)} and V<D^2> by
U-statistics), and the inherited point-source correlator.
"""

from __future__ import annotations

import numpy as np

from ..flavour import real_flavour_block
from ..lattice import shift_table, to_numpy
from ..patterns import source_matrix
from .common import epsilon_moments, n_from_blocks, ustat_pairs
from .epsilon import FermionMeasure


class ChannelMeasure(FermionMeasure):
    """Dense (``exact``) or stochastic (Z2 noise, 2 solves per sample) estimators; ``bc_signs=False`` drops the
    boundary sign of the pair operators (for comparisons only)."""

    def __init__(self, model, n_noise=4, exact=False, seed=0, cg_tol=1e-9, bc_signs=True):
        super().__init__(model, n_noise=n_noise, exact=exact, seed=seed, cg_tol=cg_tol)
        self.bc_signs = bool(bc_signs)
        self._full = None
        lat = self.lat
        if model.geom is not None:
            self.geom = model.geom
        else:  # link tables only (bond energies and dimers of models without link fields)
            from ..wedge import WedgeGeometry

            self.geom = WedgeGeometry(lat, "all", 0.0, 0.0)
        d, V = lat.d, lat.V
        self.eps = to_numpy(lat.eps).reshape(V)
        self.even = np.where(self.eps > 0)[0]
        self.odd = np.where(self.eps < 0)[0]
        self.coords = lat.coords.reshape(d, V)
        self.planes = []
        for mu in range(d):
            for nu in range(mu + 1, d):
                for sg in (+1, -1):
                    de = np.zeros(d, int)
                    de[mu] = 1
                    de[nu] = sg
                    do = np.zeros(d, int)
                    do[mu] = -1
                    do[nu] = sg
                    self.planes.append((mu, nu, sg, tuple(de), tuple(do)))
        self.n_orient = len(self.planes)

    def _solve_samples(self, fields, Y):
        """Stochastic samples [(Z (V, 2, 2) with Z[x, i, b] = (D^-1 eta_b)_i(x), E (V, 2) with E[x, b] = eta_b(x))]."""
        lat, m = self.lat, self.model
        xp = lat.xp
        V = lat.V
        out = []
        for _k in range(self.n_noise):
            Z = np.zeros((V, 2, 2), complex)
            E = np.zeros((V, 2))
            for b in range(2):
                eta = xp.zeros((V, 2), complex)
                e = self.rng.choice([-1.0, 1.0], size=V)
                eta[:, b] = xp.asarray(e)
                Z[:, :, b] = to_numpy(m.solve_Dinv(fields, eta, tol=self.cg_tol, Y=Y))
                E[:, b] = e
            out.append((Z, E))
        return out

    def bilinears(self, fields, Y=None) -> dict:
        lat, g = self.lat, self.geom
        d, V = lat.d, lat.V
        if Y is None:
            Y = self.model.D.prepare(fields)
        if self.exact:
            S = self.model.D.dense_S(fields)
            samples = [None]
        else:
            S = None
            samples = self._solve_samples(fields, Y)
        k = len(samples)

        def S_at(sample, src, dst):
            if self.exact:
                return S[src, :, dst, :]
            Z, E = sample
            return Z[src] * E[dst][:, None, :]

        idx = np.arange(V)
        ns = np.stack([n_from_blocks(S_at(smp, idx, idx)) for smp in samples])
        out = epsilon_moments(ns, self.eps)
        if True:
            Kl = np.stack([g.lsign * 2.0 * np.einsum("nii->n", S_at(smp, g.lu, g.lv)).real for smp in samples])
            Kbar = Kl.mean(0)
            out["E_bond"] = np.array([Kbar[g.ldir == mu].mean() for mu in range(d)])
            Dk = np.zeros((k, d, d))
            for mu in range(d):
                sel = g.ldir == mu
                for nu in range(d):
                    sgn = (-1.0) ** self.coords[nu, g.lpos[sel]]
                    Dk[:, mu, nu] = (Kl[:, sel] * sgn).mean(1)
            out["dimer"] = Dk.mean(0)
            if k >= 2:
                D2 = np.array([[ustat_pairs(Dk[:, mu, nu]) for nu in range(d)] for mu in range(d)])
            else:
                D2 = Dk[0] ** 2
            out["dimer_sq"] = D2
            out["dimer_sq_par"] = float(np.trace(D2) / d)
            out["dimer_sq_perp"] = float((D2.sum() - np.trace(D2)) / (d * (d - 1)))
        out.update(self.full_channels(S if self.exact else None, None if self.exact else samples))
        return out

    def _full_tables(self):
        if self._full is None:
            lat = self.lat
            d, V = lat.d, lat.V
            corners = [np.array(c, float) * np.pi for c in np.ndindex(*(2,) * d)]
            mom = list(corners)
            for c in corners:
                for rho in range(d):
                    q = c.copy()
                    q[rho] += 2.0 * np.pi / lat.shape[rho]
                    mom.append(q)
            mom = np.asarray(mom)  # (nc + nc·d, d)
            ph = np.exp(1j * (mom @ self.coords))  # (n_mom, V): e^{i p·x}
            planes = []
            for mu, _nu, _sg, de, do in self.planes:
                je, we = shift_table(lat, de)
                jo, wo = shift_table(lat, do)
                em = np.zeros(d, int)
                em[mu] = 1
                jm, _ = shift_table(lat, tuple(em))
                if not self.bc_signs:
                    we = np.ones(V)
                    wo = np.ones(V)
                planes.append(
                    dict(
                        xd=je[self.even],
                        we=we[self.even],
                        yd=jo[self.odd],
                        wo=wo[self.odd],
                        yl=jm[self.even],
                        yld=jo[jm[self.even]],
                        wl=we[self.even] * wo[jm[self.even]],
                    )
                )
            C0 = source_matrix(lat, self.model.D.pattern)
            self._full = dict(
                nc=len(corners),
                mom=mom,
                Pe=ph[:, self.even],
                Po=ph[:, self.odd],
                planes=planes,
                C0=C0,
                C0p=C0.multiply(C0 > 0).tocsr(),
            )
        return self._full

    def _reduce(self, sf, loc, src):
        """Structure-factor arrays over momenta → the stored keys."""
        T = self._full_tables()
        nc, d = T["nc"], self.lat.d
        out = {}
        for ch, nm in (("phi", "chi10"), ("lam", "chi6")):
            out[nm] = float(sf[ch][0])
            out[nm + "_corners"] = sf[ch][:nc].copy()
            out[nm + "_pmin"] = sf[ch][nc:].reshape(nc, d).mean(1)
        out["E10"], out["E6"] = float(loc["phi"]), float(loc["lam"])
        out.update(src)
        return out

    def full_channels(self, S=None, samples=None):
        """Complete correlators from the dense doublet inverse S (V,2,V,2) or from the noise samples [(Z, E)] (k ≥
        2)."""
        T = self._full_tables()
        V = self.lat.V
        xe, ye = self.even, self.odd
        Pe, Po = T["Pe"], T["Po"]
        nmom = Pe.shape[0]
        sf = {"phi": np.zeros(nmom), "lam": np.zeros(nmom)}
        loc = {"phi": 0.0, "lam": 0.0}
        no = len(T["planes"])
        tr2 = lambda A: A[:, 0, :, 0] + A[:, 1, :, 1]
        if S is not None:
            ft = lambda C: np.einsum("ky,yx,kx->k", Po, C, Pe.conj(), optimize=True).real / V
            Sye = S[ye]
            for pl in T["planes"]:
                xd, yd, we, wo = pl["xd"], pl["yd"], pl["we"], pl["wo"]
                Syd = S[yd]
                W = wo[:, None] * we[None, :]
                Gyy = real_flavour_block(S[ye, :, yd, :])
                Gxx = real_flavour_block(S[xe, :, xd, :])
                sym = lambda G, s: (G + s * G.transpose(0, 2, 1)).reshape(len(G), 16)
                P10 = sym(Gyy, 1) @ sym(Gxx, 1).T
                P6 = sym(Gyy, -1) @ sym(Gxx, -1).T
                A1 = Sye[:, :, xe, :]
                A2 = Syd[:, :, xd, :]
                A3 = Sye[:, :, xd, :]
                A4 = Syd[:, :, xe, :]
                KK1 = 4.0 * tr2(A1).real * tr2(A2).real
                KK2 = 4.0 * tr2(A3).real * tr2(A4).real
                tG1 = 2.0 * np.einsum("ypxq,yqxp->yx", A1, A2).real
                tG2 = 2.0 * np.einsum("ypxq,yqxp->yx", A3, A4).real
                sf["phi"] += ft(W * (P10 - 2 * KK1 - 2 * tG1 + 2 * tG2 + 2 * KK2))
                sf["lam"] += ft(W * (P6 - 2 * KK1 + 2 * tG1 + 2 * tG2 - 2 * KK2))
                yl, yld, wl = pl["yl"], pl["yld"], pl["wl"]
                B = lambda u, v: S[u, :, v, :]
                Gyy = real_flavour_block(B(yl, yld))
                Gxx = real_flavour_block(B(xe, xd))
                d10 = (sym(Gyy, 1) * sym(Gxx, 1)).sum(1)
                d6 = (sym(Gyy, -1) * sym(Gxx, -1)).sum(1)
                t1 = lambda A: (A[:, 0, 0] + A[:, 1, 1]).real
                kk1 = 4.0 * t1(B(yl, xe)) * t1(B(yld, xd))
                kk2 = 4.0 * t1(B(yl, xd)) * t1(B(yld, xe))
                g1 = 2.0 * np.einsum("npq,nqp->n", B(yl, xe), B(yld, xd)).real
                g2 = 2.0 * np.einsum("npq,nqp->n", B(yl, xd), B(yld, xe)).real
                loc["phi"] += (wl * (d10 - 2 * kk1 - 2 * g1 + 2 * g2 + 2 * kk2)).mean()
                loc["lam"] += (wl * (d6 - 2 * kk1 + 2 * g1 + 2 * g2 - 2 * kk2)).mean()
            Sm = S.reshape(2 * V, 2 * V)
            trS = (S[:, 0, :, 0] + S[:, 1, :, 1]).real  # (V, V): Re tr S(x, z)
            C0, C0p = T["C0"], T["C0p"]
            cp = C0p.tocoo()
            per_site = np.zeros(V)
            np.add.at(per_site, cp.row, 2.0 * cp.data * trS[cp.row, cp.col])
            C2 = np.kron(C0.toarray(), np.eye(2))
            dS = (Sm @ C2 @ Sm).reshape(V, 2, V, 2)
            dtr = (dS[:, 0, :, 0] + dS[:, 1, :, 1]).real
            dh = 2.0 * (cp.data * dtr[cp.row, cp.col]).sum() / V
            src = dict(
                phi_src=float(per_site.sum() / V),
                phi_src_even=float(per_site[xe].sum() / V),
                phi_src_odd=float(per_site[ye].sum() / V),
                phi_src_dh=float(dh),
            )
        else:
            k = len(samples)
            assert k >= 2, "the complete estimator needs n_noise >= 2"
            Zs = [z for z, e in samples]
            Es = [e for z, e in samples]
            al = [2.0 * np.stack([z[:, 0, 0].real, z[:, 1, 1].real], 1) for z in Zs]  # α[x, b] = 2 Re Z[x, b, b]
            npair = k * (k - 1)

            def cosum(F, G):
                """Σ_{x,y} cos(p·(x−y)) F_c(y) G_c(x), summed over the component axis: F (m, n_o), G (m, n_e) real."""
                Fp = Po @ F.T
                Fm = Po.conj() @ F.T
                Gp = Pe @ G.T  # (n_mom, m)
                return 0.5 * (Fm * Gp + Fp * Gp.conj()).sum(1)

            for pl in T["planes"]:
                xd, yd, we, wo = pl["xd"], pl["yd"], pl["we"], pl["wo"]
                yl, yld, wl = pl["yl"], pl["yld"], pl["wl"]
                # disconnected parts: one-point functions per sample, U-statistic in momentum space
                Pb = np.zeros((k, 2, nmom, 16), complex)
                Pf = np.zeros((k, 2, nmom, 16), complex)
                Lb = np.zeros((k, 2, len(xe), 16))
                Lf = np.zeros((k, 2, len(xe), 16))
                for i in range(k):
                    Gy = real_flavour_block(Zs[i][ye] * Es[i][yd][:, None, :])
                    Gx = real_flavour_block(Zs[i][xe] * Es[i][xd][:, None, :])
                    Gl = real_flavour_block(Zs[i][yl] * Es[i][yld][:, None, :])
                    for c, sg_ in enumerate((1.0, -1.0)):
                        Pb[i, c] = Po @ ((Gy + sg_ * Gy.transpose(0, 2, 1)).reshape(-1, 16) * wo[:, None])
                        Pf[i, c] = Pe @ ((Gx + sg_ * Gx.transpose(0, 2, 1)).reshape(-1, 16) * we[:, None])
                        Lb[i, c] = (Gl + sg_ * Gl.transpose(0, 2, 1)).reshape(-1, 16)
                        Lf[i, c] = (Gx + sg_ * Gx.transpose(0, 2, 1)).reshape(-1, 16)
                disc = (
                    (Pb.sum(0) * Pf.sum(0).conj()).sum(-1) - (Pb * Pf.conj()).sum((0, -1))
                ).real / npair  # (2, n_mom)
                dloc = (((Lb.sum(0) * Lf.sum(0)).sum(-1) - (Lb * Lf).sum((0, -1))) * wl).mean(-1) / npair  # (2,)
                kk1 = kk2 = g1 = g2 = 0.0
                l_kk1 = l_kk2 = l_g1 = l_g2 = 0.0
                for i in range(k):
                    Zi, Ei, ai = Zs[i], Es[i], al[i]
                    for j in range(k):
                        if i == j:
                            continue
                        Zj, Ej, aj = Zs[j], Es[j], al[j]
                        # K₁K₂ / K₃K₄: K(y,x) ≈ Σ_b α_i[y,b] E_i[x,b],  K(y′,x′) ≈ Σ_c α_j[y′,c] E_j[x′,c]
                        FA = (ai[ye][:, :, None] * aj[yd][:, None, :] * wo[:, None, None]).reshape(-1, 4).T
                        G1 = (Ei[xe][:, :, None] * Ej[xd][:, None, :] * we[:, None, None]).reshape(-1, 4).T
                        G2 = (Ei[xd][:, :, None] * Ej[xe][:, None, :] * we[:, None, None]).reshape(-1, 4).T
                        kk1 = kk1 + cosum(FA, G1).real
                        kk2 = kk2 + cosum(FA, G2).real
                        # tr(G₁G₂) = 2 Re Σ_pq S_pq(y,x) S_qp(y′,x′) ≈ Z_i[y,p,q] E_i[x,q] · Z_j[y′,q,p]
                        # E_j[x′,p]
                        FB = (
                            (Zi[ye] * Zj[yd].transpose(0, 2, 1) * wo[:, None, None]).reshape(-1, 4).T
                        )  # component (p, q)
                        H1 = (
                            (Ei[xe][:, None, :] * Ej[xd][:, :, None] * we[:, None, None]).reshape(-1, 4).T
                        )  # E_i[x,q] E_j[x′,p]
                        H2 = (Ei[xd][:, None, :] * Ej[xe][:, :, None] * we[:, None, None]).reshape(-1, 4).T
                        g1 = g1 + 2.0 * cosum(FB, H1).real
                        g2 = g2 + 2.0 * cosum(FB, H2).real
                        # local (y = x + μ̂)
                        l_kk1 += ((ai[yl] * Ei[xe]).sum(1) * (aj[yld] * Ej[xd]).sum(1) * wl).mean()
                        l_kk2 += ((ai[yl] * Ei[xd]).sum(1) * (aj[yld] * Ej[xe]).sum(1) * wl).mean()
                        zz = Zi[yl] * Zj[yld].transpose(0, 2, 1)
                        l_g1 += 2.0 * ((zz * Ei[xe][:, None, :] * Ej[xd][:, :, None]).sum((1, 2)).real * wl).mean()
                        l_g2 += 2.0 * ((zz * Ei[xd][:, None, :] * Ej[xe][:, :, None]).sum((1, 2)).real * wl).mean()
                kk1, kk2, g1, g2 = (v / npair / V for v in (kk1, kk2, g1, g2))
                l_kk1, l_kk2, l_g1, l_g2 = (v / npair for v in (l_kk1, l_kk2, l_g1, l_g2))
                sf["phi"] += disc[0] / V - 2 * kk1 - 2 * g1 + 2 * g2 + 2 * kk2
                sf["lam"] += disc[1] / V - 2 * kk1 + 2 * g1 + 2 * g2 - 2 * kk2
                loc["phi"] += dloc[0] - 2 * l_kk1 - 2 * l_g1 + 2 * l_g2 + 2 * l_kk2
                loc["lam"] += dloc[1] - 2 * l_kk1 + 2 * l_g1 + 2 * l_g2 - 2 * l_kk2
            # source direction: ⟨O_h⟩ = Σ_{C₀[x,z] > 0} C₀[x,z] · 2 Re tr S(x,z);  ∂/∂h: S → S C₀ S
            C0, C0p = T["C0"], T["C0p"]
            per_site = np.zeros(V)
            for i in range(k):
                per_site += (al[i] * (C0p @ Es[i])).sum(1) / k
            dh = 0.0
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    # Σ_{xz} C₀⁺[x,z] S_pq(x,w) C₀[w,w′] S_qp(w′,z) ≈ [Σ_x Z_i[x,p,q] (C₀⁺E_j)[x,p]] [Σ_w
                    # E_i[w,q] (C₀ Z_j)[w,q,p]]
                    A = np.einsum("xpq,xp->pq", Zs[i], C0p @ Es[j])
                    Bm = np.einsum("wq,wqp->qp", Es[i], (C0 @ Zs[j].reshape(V, 4)).reshape(V, 2, 2))
                    dh += 2.0 * (A * Bm.T).sum().real
            dh /= npair * V
            src = dict(
                phi_src=float(per_site.sum() / V),
                phi_src_even=float(per_site[xe].sum() / V),
                phi_src_odd=float(per_site[ye].sum() / V),
                phi_src_dh=float(dh),
            )
        for ch in sf:
            sf[ch] = sf[ch] / no
        for ch in loc:
            loc[ch] = loc[ch] / no
        return self._reduce(sf, loc, src)
