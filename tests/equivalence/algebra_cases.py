"""Algebra-group computations, written twice: against the notebook modules (``unhiggsed.spin10``, ``liealg``,
``grassmann``, ``eta``, ``staggered`` and the claim-check code of the notebook) and against ``masspairing.algebra``.
Each side returns a dict name -> array; ``make_algebra_reference.py`` freezes the notebook side.
"""

from __future__ import annotations

import itertools

import numpy as np

SEED = 20261002


def _poly_arrays(p):
    """A sparse Grassmann polynomial {mask: coeff} as (sorted masks as float, coefficients)."""
    keys = sorted(p)
    return np.array(keys, dtype=float), np.array([p[k] for k in keys], dtype=complex)


def _dec_array(dec):
    """A decomposition {highest weight: multiplicity} as a sorted integer array of rows (weight..., multiplicity)."""
    return np.array(sorted(tuple(w) + (m,) for w, m in dec.items()), dtype=int)


# ---------------------------------------------------------------------------------------------------------------
# shared drivers: take a namespace of functions with the notebook or the clean names
# ---------------------------------------------------------------------------------------------------------------
def _spin10(sp, out):
    G = sp.gammas()
    out["gammas"] = np.stack(G)
    out["C"] = sp.charge_conjugation(G)
    out["P"] = sp.weyl_basis(G)
    out["B1"] = np.stack(sp.bilinear_blocks(1))
    out["B3"] = np.stack(sp.bilinear_blocks(3))
    out["B5"] = np.stack(sp.bilinear_blocks(5))
    out["su5_126"] = sp.su5_singlet_126()
    W, wts = sp.weyl_weight_basis(G)
    out["W"], out["wts"] = W, wts
    rv = sp.root_vectors(G)
    out["roots"] = np.array(sorted(rv), dtype=int)
    out["root_vectors"] = np.stack([rv[k] for k in sorted(rv)])
    out["so10"] = np.stack(sp.so10_generators(G))
    out["cartan"] = np.stack(sp.cartan_generators(G))


def _characters(la, out):
    s16 = la.spinor16(-1)
    out["dec_S22"] = _dec_array(la.decompose(la.ms_schur(s16, (2, 2))))
    out["inv_S222"] = np.array([la.invariants(la.ms_schur(s16, (2, 2, 2)))])
    c126 = la.char126()
    out["dec_126x126"] = _dec_array(la.decompose(la.ms_tensor(c126, c126)))
    out["dec_sym2_126"] = _dec_array(la.decompose(la.ms_sym_powers(c126, 2)[2]))
    out["dec_ext2_126"] = _dec_array(la.decompose(la.ms_ext_powers(c126, 2)[2]))
    out["inv_sym4_126"] = np.array([la.invariants(la.ms_sym_powers(c126, 4)[4])])
    for name, ch in (
        ("16", s16),
        ("10", la.vector10()),
        ("126", c126),
        ("120", la.ms_ext_powers(s16, 2)[2]),
        ("45", la.ms_ext_powers(la.vector10(), 2)[2]),
    ):
        out[f"ps_{name}"] = _dec_array(la.decompose(la.branch_d5_to_ps(ch), 2, 3))
    V = la.ms_tensor({w + (0, 0): 1 for w in s16}, {(0,) * 5 + (1, 0): 1, (0,) * 5 + (-1, 0): 1})
    Vb = la.ms_tensor({tuple(-x for x in w) + (0, 0): 1 for w in s16}, {(0,) * 5 + (0, 1): 1, (0,) * 5 + (0, -1): 1})
    out["dec_L2V"] = _dec_array(la.decompose(la.ms_ext_powers(V, 2)[2], k=2))
    out["mult_partner"] = np.array(
        [
            la.multiplicity(la.ms_tensor(la.ms_tensor(Vb, Vb), V), max(la.spinor16(+1)) + (1, 0), k=2),
            la.multiplicity(la.ms_tensor(la.ms_tensor(V, V), V), max(la.spinor16(+1)) + (1, 0), k=2),
        ]
    )
    out["weyl_dims"] = np.array([la.weyl_dim(lam) for lam in ((2, 0, 0, 0, 0), (2, 2, 2, 2, -2), (4, 0, 0, 0, 0))])


def _eta(et, out):
    vals = []
    for m in (2, 4, 8):
        for n in range(1, 4 * m + 1):
            for h in et.spin_z2m_structures(n, m):
                for q in (1, 3):
                    vals.append((n, m, h, q, et.xi_spin_z2m(n, m, h, q)))
    out["xi_z2m"] = np.array(vals)
    out["xi_zn"] = np.array([et.xi_spin_zn(n, s) for n in (3, 5, 7) for s in (1, 2)])
    spec = [et.eta_spectral(n, l) for n, l in ((2, 1), (4, 1), (8, 3))]
    out["eta_spectral"] = np.array([s[0] for s in spec])
    out["eta_resid"] = np.array([s[1] for s in spec])


def _grassmann(gr, sp, out):
    G = sp.gammas()
    C = sp.charge_conjugation(G)
    W, wts = sp.weyl_weight_basis(G)
    Phi10 = [gr.bilinear(W.T @ C @ g @ W) for g in G]
    sq = {}
    for p in Phi10:
        sq = gr.add(sq, gr.mul(p, p))
    out["phi10sq_masks"], out["phi10sq_coef"] = _poly_arrays(gr.clean(sq))
    m5 = W.T @ C @ np.linalg.multi_dot([G[i] for i in (0, 2, 4, 6, 8)]) @ W
    out["bilin_bar_masks"], out["bilin_bar_coef"] = _poly_arrays(gr.bilinear(m5.conj(), offset=32))
    wt32 = np.array(
        [tuple(int(round(2 * x)) for x in wts[a]) + (1 if al == 0 else -1,) for a in range(16) for al in (0, 1)]
    )
    monos = [S for S in itertools.combinations(range(32), 4) if not np.any(wt32[list(S)].sum(0))]
    E = sp.root_vectors(G)[(1, -1, 0, 0, 0)]
    D, rows = gr.derivation_matrix(np.kron(W.conj().T @ E @ W, np.eye(2)), monos)
    out["derivation_matrix"] = D.toarray()
    out["derivation_rows"] = np.array(sorted(rows.items()), dtype=float)
    X = np.kron(W.conj().T @ sp.so10_generators(G)[7] @ W, np.eye(2))
    out["derivation_masks"], out["derivation_coef"] = _poly_arrays(gr.derivation(gr.clean(sq), X))
    out["cc_masks"], out["cc_coef"] = _poly_arrays(gr.current_current([np.eye(16) / 4]))
    out["rank_of"] = np.array([gr.rank_of(Phi10[:4])[0]])


def _lattice_grass(st, pf, field, out):
    """``field`` gives (chi, K, eps_term, sym2_term, lam2_term, set_up(n)) for either side."""
    for L, d in ((4, 2), (2, 3), (4, 1)):
        M, S, idx = st.staggered_matrix(L, d)
        out[f"M_{L}_{d}"] = M
        out[f"quads_{L}_{d}"] = np.array(st.plaquette_quads(L, d, S, idx))
        ls = st.oriented_links(M, S, idx, L, d)
        out[f"links_{L}_{d}"] = np.array([k + (v,) for k, v in sorted(ls.items())], dtype=float)
    M, S, idx = st.staggered_matrix(4, 2)
    out["line_quads_4_2"] = np.array(st.line_quads(4, 2, S, idx))
    out["M_4_2_periodic"] = st.staggered_matrix(4, 2, bc="periodic")[0]
    rng = np.random.default_rng(SEED + 1)
    A = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    out["pfaffian"] = np.array([pf(A - A.T), pf((A - A.T).real)])
    f = field(4)
    T10 = f["sym2"]((0, 1, 2, 3))
    out["T10_masks"], out["T10_coef"] = _poly_arrays(T10.t)
    E = f["eps"](0)
    out["exp_masks"], out["exp_coef"] = _poly_arrays(T10.exp_nilpotent(8).t)
    U = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    V = f["map_su4"](U, [1, 1, -1, -1])
    img = f["substitute"](E + T10 * 0.5, V, 16)
    out["subst_masks"], out["subst_coef"] = _poly_arrays(img.t)
    M, S, idx = st.staggered_matrix(2, 3)
    f = field(M.shape[0])
    quads = st.plaquette_quads(2, 3, S, idx)
    res = f["bag_weights"](M, quads, f["sym2"], 8, False)
    out["bags_2_3_cfg"] = np.array([cfg for cfg, _ in res])
    out["bags_2_3_w"] = np.array([w for _, w in res])
    res6 = f["bag_weights"](M, quads, f["lam2"], 3, True)
    out["bags6_2_3_w"] = np.array([w for _, w in res6])
    M, S, idx = st.staggered_matrix(4, 2)
    quads = st.plaquette_quads(4, 2, S, idx)
    phi = []
    for _ in quads:
        a, b = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        phi.append((a + a.T) / 2 + 1j * (b + b.T) / 2)
    out["yukawa"] = st.sym2_yukawa_matrix(M, quads, phi)
    ls = st.oriented_links(M, S, idx, 4, 2)
    keys = sorted(ls)
    G = np.linalg.inv(M)
    lk = [keys[0], keys[5], keys[11]]
    pfs = f["link_pfaffians"](G, lk, [ls[k] for k in lk])
    out["link_pf"] = np.array([pfs[m] for m in sorted(pfs)])
    out["kmono_table"] = f["kmonomial_table"](pfs, 3, 4)


def _hamiltonian(ha, out):
    E = ha.hs_ensembles(2, [(0, 1)])
    for name, ens in E.items():
        out[f"hs_{name}"] = np.stack([A for A, _ in ens])
        out[f"hs_kind_{name}"] = np.array([kind == "herm" for _, kind in ens])
    rng = np.random.default_rng(21)
    dims = []
    for name in ("eps_pos", "pairing", "dens_exch", "spin"):
        r = ha.mtr_classify(E["band"] + E[name], rng)
        dims.append((r["dimL_sym"], r["dimL_anti"], r["dimC"], r["Tplus"] is not None, r["Tminus"] is not None))
        if r["cls"] is not None:
            out[f"mtr_U1_{name}"], out[f"mtr_Q_{name}"] = r["U1"], r["Q"]
    out["mtr_dims"] = np.array(dims, dtype=int)
    ops = ha.cluster_ops(2, [(0, 1)])
    out["H10"] = ha.h10_operator(ops)
    out["Heps"] = ops["Heps"]
    out["Hband"] = ops["Hband"]


def _bags148(Lat, kin, cm, sp_, bags, out):
    lat = Lat((2, 2, 2, 4))
    V = lat.V
    K, C, C0 = kin(lat), cm(lat), sp_(lat)
    h = 0.5
    sets = [()] + [B for k in (2,) for B in itertools.combinations(range(V), k)] + [(0, 3, 9, 20), (1, 2, 5, 30)]
    rng = np.random.default_rng(148)
    big = [tuple(sorted(rng.choice(V, size=k, replace=False))) for k in (6, 12, 24, 32)]
    Ms = [K, K, K, K - h * C]
    Gs, W = bags["weights"](Ms, sets + big)
    out["bag148_w"] = np.array([W[B][0] for B in sets + big])
    out["bag148_full"] = np.array([bags["wick_full"](Gs, B) for B in sets[1:] + big])
    for name, Cp in (("C", C), ("C0", C0)):
        Nl, Zk = bags["order"](Ms, [Cp, Cp, Cp, None], sets + big)
        Nh, _ = bags["order"](Ms, [None, None, None, Cp], sets + big)
        ks = sorted(Zk)
        out[f"order148_{name}"] = np.array([[k, Nl[k], Nh[k], Zk[k]] for k in ks])
    Mb = [K - h * C] * 4
    Nt, Zt = bags["order"](Mb, [cm(lat, -1)] * 4, sets)
    out["order148_blind"] = np.array([[k, Nt[k], Zt[k]] for k in sorted(Zt)])


# ---------------------------------------------------------------------------------------------------------------
# notebook side
# ---------------------------------------------------------------------------------------------------------------
def notebook_side():
    from unhiggsed import eta as net
    from unhiggsed import grassmann as ngr
    from unhiggsed import liealg as nla
    from unhiggsed import spin10 as nsp
    from unhiggsed import staggered as nst
    from unhiggsed.hmc.chiral_mass import chiral_mass as ncm
    from unhiggsed.hmc.majorana import pfaffian_householder
    from unhiggsed.hmc.wedge import WedgeLattice, build_kinetic_d, source_pattern

    out = {}
    _spin10(nsp, out)
    _characters(nla, out)
    _eta(net, out)
    _grassmann(ngr, nsp, out)

    def field(n):
        nst.set_lattice(n)
        return dict(
            sym2=nst.sym2_term,
            lam2=nst.lam2_term,
            eps=nst.eps_term,
            map_su4=lambda U, par: nst.field_map_su4(U, len(par), par),
            substitute=nst.substitute,
            bag_weights=lambda M, q, t, ks, nm: nst.bag_weights(M, q, t, ksum=ks, normalise=nm),
            link_pfaffians=nst.link_pfaffians,
            kmonomial_table=nst.kmonomial_table,
        )

    _lattice_grass(nst, nst.pfaffian, field, out)
    _hamiltonian(nst, out)

    # the C148 bag functions are defined inside the notebook check; reproduce them with the notebook primitives
    def pf(A):
        n = A.shape[0]
        if n == 0:
            return 1.0
        if n == 2:
            return A[0, 1]
        if n == 4:
            return A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2]
        s, lg = pfaffian_householder(A)
        return s * np.exp(lg)

    def weights(Ms, subsets):
        Gs = [np.linalg.inv(M) for M in Ms]
        out_ = {}
        for B in subsets:
            idx = list(B)
            fac = [pf(G[np.ix_(idx, idx)]) for G in Gs]
            out_[B] = (float(np.prod(fac)), fac)
        return Gs, out_

    def wick_full(Gs, B):
        n = 4 * len(B)
        A = np.zeros((n, n))
        for i, x in enumerate(B):
            for j, y in enumerate(B):
                for a in range(4):
                    A[4 * i + a, 4 * j + a] = Gs[a][x, y]
        return pf(A)

    def order(Ms, Cps, subsets):
        V = Ms[0].shape[0]
        Nk, Zk = {}, {}
        all_idx = np.arange(V)
        eps = 1e-6
        for B in subsets:
            k = len(B)
            comp = np.setdiff1d(all_idx, B)
            p, d = [], []
            for a in range(4):
                A = Ms[a][np.ix_(comp, comp)]
                Cp = Cps[a]
                sv_min = np.linalg.svd(A, compute_uv=False)[-1] if A.shape[0] else 1.0
                if sv_min > 1e-9:
                    s_, l_ = (
                        pfaffian_householder(A)
                        if A.shape[0] > 4
                        else (np.sign(pf(A)) or 1.0, np.log(abs(pf(A)) + 1e-300))
                    )
                    pa = s_ * np.exp(l_)
                    da = (
                        0.0
                        if Cp is None
                        else -0.5
                        * pa
                        * np.einsum("ij,ji->", np.linalg.solve(A, np.eye(A.shape[0])), Cp[np.ix_(comp, comp)])
                    )
                else:
                    pa = 0.0
                    da = (
                        0.0 if Cp is None else float(np.imag(nst.pfaffian(A - 1j * eps * Cp[np.ix_(comp, comp)])) / eps)
                    )
                p.append(pa)
                d.append(da)
            Zk[k] = Zk.get(k, 0.0) + float(np.prod(p))
            val = 0.0
            for a in range(4):
                val += d[a] * float(np.prod([p[b] for b in range(4) if b != a]))
            Nk[k] = Nk.get(k, 0.0) + val
        return Nk, Zk

    _bags148(
        lambda shape: WedgeLattice(shape),
        lambda lat: build_kinetic_d(lat).toarray(),
        lambda lat, ds=1: ncm(lat, dual_sign=ds).toarray(),
        lambda lat: source_pattern(lat).toarray(),
        dict(weights=weights, wick_full=wick_full, order=order),
        out,
    )
    _signfree_notebook(out)
    return out


def _signfree_notebook(out):
    """C040 on-site structure, C165 class distances and commutants, C152 character sums -- notebook code."""
    import scipy.linalg as sla
    import scipy.sparse as sps
    from unhiggsed.hmc.chiral_mass import PH, chiral_mass, plane_op
    from unhiggsed.hmc.dirac import Dirac
    from unhiggsed.hmc.lattice import Lattice
    from unhiggsed.hmc.majorana import GAMMA
    from unhiggsed.hmc.observables import n_from_blocks
    from unhiggsed.hmc.wedge import WedgeLattice, build_kinetic_d, source_pattern

    rng = np.random.default_rng(SEED + 2)
    lat = Lattice((4, 4, 4, 4))
    sig = lat.random_sigma(rng)
    M = Dirac(lat, 1.3).dense(sig)
    Minv = np.linalg.inv(M)
    blocks = Minv.reshape(lat.V, 2, lat.V, 2)[np.arange(lat.V), :, np.arange(lat.V), :]
    out["onsite_n"] = n_from_blocks(blocks)
    out["onsite_ev"] = np.sort_complex(np.linalg.eigvals(M))

    def realmap(f):
        Mm = np.zeros((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = 1.0
            z = np.array([e[0] + 1j * e[1], e[2] + 1j * e[3]])
            w = f(z)
            Mm[:, j] = [w[0].real, w[0].imag, w[1].real, w[1].imag]
        return Mm

    tau2 = np.array([[0, -1j], [1j, 0]])
    Ji, Jt = realmap(lambda z: 1j * z), realmap(lambda z: tau2 @ np.conj(z))
    out["Ji"], out["Jt"] = Ji, Jt
    rows = [np.kron(np.eye(4), g) - np.kron(g.T, np.eye(4)) for g in (Ji, Jt)]
    out["commutant_dim"] = np.array([sla.null_space(np.vstack(rows)).shape[1]])
    lat4 = WedgeLattice((4,) * 4)
    V4 = lat4.V
    B = [np.eye(4)] + list(GAMMA)

    def class_distance(Mm):
        Mb = Mm.reshape(V4, 4, V4, 4)
        P = np.zeros_like(Mb)
        for b in B:
            P += np.einsum("xayb,ab->xy", Mb, b)[:, None, :, None] * b[None, :, None, :] / 4.0
        return np.linalg.norm(P - Mb) / np.linalg.norm(Mb)

    C0, Cx = source_pattern(lat4).toarray(), chiral_mass(lat4).toarray()
    out["class_dist"] = np.array(
        [
            class_distance(np.kron(Cx, np.eye(4))),
            class_distance(np.kron(C0, np.diag([0, 0, 0, 1.0]))),
            class_distance(np.kron(Cx, np.diag([0, 0, 1, 1.0]))),
            class_distance(np.kron(Cx, np.diag([0.2, 0.5, 1.0, 1.3]))),
        ]
    )
    # C152 item 2 (character sums), notebook code
    K = build_kinetic_d(lat4)
    V = V4

    def site_map(perm, refl):
        co = lat4.coords.reshape(4, V)
        new = np.zeros_like(co)
        for mu in range(4):
            new[perm[mu]] = (refl[mu] * co[mu]) % lat4.shape[mu]
        return np.ravel_multi_index(tuple(new), lat4.shape)

    def translation(mu):
        co = lat4.coords.reshape(4, V).copy()
        co[mu] = (co[mu] + 1) % lat4.shape[mu]
        return np.ravel_multi_index(tuple(co), lat4.shape)

    def sign_field(g):
        Kc = K.tocsr()
        Kd = Kc.toarray()
        s = np.zeros(V)
        s[0] = 1.0
        queue = [0]
        while queue:
            x = queue.pop()
            for y in Kc.indices[Kc.indptr[x] : Kc.indptr[x + 1]]:
                if Kd[x, y] == 0.0:
                    continue
                r = Kd[x, y] / Kd[g[x], g[y]]
                if s[y] == 0.0:
                    s[y] = s[x] * r
                    queue.append(y)
        return s

    gens = [translation(mu) for mu in range(4)] + [site_map(p, (1, 1, 1, 1)) for p in ((1, 0, 2, 3), (0, 2, 1, 3))]
    for mu in range(4):
        r = [1, 1, 1, 1]
        r[mu] = -1
        gens.append(site_map((0, 1, 2, 3), tuple(r)))
    gens = [(g, sign_field(g)) for g in gens]
    Ts = [sps.csr_matrix((s, (g, np.arange(V))), shape=(V, V)) for g, s in gens]
    act = lambda T, C: (T @ C @ T.T).tocsr()
    vec = lambda C: C.toarray().reshape(-1)

    def closure(seed):
        basis, mats = [], []

        def add(C):
            v = vec(C)
            if basis:
                Bm = np.array(basis)
                coef = np.linalg.lstsq(Bm.T, v, rcond=None)[0]
                if np.linalg.norm(Bm.T @ coef - v) < 1e-9 * max(1.0, np.linalg.norm(v)):
                    return
            basis.append(v)
            mats.append(C)

        for C in seed:
            add(C)
        i = 0
        while i < len(mats):
            for T in Ts:
                add(act(T, mats[i]))
            i += 1
        return np.linalg.qr(np.array(basis).T)[0]

    QW = closure([source_pattern(lat4)])
    QM = closure([plane_op(lat4, *D) for D in PH])
    out["orbit_QW"], out["orbit_QM"] = QW, QM

    def rep(Q, T):
        return np.array([Q.T @ vec(act(T, sps.csr_matrix(Q[:, k].reshape(V, V)))) for k in range(Q.shape[1])]).T

    RW, RM = [rep(QW, T) for T in Ts], [rep(QM, T) for T in Ts]
    key = lambda g, s: g.tobytes() + (s * s[0]).astype(np.int8).tobytes()
    ident = (np.arange(V), np.ones(V))
    elems = {key(*ident): (ident, np.eye(QW.shape[1]), np.eye(QM.shape[1]))}
    queue = [key(*ident)]
    while queue:
        (g, s), Rw, Rm = elems[queue.pop()]
        for (tg, ts), Tw, Tm in zip(gens, RW, RM, strict=True):
            g2, s2 = tg[g], s * ts[g]
            k2 = key(g2, s2)
            if k2 not in elems:
                elems[k2] = ((g2, s2), Tw @ Rw, Tm @ Rm)
                queue.append(k2)
    cW = np.array([np.trace(e[1]) for e in elems.values()])
    cM = np.array([np.trace(e[2]) for e in elems.values()])
    cW2 = np.array([np.trace(e[1] @ e[1]) for e in elems.values()])
    cW3 = np.array([np.trace(e[1] @ e[1] @ e[1]) for e in elems.values()])
    out["chars_W"], out["chars_M"] = cW, cM
    out["chars_W23"] = np.stack([cW2, cW3])


# ---------------------------------------------------------------------------------------------------------------
# clean side
# ---------------------------------------------------------------------------------------------------------------
def clean_side():
    from types import SimpleNamespace

    from masspairing.algebra import bags as cbags
    from masspairing.algebra import characters as cla
    from masspairing.algebra import eta as cet
    from masspairing.algebra import grassmann as cgr
    from masspairing.algebra import hamiltonian as cha
    from masspairing.algebra import spin10 as csp
    from masspairing.algebra import staggered as cst
    from masspairing.lattice import Lattice, kinetic_matrix
    from masspairing.patterns import chiral_mass, source_pattern

    out = {}
    _spin10(csp, out)
    _characters(cla, out)
    _eta(cet, out)
    _grassmann(cgr, csp, out)

    def field(n):
        fa = cst.FieldAlgebra(n)
        return dict(
            sym2=fa.sym2_term,
            lam2=fa.lam2_term,
            eps=fa.eps_term,
            map_su4=fa.map_su4,
            substitute=cgr.substitute,
            bag_weights=lambda M, q, t, ks, nm: cbags.bag_weights(M, q, t, ksum=ks, normalise=nm),
            link_pfaffians=cbags.link_pfaffians,
            kmonomial_table=cbags.kmonomial_table,
        )

    st = SimpleNamespace(
        staggered_matrix=cst.staggered_matrix,
        plaquette_quads=cst.plaquette_quads,
        line_quads=cst.line_quads,
        oriented_links=cst.oriented_links,
        sym2_yukawa_matrix=cst.sym2_yukawa_matrix,
    )
    _lattice_grass(st, cgr.pfaffian, field, out)
    _hamiltonian(cha, out)
    _bags148(
        lambda shape: Lattice(shape),
        lambda lat: kinetic_matrix(lat).toarray(),
        lambda lat, ds=1: chiral_mass(lat, dual_sign=ds).toarray(),
        lambda lat: source_pattern(lat).toarray(),
        dict(weights=cbags.site_bag_weights, wick_full=cbags.wick_full, order=cbags.order_coefficients),
        out,
    )
    _signfree_clean(out)
    return out


def _signfree_clean(out):
    from masspairing.algebra.orbits import group_representations, orbit_span, perm_matrix, power_traces
    from masspairing.algebra.signfree import class_distance, commutant, onsite_structure, quaternion_units
    from masspairing.lattice import Lattice, kinetic_matrix
    from masspairing.operator import DoubletOperator
    from masspairing.patterns import PH, chiral_mass, plane_op, source_pattern
    from masspairing.symmetry import sign_field, site_map, translation

    rng = np.random.default_rng(SEED + 2)
    lat = Lattice((4, 4, 4, 4))
    sig = lat.random_sigma(rng)
    M = DoubletOperator(lat, 1.3).dense(sig.reshape(-1))
    out["onsite_n"] = onsite_structure(M, lat.V)["n"]
    out["onsite_ev"] = np.sort_complex(np.linalg.eigvals(M))
    Ji, Jt = quaternion_units()
    out["Ji"], out["Jt"] = Ji, Jt
    out["commutant_dim"] = np.array([commutant([Ji, Jt], 4).shape[1]])
    lat4 = Lattice((4,) * 4)
    V4 = lat4.V
    C0, Cx = source_pattern(lat4).toarray(), chiral_mass(lat4).toarray()
    out["class_dist"] = np.array(
        [
            class_distance(np.kron(Cx, np.eye(4)), V4),
            class_distance(np.kron(C0, np.diag([0, 0, 0, 1.0])), V4),
            class_distance(np.kron(Cx, np.diag([0, 0, 1, 1.0])), V4),
            class_distance(np.kron(Cx, np.diag([0.2, 0.5, 1.0, 1.3])), V4),
        ]
    )
    K = kinetic_matrix(lat4)
    gens = [translation(lat4, mu) for mu in range(4)]
    gens += [site_map(lat4, p, (1, 1, 1, 1)) for p in ((1, 0, 2, 3), (0, 2, 1, 3))]
    for mu in range(4):
        r = [1, 1, 1, 1]
        r[mu] = -1
        gens.append(site_map(lat4, (0, 1, 2, 3), tuple(r)))
    gens = [(g, sign_field(K, g)) for g in gens]
    Ts = [perm_matrix(g, s) for g, s in gens]
    QW = orbit_span([source_pattern(lat4)], Ts)
    QM = orbit_span([plane_op(lat4, *D) for D in PH], Ts)
    out["orbit_QW"], out["orbit_QM"] = QW, QM
    elems = group_representations(gens, [QW, QM])
    pW = power_traces([e[0] for e in elems], 3)
    out["chars_W"] = pW[0]
    out["chars_M"] = np.array([np.trace(e[1]) for e in elems])
    out["chars_W23"] = np.stack([pW[1], pW[2]])
