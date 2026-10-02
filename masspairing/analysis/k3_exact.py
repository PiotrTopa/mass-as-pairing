"""Exactness tools of the instruments I.4-I.7 (link-field model, complete channel estimator, pairing source).

* Exact Hubbard-Stratonovich side on tiny lattices (``pf_polynomial``, ``hs_exact``): for one real flavour
  Pf(K - A(s))/Pf(K) = <exp(sum_f s_f B_f)>_0 is a polynomial P(s) of degree <= V/2 in the link fields; the
  four-flavour weight is P^4 = (P^2)^2 and its Gaussian averages E_s[(P^2 Q)^2] are evaluated exactly in the
  probabilists' Hermite basis of t_f = s_f / sigma_f (E[He_a He_b] = a! delta_ab). The fermion-bag side
  (``bag_reference``) enumerates <prod_Q T_Q^{k_Q}/k_Q!>_0 of the completed vertex to all orders.
* Exact-determinant Metropolis (``metropolis``): single-variable updates of (sigma, s) with the weight
  det D_c(sigma, s) e^{-S_B} from a dense log-determinant (no pseudofermions, no molecular dynamics), and the RHMC
  reference runs (``rhmc``) with the same measurement; the chains of the claims are registered in ``CHAINS``.
* The complete 10/6-channel correlators from an independent real-flavour Wick evaluation (``pair_pieces``,
  ``channel_reference``), with the controls that drop the K3K4 exchange term or replace tr(G3 G4) by the contraction
  sum_ab G3^{ab} G4^{ab}, and the noise-sample K3K4 term at p = 0 (``stochastic_k3k4``).
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from ..action import build_model
from ..algebra.bags import bag_weights
from ..algebra.grassmann import Grass, wick
from ..algebra.staggered import FieldAlgebra, staggered_matrix
from ..flavour import real_flavour_block
from ..hmc import HMC
from ..lattice import Lattice, shift_table
from ..measure import ChannelMeasure, scalar_observables
from ..wedge import link_field_observables

# ---- polynomials in the link fields: (keys, c) with key = sum_f n_f 16^f (exponents < 16, at most 15 fields) ------
BASE = 16


def encode(E):
    E = np.asarray(E, dtype=np.int64)
    return (E * BASE ** np.arange(E.shape[1], dtype=np.int64)).sum(1)


def decode(keys, nf):
    return (np.asarray(keys)[:, None] // BASE ** np.arange(nf, dtype=np.int64)) % BASE


def _aggregate(keys, c):
    u, inv = np.unique(keys, return_inverse=True)
    out = np.bincount(inv.reshape(-1), weights=c, minlength=len(u))
    keep = out != 0.0
    return u[keep], out[keep]


def poly_mul(p, q):
    (k1, c1), (k2, c2) = p, q
    return _aggregate((k1[:, None] + k2[None, :]).reshape(-1), (c1[:, None] * c2[None, :]).reshape(-1))


def poly_shift(p, f, k=1):
    """p(s) s_f^k."""
    return p[0] + k * BASE**f, p[1]


def poly_eval(p, s):
    E = decode(p[0], len(s))
    return float((p[1] * np.prod(np.asarray(s, float)[None, :] ** E, axis=1)).sum())


def pf_polynomial(geom, K0):
    """P(s) = Pf(K0 - A(s))/Pf(K0) for one real flavour by Wick's theorem: B_f = sum_l I[f, l] chi_u chi_v with the
    incidence the operator uses; monomials s^n with n_f <= 2 (B_f^3 = 0 on two disjoint links), sum n <= V/2."""
    V = K0.shape[0]
    G = np.linalg.inv(K0)
    nf = geom.n_fields
    I = geom.I.tocsr()
    B = []
    for f in range(nf):
        b = Grass()
        for link, cf in zip(
            I.indices[I.indptr[f] : I.indptr[f + 1]], I.data[I.indptr[f] : I.indptr[f + 1]], strict=True
        ):
            b = b + Grass.gen(int(geom.lu[link])) * Grass.gen(int(geom.lv[link])) * float(cf)
        B.append(b)
    Bpow = [[Grass.one(), b, b * b * 0.5] for b in B]
    terms = {}

    def rec(f, rem, n, p):
        if f == nf:
            val = sum(c * wick(G, [i for i in range(V) if m >> i & 1]).real for m, c in p.t.items())
            if abs(val) > 1e-14:
                terms[tuple(n)] = val
            return
        for k in range(min(2, rem) + 1):
            q = p if k == 0 else p * Bpow[f][k]
            if not q.t:
                continue
            rec(f + 1, rem - k, n + [k], q)

    rec(0, V // 2, [], Grass.one())
    assert nf <= 15
    return encode(np.array(list(terms), dtype=np.int64).reshape(-1, nf)), np.array(list(terms.values()))


def _hermite_table(nmax):
    """t^n = sum_k a_nk He_k(t), a_nk = n! / (k! ((n-k)/2)! 2^((n-k)/2)) for n - k even."""
    A = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        for k in range(n % 2, n + 1, 2):
            j = (n - k) // 2
            A[n, k] = math.factorial(n) / (math.factorial(k) * math.factorial(j) * 2**j)
    return A


def hermite(p, sigma):
    """Coefficients of p(sigma t) in the product Hermite basis He_alpha(t): (alpha keys, coefficients)."""
    keys, c = p
    sigma = np.asarray(sigma, float)
    nf = len(sigma)
    E = decode(keys, nf)
    A = _hermite_table(int(E.max()) if E.size else 0)
    lens = np.array([np.count_nonzero(A[m]) for m in range(A.shape[0])])
    KS = np.zeros((A.shape[0], lens.max()), dtype=np.int64)
    for m in range(A.shape[0]):
        KS[m, : lens[m]] = np.nonzero(A[m])[0]
    coef = c * np.prod(sigma[None, :] ** E, axis=1)
    akey = np.zeros(len(c), dtype=np.int64)
    src = np.arange(len(c))  # term of p behind each expanded row
    for f in range(nf):
        n = E[src, f]
        cnt = lens[n]
        idx = np.repeat(np.arange(len(n)), cnt)
        pos = np.arange(len(idx)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        k = KS[n[idx], pos]
        coef = coef[idx] * A[n[idx], k]
        akey = akey[idx] + k * BASE**f
        src = src[idx]
    return _aggregate(akey, coef)


def gaussian_moment(p, q, sigma, hq=None):
    """E_{s_f ~ N(0, sigma_f^2)}[p(s) q(s)] exactly: sum_alpha p_alpha q_alpha alpha! (``hq``: q already in the
    Hermite basis)."""
    ka, ca = hermite(p, sigma)
    kb, cb = hq if hq is not None else hermite(q, sigma)
    common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    alpha = decode(common, len(sigma))
    fact = np.array([math.factorial(k) for k in range(int(alpha.max(initial=0)) + 1)], float)
    return float((ca[ia] * cb[ib] * np.prod(fact[alpha], axis=1)).sum())


def hs_exact(geom, K0, sigma=None, poly=None):
    """Z/Z0 = E[P^4], <s_f^2>, <s_f> and <sum_Q term_Q> = <sum_f (s_f^2 - 2 g_f)/(4 g_f)> of the link-field model.
    sigma: prior widths (default sqrt(2 g_f) of the quad couplings; the per-link model needs sqrt(1/(2 inv4g)))."""
    poly = pf_polynomial(geom, K0) if poly is None else poly
    g = np.asarray(geom.f_g, float)
    sigma = np.sqrt(2.0 * g) if sigma is None else np.asarray(sigma, float)
    P2 = poly_mul(poly, poly)
    hP2 = hermite(P2, sigma)
    Z = gaussian_moment(P2, None, sigma, hq=hP2)
    s2, s1 = [], []
    for f in range(geom.n_fields):
        hf = hermite(poly_shift(P2, f), sigma)
        s2.append(gaussian_moment(poly_shift(P2, f), None, sigma, hq=hf) / Z)
        s1.append(gaussian_moment(P2, None, sigma, hq=hf) / Z)
    s2, s1 = np.array(s2), np.array(s1)
    return dict(Z_over_Z0=Z, sum_s2=float(s2.sum()), s2=s2, s1=s1, sum_term=float(((s2 - 2 * g) / (4 * g)).sum()))


def bag_reference(geom, kind="completed"):
    """All-orders bag enumeration on 2^3 (staggered units: chi' = sqrt(2) chi, couplings g/4).

    kind: "completed" (the HS vertex g1 (Kt_nu + Kt_nu')^2 + g2 (Kt_mu + Kt_mu')^2 per quad), "pure" (the uncompleted
    g10 T_Q + g6 T_6) or "links" (the per-link vertex c_l g K_l^2). Returns Z/Z0, <sum term>, number of
    configurations."""
    L, d = geom.lat.shape[0], geom.lat.d
    M, _, _ = staggered_matrix(L, d)
    fa = FieldAlgebra(geom.lat.V)
    if kind == "links":
        lk = [(int(u), int(v)) for u, v in zip(geom.lu, geom.lv, strict=True)]
        cl = {q: float(n) for q, n in zip(lk, geom.link_count, strict=True)}
        res = bag_weights(
            M, lk, lambda q: fa.K(*q) * fa.K(*q) * (cl[q] * geom.g10 / 4), ksum=8, kmax=2, normalise=False
        )
    else:
        qmap = {tuple(q): i for i, q in enumerate(geom.qsites)}

        def term(q):
            if kind == "pure":
                return fa.sym2_term(q) * (geom.g10 / 4) + fa.lam2_term(q) * (geom.g6 / 4)
            ln = geom.qlinks[qmap[tuple(q)]]
            Kt = [fa.K(int(geom.lu[i]), int(geom.lv[i])) * float(geom.lsign[i]) for i in ln]
            B1, B2 = Kt[0] + Kt[1], Kt[2] + Kt[3]
            return B1 * B1 * (geom.g[0] / 4) + B2 * B2 * (geom.g[1] / 4)

        res = bag_weights(M, [tuple(q) for q in geom.qsites], term, ksum=8, normalise=False)
    Z = 1.0 + sum(w for _, w in res)
    return dict(Z_over_Z0=Z, sum_term=sum(sum(c) * w for c, w in res) / Z, n_configs=len(res))


# ---- exact-determinant Metropolis and RHMC reference runs -------------------------------------------------------
def collect(series, model, fm, fields):
    """Scalar observables of one configuration: sigma field, link fields, all scalar keys of the dense channel
    measure (complete 10/6 channels, eps channel, bonds, source direction)."""
    sigma, s = model.split(fields)
    out = {}
    if model.lat.d == 4:
        so = scalar_observables(model.lat, sigma)
        out.update(sigma2=so["sigma2"], Sigma_stag_abs=so["Sigma_stag_abs"])
    out["s2"] = link_field_observables(model.geom, s)["s2"]
    b = fm.bilinears(fields)
    for k, v in b.items():
        if np.ndim(v) == 0:
            out[k] = float(v)
    out["E_bond"] = float(np.mean(b["E_bond"]))
    for k, v in out.items():
        series.setdefault(k, []).append(v)


def metropolis(model, nsweeps, seed, delta_s=0.7, delta_sig=0.6, meas_every=5, measure_from=0, check_sign=False):
    """Single-variable Metropolis on (sigma, s) with the dense determinant: weight det D_c(sigma, s) e^{-S_B}.

    A sweep visits the V sigma sites and the link fields in a random order; a sigma proposal adds delta_sig N(0, 1)
    to the triplet at one site, a link proposal delta_s N(0, 1) to one field (dense entries updated from the
    incidence). Measurement every ``meas_every`` sweeps from sweep ``measure_from``; ``check_sign`` asserts det > 0
    on every proposal."""
    lat, g = model.lat, model.geom
    V = lat.V
    y = model.y
    rng = np.random.default_rng(seed)
    fm = ChannelMeasure(model, exact=True)
    F = model.start(rng, "hot")
    D = model.D.dense(F)
    ld = np.linalg.slogdet(D)[1]
    sb = model.s_boson(F)
    I = g.I.tocsr()
    touch = []
    for f in range(g.n_fields):
        ent = []
        for link, cf in zip(
            I.indices[I.indptr[f] : I.indptr[f + 1]], I.data[I.indptr[f] : I.indptr[f + 1]], strict=True
        ):
            u, v = int(g.lu[link]), int(g.lv[link])
            for c in (0, 1):
                ent.append((2 * u + c, 2 * v + c, -cf))
                ent.append((2 * v + c, 2 * u + c, cf))
        touch.append(ent)

    def yblock(s3):
        return 1j * y * np.array([[s3[2], s3[0] - 1j * s3[1]], [s3[0] + 1j * s3[1], -s3[2]]])

    series = {}
    for sweep in range(nsweeps):
        for i in rng.permutation(V + g.n_fields):
            Fp = F.copy()
            Dp = D.copy()
            if i < V:
                x = int(i)
                Fp[[x, V + x, 2 * V + x]] += delta_sig * rng.normal(size=3)
                Dp[2 * x : 2 * x + 2, 2 * x : 2 * x + 2] = yblock(Fp[[x, V + x, 2 * V + x]])
            else:
                f = int(i - V)
                ds = delta_s * rng.normal()
                Fp[model.nsig + f] += ds
                for a, b, cf in touch[f]:
                    Dp[a, b] += cf * ds
            sign, ldp = np.linalg.slogdet(Dp)
            if check_sign:
                assert abs(sign - 1.0) < 1e-8, "negative or complex weight"
            sbp = model.s_boson(Fp)
            if np.log(rng.random()) < (ldp - ld) - (sbp - sb):
                F, D, ld, sb = Fp, Dp, ldp, sbp
        if sweep >= measure_from and sweep % meas_every == 0:
            collect(series, model, fm, F)
    assert abs(D - model.D.dense(F)).max() < 1e-10
    return {k: np.asarray(v) for k, v in series.items()}


def rhmc(model, ntraj, seed, nsteps, ntherm=200, tau_jitter=0.0):
    """RHMC reference run with the measurement of ``metropolis`` after every trajectory past ``ntherm``."""
    hmc = HMC(model, tau=1.0, nsteps=nsteps, seed=seed, tau_jitter=tau_jitter)
    fm = ChannelMeasure(model, exact=True)
    F = model.start(hmc.rng, "hot")
    series = {}
    for it in range(ntherm + ntraj):
        F, info = hmc.trajectory(F)
        if it < ntherm:
            continue
        series.setdefault("accepted", []).append(info["accepted"])
        collect(series, model, fm, F)
    return {k: np.asarray(v) for k, v in series.items()}


def hmc_2x3(ntraj, seed=73, ntherm=300):
    """HMC on 2^3 at y = 0 (interior point g10 = 0.4, g6 = 0.1): sum s^2, sum term, sum s and the fermionic bond
    estimator sum_f 2 g_f <B_f>_ferm (dense) per trajectory."""
    model = model_2x3(0.4, 0.1)
    g = model.geom
    hmc = HMC(model, tau=1.0, nsteps=5, seed=seed)
    F = model.start(hmc.rng, "hot")
    gf = np.asarray(g.f_g, float)
    out = {k: [] for k in ("s2", "term", "s1", "bond", "accepted")}
    for it in range(ntherm + ntraj):
        F, info = hmc.trajectory(F)
        if it < ntherm:
            continue
        s = model.split(F)[1]
        S = model.D.dense_S(F)
        Kl = g.lsign * 2.0 * np.einsum("nii->n", S[g.lu, :, g.lv, :]).real
        bond = sum(
            2 * gf[f] * Kl[g.qlinks[g.f_quad[f], 2 * g.f_pair[f] : 2 * g.f_pair[f] + 2]].sum()
            for f in range(g.n_fields)
        )
        out["accepted"].append(info["accepted"])
        out["s2"].append(float((s**2).sum()))
        out["term"].append(float(((s**2 - 2 * gf) / (4 * gf)).sum()))
        out["s1"].append(float(s.sum()))
        out["bond"].append(float(bond))
    return {k: np.asarray(v) for k, v in out.items()}


def model_2x3(g10, g6, **kw):
    lat = Lattice((2, 2, 2), bc=(-1, -1, -1))
    return build_model(lat, 0.0, 0.0, 1.0, g10=g10, g6=g6, cg_tol=1e-10, **kw)


def flip_nu_sign(geom):
    """Control: a wrong sign in one Kt term of every nu pair (Kt_nu - Kt_nu' instead of Kt_nu + Kt_nu')."""
    I2 = geom.I.tolil()
    for f in range(geom.n_fields):
        if geom.f_pair[f] == 0:
            link = geom.qlinks[geom.f_quad[f], 1]
            I2[f, link] = -I2[f, link]
    geom.I = I2.tocsr()
    geom.IT = geom.I.T.tocsr()
    geom._dev = {}
    return geom


def flip_odd_source(model):
    """Control: the odd-sublattice half of the pairing pattern with the wrong sign (operator only; the measured
    source direction keeps the correct pattern)."""
    D = model.D
    t = D.merged_tables()
    eps = model.lat.eps_np.reshape(-1)
    rows = np.repeat(np.arange(model.lat.V), np.diff(t["indptr"]))
    t["ent_c"] = t["ent_c"] * np.where(eps[rows] < 0, -1.0, 1.0)
    t["_dev"].clear()
    C = D.C.tocoo()
    C.data = C.data * np.where(eps[C.row] < 0, -1.0, 1.0)
    D.C = C.tocsr()
    return model


def _wedge_2x4(quads="all", g10=0.4, g6=0.1, **kw):
    lat = Lattice((2, 2, 2, 2), bc=(-1, -1, -1, -1))
    return build_model(lat, 2.0, 0.0, 1.0, g10=g10, g6=g6, quads=quads, **kw)


def _source_2x2x2x4(h, **kw):
    lat = Lattice((2, 2, 2, 4), bc=(-1, -1, -1, -1))
    return build_model(lat, 2.0, 0.0, 1.0, g10=0.4, g6=0.1, h=h, **kw)


def _links_sabotage(model):
    model.geom.inv4g = np.full(model.geom.n_fields, 1.0 / (4.0 * model.geom.g10))
    model.geom._dev = {}
    return model


# name -> (runner, claims); every runner takes the number of sweeps/trajectories n (a prefix for the live checks)
CHAINS = {
    "wedge_2x3_hmc": dict(run=lambda n: hmc_2x3(n), n=3000, used_by="I.4"),
    "wedge_2x4_metropolis": dict(run=lambda n: metropolis(_wedge_2x4(), n, 76), n=10000, used_by="I.4"),
    "wedge_2x4_rhmc": dict(run=lambda n: rhmc(_wedge_2x4(cg_tol=1e-10), n, 77, nsteps=6), n=2000, used_by="I.4"),
    "wedge_2x4_rhmc_flip": dict(
        run=lambda n: rhmc(_flip(_wedge_2x4(cg_tol=1e-10)), n, 78, nsteps=6), n=600, used_by="I.4"
    ),
    "links_2x4_metropolis": dict(
        run=lambda n: metropolis(_wedge_2x4("links", 0.15, 0.0), n, 112, delta_s=0.9), n=10000, used_by="I.4"
    ),
    "links_2x4_rhmc": dict(
        run=lambda n: rhmc(_wedge_2x4("links", 0.15, 0.0, cg_tol=1e-10), n, 113, nsteps=8, tau_jitter=0.3),
        n=2000,
        used_by="I.4",
    ),
    "links_2x4_rhmc_prior": dict(
        run=lambda n: rhmc(
            _links_sabotage(_wedge_2x4("links", 0.15, 0.0, cg_tol=1e-10)), n, 114, nsteps=8, tau_jitter=0.3
        ),
        n=600,
        used_by="I.4",
    ),
    **{
        f"source_metropolis_h{h:g}": dict(
            run=lambda n, h=h, s=s: metropolis(_source_2x2x2x4(h), n, s, measure_from=200, check_sign=True),
            n=8000,
            used_by="I.7",
        )
        for h, s in ((0.0, 1120), (0.05, 1121), (0.1, 1122), (0.2, 1123))
    },
    "source_rhmc_h0.1": dict(
        run=lambda n: rhmc(_source_2x2x2x4(0.1, cg_tol=1e-10), n, 1124, nsteps=8, tau_jitter=0.3), n=2500, used_by="I.7"
    ),
    "source_rhmc_h0.1_flip": dict(
        run=lambda n: rhmc(flip_odd_source(_source_2x2x2x4(0.1, cg_tol=1e-10)), n, 1125, nsteps=8, tau_jitter=0.3),
        n=800,
        used_by="I.7",
    ),
}


def _flip(model):
    flip_nu_sign(model.geom)
    return model


def run_chain(name, n=None):
    ch = CHAINS[name]
    return ch["run"](ch["n"] if n is None else n)


# series stored for the checks (the 2^3 HMC run stores all of its own)
STORED_KEYS = (
    "accepted",
    "sigma2",
    "O4",
    "Sigma_stag_abs",
    "phi_stag_sq",
    "s2",
    "E_bond",
    "dimer_sq_par",
    "chi10",
    "chi6",
    "E10",
    "E6",
    "phi_src",
    "phi_src_even",
    "phi_src_odd",
    "phi_src_dh",
)


def save_chain(series, name, outdir):
    out = pathlib.Path(outdir) / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = series if name == "wedge_2x3_hmc" else {k: v for k, v in series.items() if k in STORED_KEYS}
    np.savez_compressed(out, meta=json.dumps(dict(name=name, n=CHAINS[name]["n"])), **keep)
    return out


def generate(name, outdir):
    """Run a registered chain in full and store its series (data/derived/k3/chains/<name>.npz)."""
    return save_chain(run_chain(name), name, outdir)


def load_chain(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files if k != "meta"}


def blocked(x, nb=20):
    """Mean and error from nb equal blocks."""
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


# ---- complete channels from an independent real-flavour Wick evaluation ------------------------------------------
def flavour_propagator(S):
    """G^{ab}(x, z) (V, 4, V, 4), real, from the doublet inverse S (V, 2, V, 2)."""
    return real_flavour_block(S.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)


def pair_pieces(Gf, y, yp, x, xp, variant=None):
    """sum_ab <Phibar^{ab}(y, y') Phi^{ab}(x, x')> ('phi') and the Lambda analogue ('lam') for index arrays, by Wick's
    theorem with G1 = G(y, x), G2 = G(y', x'), G3 = G(y, x'), G4 = G(y', x), K = tr G:
    disc - 2 K1K2 -+ 2 tr(G1 G2) + 2 tr(G3 G4) +- 2 K3K4. variant: None, 'no_K3K4' or 'old_contraction'
    (tr(G3 G4) -> sum_ab G3^{ab} G4^{ab})."""
    B = lambda u, v: Gf[u, :, v, :]
    Gyy, Gxx = B(y, yp), B(x, xp)
    sym = lambda G, s: G + s * G.transpose(0, 2, 1)
    disc10 = np.einsum("nab,nab->n", sym(Gyy, 1), sym(Gxx, 1))
    disc6 = np.einsum("nab,nab->n", sym(Gyy, -1), sym(Gxx, -1))
    G1, G2, G3, G4 = B(y, x), B(yp, xp), B(y, xp), B(yp, x)
    K = lambda M: np.einsum("naa->n", M)
    KK1, trGG1 = K(G1) * K(G2), np.einsum("nab,nba->n", G1, G2)
    KK2 = K(G3) * K(G4) * (0.0 if variant == "no_K3K4" else 1.0)
    trGG2 = np.einsum("nab,nab->n" if variant == "old_contraction" else "nab,nba->n", G3, G4)
    return dict(
        phi=disc10 - 2 * KK1 - 2 * trGG1 + 2 * trGG2 + 2 * KK2,
        lam=disc6 - 2 * KK1 + 2 * trGG1 + 2 * trGG2 - 2 * KK2,
    )


def channel_reference(meas, S, variant=None, bc_signs=True):
    """Corner structure factors (16 corners) and local energies of both channels from ``pair_pieces``, class-averaged
    over the 12 plane orientations of ``meas`` (a ChannelMeasure), with the fermionic boundary sign of each pair."""
    lat = meas.lat
    V, d = lat.V, lat.d
    Gf = flavour_propagator(S)
    xe, ye = meas.even, meas.odd
    cx, cy = meas.coords[:, xe].T, meas.coords[:, ye].T
    corners = [np.array(c, float) * np.pi for c in np.ndindex(*(2,) * d)]
    out = {"chi10_corners": np.zeros(len(corners)), "chi6_corners": np.zeros(len(corners)), "E10": 0.0, "E6": 0.0}
    for mu, _nu, _sg, de, do in meas.planes:
        je, we = shift_table(lat, de)
        jo, wo = shift_table(lat, do)
        em = np.zeros(d, int)
        em[mu] = 1
        jm, _ = shift_table(lat, tuple(em))
        if not bc_signs:
            we, wo = np.ones(V), np.ones(V)
        xd, yd = je[xe], jo[ye]
        n_y, n_x = len(ye), len(xe)
        yy, xx = np.repeat(ye, n_x), np.tile(xe, n_y)
        pc = pair_pieces(Gf, yy, np.repeat(yd, n_x), xx, np.tile(xd, n_y), variant)
        W = np.outer(wo[ye], we[xe]).reshape(-1)
        for k, p in enumerate(corners):
            ph = np.cos((cx @ p)[None, :] - (cy @ p)[:, None]).reshape(-1)
            out["chi10_corners"][k] += (W * pc["phi"] * ph).sum() / V
            out["chi6_corners"][k] += (W * pc["lam"] * ph).sum() / V
        yl = jm[xe]
        loc = pair_pieces(Gf, yl, jo[yl], xe, xd, variant)
        wl = we[xe] * wo[yl]
        out["E10"] += (wl * loc["phi"]).mean()
        out["E6"] += (wl * loc["lam"]).mean()
    no = len(meas.planes)
    for k in out:
        out[k] = out[k] / no
    out["chi10"], out["chi6"] = float(out["chi10_corners"][0]), float(out["chi6_corners"][0])
    return out


def stochastic_k3k4(meas, samples):
    """The 2 K3K4 contribution to the noise-sample chi10 at p = 0 (class-averaged), U-statistic over ordered pairs of
    distinct samples, as in the complete estimator."""
    T = meas._full_tables()
    V = meas.lat.V
    xe, ye = meas.even, meas.odd
    k = len(samples)
    Zs = [z for z, _ in samples]
    Es = [e for _, e in samples]
    al = [2.0 * np.stack([z[:, 0, 0].real, z[:, 1, 1].real], 1) for z in Zs]
    tot = 0.0
    for pl in T["planes"]:
        xd, yd, we, wo = pl["xd"], pl["yd"], pl["we"], pl["wo"]
        kk2 = 0.0
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                FA = (al[i][ye][:, :, None] * al[j][yd][:, None, :] * wo[:, None, None]).reshape(-1, 4).sum(0)
                G2 = (Es[i][xd][:, :, None] * Es[j][xe][:, None, :] * we[:, None, None]).reshape(-1, 4).sum(0)
                kk2 += float((FA * G2).sum())
        tot += 2.0 * kk2 / (k * (k - 1)) / V
    return tot / len(T["planes"])


def load_config(name, cg_tol=1e-10):
    """(fields, model) of a stored configuration data/configs/k3/<name>.npz."""
    from ..data import CONFIGS

    d = np.load(CONFIGS / "k3" / f"{name}.npz", allow_pickle=True)
    m = json.loads(str(d["meta"]))
    lat = Lattice(tuple(m["shape"]), bc=tuple(m["bc"]))
    model = build_model(lat, m["y"], m["kappa"], m["lam"], g10=m["g10"], g6=m["g6"], quads=m["quads"], cg_tol=cg_tol)
    return np.asarray(d["fields"]), model
