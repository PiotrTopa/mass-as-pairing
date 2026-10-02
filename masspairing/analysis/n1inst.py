"""Estimators and reference samplers of the N1 instrument claims (K4.1, K5.1, K5.2, I.8-I.11, K6.1, K6.2).

* Exact-determinant Metropolis samplers of sigma, the references the RHMC is compared with:
  ``metropolis_pfaffian`` (flavour-selective source, weight |Pf M| = |det M|^(1/2) of the real 4V operator, dense
  log-determinant per proposal) and ``metropolis_woodbury`` (flavour-blind source, weight det D_c > 0, rank-2 Woodbury
  updates of the dense inverse D_c^-1, det ratio det(1_2 + Delta D^-1_xx)).
* ``rhmc_series``: the RHMC chain with trajectory-length jitter and the same per-configuration observables.
* Statistics of the comparisons: ``blocked`` (blocked mean and error), ``reweighted`` (sign-reweighted blocked
  jackknife).
* ``takeover_rows``: the composite-takeover table of the nodal source on one flavour (K4.1).
* Patterns of the controls: ``flipped_pattern`` (the phase-flipped plane pair), ``one_link_mass``.
* ``free_phi_L``: the free light amplitude per flavour, the box-shape effect of K5.2; ``stabiliser_elements``: which
  elements of the stabiliser of C_chi flip the (0,3) light component.
* ``winding_record``: the corner-block loop readouts of one stored configuration (K6.2).
"""

from __future__ import annotations

import itertools
import json
import re
import time

import numpy as np
import scipy.sparse as sps

from ..action import build_model
from ..corner import CornerBlocks, analyse, classify, doublet_operator, quantised
from ..flavour import GAMMA
from ..hmc import HMC
from ..lattice import Lattice, kinetic_matrix
from ..measure import N1Measure, scalar_observables
from ..patterns import DUAL, TasteProjector, chiral_mass, plane_op, source_pattern
from ..pfaffian import pf_sign_config, pfaffian_householder
from ..symmetry import closure, generators, site_map, stabiliser, transform


# ---- control patterns ---------------------------------------------------------------------------------
def flipped_pattern(lat, comp=(0, 3)):
    """The phase-flipped plane pair: (plane (0,3) without its zeta phase + its dual) / 8. Sign-free (real
    antisymmetric) but not a chiral mass: it gaps both doublets at h/2."""
    return ((plane_op(lat, *comp, phase=()) + plane_op(lat, *DUAL[tuple(comp)])) / 8.0).tocsr()


def one_link_mass(lat, mu, S):
    """The one-link (reduced-staggered, shift-breaking) scalar mass along mu with phase string S."""
    from ..lattice import shift_table

    V, d = lat.V, lat.d
    co = lat.coords.reshape(d, V)
    zeta = (-1.0) ** (sum(co[r] for r in S) % 2) if S else np.ones(V)
    de = np.zeros(d, int)
    de[mu] = 1
    j, w = shift_table(lat, de)
    x = np.arange(V)
    v = w * zeta
    C = sps.csr_matrix((np.concatenate([v, -v]), (np.concatenate([x, j]), np.concatenate([j, x]))), shape=(V, V))
    C.sum_duplicates()
    return C


# ---- statistics -----------------------------------------------------------------------------------------
def blocked(x, nb=20):
    """Mean and error of nb equal blocks (the tail beyond nb * len // nb is dropped)."""
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb))


def reweighted(o, s, nb=20):
    """Sign-reweighted blocked mean <O s>/<s> and its jackknife error over nb blocks."""
    o, s = np.asarray(o, float), np.asarray(s, float)
    n = len(o) // nb * nb
    ob = (o[:n] * s[:n]).reshape(nb, -1).mean(1)
    sb = s[:n].reshape(nb, -1).mean(1)
    full = ob.sum() / sb.sum()
    parts = np.array([(ob.sum() - ob[b]) / (sb.sum() - sb[b]) for b in range(nb)])
    return float(full), float(np.sqrt((nb - 1) / nb * ((parts - parts.mean()) ** 2).sum()))


def pull(a, ea, b, eb):
    return (a - b) / np.sqrt(ea**2 + eb**2)


# ---- per-configuration observables of the comparisons -----------------------------------------------------
SELECTIVE_KEYS = [
    "sigma2",
    "Sigma_stag_abs",
    "O4",
    "phi_stag_sq",
    "E_bond",
    "dimer_sq_par",
    "phi_f_light",
    "phi_f_heavy",
    "phi_f_heavy_even",
    "phi_f_heavy_odd",
    "chi_f_light",
    "chi_f_heavy",
    "chi_f_conn_heavy",
    "phi_T",
    "Cf_heavy_1",
    "Cf_light_1",
    "C_T_1",
]

TASTE_KEYS = [
    "sigma2",
    "Sigma_stag_abs",
    "O4",
    "phi_stag_sq",
    "phi_R",
    "chi_R_sum",
    "chi_R_conn",
    "phi_L",
    "chi_L_sum",
    "chi_L_sum_conn",
    "chi_L_pmin",
    "phi6L_stag_sq",
    "O4L",
    "CL_1",
    "CR_1",
    "GL_1",
    "GL_p0",
    "GR_p0",
    "phi_T_R",
    "phi_T_L",
]


def selective_row(lat, model, fm, fields):
    """Observables of the flavour-selective comparison (source on flavour 4)."""
    so = scalar_observables(lat, model.split(fields)[0])
    fo = fm.bilinears(fields)
    co = fm.correlator(fields)
    return dict(
        sigma2=so["sigma2"],
        Sigma_stag_abs=so["Sigma_stag_abs"],
        O4=fo["O4"],
        phi_stag_sq=fo["phi_stag_sq"],
        E_bond=float(np.mean(fo["E_bond"])),
        dimer_sq_par=fo["dimer_sq_par"],
        phi_f_light=float(fo["phi_f"][:3].sum()),
        phi_f_heavy=float(fo["phi_f"][3]),
        phi_f_heavy_even=float(fo["phi_f_even"][3]),
        phi_f_heavy_odd=float(fo["phi_f_odd"][3]),
        chi_f_light=float(fo["chi_f"][:3].sum()),
        chi_f_heavy=float(fo["chi_f"][3]),
        chi_f_conn_heavy=float(fo["chi_f_conn"][3]),
        phi_T=fo["phi_T"],
        Cf_heavy_1=float(co["Cf_f"][3, 1]),
        Cf_light_1=float(co["Cf_f"][0, 1]),
        C_T_1=float(fo["C_T"][1]),
    )


def taste_row(lat, model, fm, fields):
    """Observables of the N1 comparison (flavour-blind taste-chiral mass)."""
    so = scalar_observables(lat, model.split(fields)[0])
    fo = fm.bilinears(fields)
    return dict(
        sigma2=so["sigma2"],
        Sigma_stag_abs=so["Sigma_stag_abs"],
        O4=fo["O4"],
        phi_stag_sq=fo["phi_stag_sq"],
        phi_R=float(fo["phi_f"].sum()),
        chi_R_sum=fo["chi_f_sum"],
        chi_R_conn=float(fo["chi_f_conn"].sum()),
        phi_L=float(fo["phi_L"].sum()),
        chi_L_sum=fo["chi_L_sum"],
        chi_L_sum_conn=fo["chi_L_sum_conn"],
        chi_L_pmin=float(fo["chi_L_pmin"].sum()),
        phi6L_stag_sq=fo["phi6L_stag_sq"],
        O4L=fo["O4L"],
        CL_1=float(fo["CL_t"][1]),
        CR_1=float(fo["CR_t"][1]),
        GL_1=float(fo["GL_t"][1]),
        GL_p0=fo["GL_p0"],
        GR_p0=fo["GR_p0"],
        phi_T_R=fo["phi_T_R"],
        phi_T_L=fo["phi_T_L"],
    )


def _append(series, row):
    for k, v in row.items():
        series.setdefault(k, []).append(v)


def _finish(series):
    return {k: np.asarray(v) for k, v in series.items()}


# ---- exact-determinant Metropolis samplers ----------------------------------------------------------------
def metropolis_pfaffian(
    shape, bc, y, kappa, lam, h_sel, mask, nsweeps, seed, ntherm=200, meas_every=3, delta_sig=0.6, record_trace=False
):
    """Metropolis on sigma with the exact weight |Pf M| e^(-S_B), |Pf M| = |det M|^(1/2) (M the real 4V operator
    of the flavour-selective source; no link fields). Per proposal one dense log-determinant. Measures
    ``selective_row`` and the Householder sign of Pf M every ``meas_every`` sweeps from sweep ``ntherm``.

    Returns the series (arrays), ``acc`` and, with ``record_trace``, the per-sweep boson action ``trace_sb``.
    """
    lat = Lattice(shape, bc=bc)
    V = lat.V
    rng = np.random.default_rng(seed)
    model = build_model(lat, y, kappa, lam, h_sel=h_sel, mask=mask)
    assert model.D.n_fields == 0
    fm = N1Measure(model, exact=True)
    F = model.start(rng, "hot")
    M = model.D.dense_real4(F)

    def logabs(A):
        return 0.5 * np.linalg.slogdet(A)[1]

    lp = logabs(M)
    sb = model.s_boson(F)

    def yblock(s3):
        return y * np.einsum("a,abc->bc", s3, GAMMA)

    series, trace = {}, []
    nacc = nprop = 0
    for sweep in range(nsweeps):
        for i in rng.permutation(V):
            x = int(i)
            Fp = F.copy()
            Mp = M.copy()
            Fp[[x, V + x, 2 * V + x]] += delta_sig * rng.normal(size=3)
            blk = slice(4 * x, 4 * x + 4)
            Mp[blk, blk] = M[blk, blk] - yblock(F[[x, V + x, 2 * V + x]]) + yblock(Fp[[x, V + x, 2 * V + x]])
            lpp = logabs(Mp)
            sbp = model.s_boson(Fp)
            nprop += 1
            if np.log(rng.random()) < (lpp - lp) - (sbp - sb):
                F, M, lp, sb = Fp, Mp, lpp, sbp
                nacc += 1
        if record_trace:
            trace.append(sb)
        if sweep >= ntherm and sweep % meas_every == 0:
            row = selective_row(lat, model, fm, F)
            row["sign"] = pfaffian_householder(M)[0]
            _append(series, row)
    assert abs(M - model.D.dense_real4(F)).max() < 1e-9
    out = _finish(series)
    out["acc"] = nacc / nprop
    if record_trace:
        out["trace_sb"] = np.asarray(trace)
    out["fields_final"] = np.asarray(F)
    return out


def metropolis_woodbury(
    shape,
    bc,
    y,
    kappa,
    lam,
    h,
    nsweeps,
    seed,
    ntherm=300,
    meas_every=8,
    refresh=50,
    delta_sig=0.6,
    pattern="chiral",
    record_trace=False,
    measure=True,
):
    """Metropolis on sigma with the exact weight det D_c e^(-S_B) (flavour-blind source h C, no link fields).

    D_c^-1 is kept by rank-2 Woodbury updates; the determinant ratio of a one-site proposal is det(1_2 + Delta
    D^-1_xx) (real up to rounding: its largest relative imaginary part is returned as ``worst_im``); the inverse is
    recomputed every ``refresh`` sweeps (asserting agreement to 1e-8). Measures ``taste_row`` every ``meas_every``
    sweeps from sweep ``ntherm``; the configuration of the last measurement is returned as ``fields_last``.
    """
    lat = Lattice(shape, bc=bc)
    V = lat.V
    rng = np.random.default_rng(seed)
    model = build_model(lat, y, kappa, lam, h=h, pattern=pattern)
    fm = N1Measure(model, exact=True, taste=True)
    assert fm.tp.n_boundary == 0, "the taste channels must be non-trivial on the test box"
    F = model.start(rng, "hot")
    Dinv = np.linalg.inv(model.D.dense(F))
    sb = model.s_boson(F)

    def yblock(s3):
        return 1j * y * np.array([[s3[2], s3[0] - 1j * s3[1]], [s3[0] + 1j * s3[1], -s3[2]]])

    series, trace = {}, []
    nacc = nprop = 0
    worst_im = 0.0
    last = None
    for sweep in range(nsweeps):
        for xx in rng.permutation(V):
            x = int(xx)
            sl = slice(2 * x, 2 * x + 2)
            Fp = F.copy()
            Fp[[x, V + x, 2 * V + x]] += delta_sig * rng.normal(size=3)
            Delta = yblock(Fp[[x, V + x, 2 * V + x]]) - yblock(F[[x, V + x, 2 * V + x]])
            Mx = np.eye(2) + Delta @ Dinv[sl, sl]
            r = np.linalg.det(Mx)
            worst_im = max(worst_im, abs(r.imag) / abs(r))
            sbp = model.s_boson(Fp)
            nprop += 1
            if np.log(rng.random()) < np.log(max(r.real, 1e-300)) - (sbp - sb):
                # (D + E Delta E^T)^-1 = D^-1 - D^-1 E (1 + Delta E^T D^-1 E)^-1 Delta E^T D^-1
                col = Dinv[:, sl]
                row = Dinv[sl, :]
                Dinv = Dinv - col @ (np.linalg.solve(Mx, Delta) @ row)
                F, sb = Fp, sbp
                nacc += 1
        if (sweep + 1) % refresh == 0:
            fresh = np.linalg.inv(model.D.dense(F))
            assert abs(fresh - Dinv).max() < 1e-8, abs(fresh - Dinv).max()
            Dinv = fresh
        if record_trace:
            trace.append(sb)
        if measure and sweep >= ntherm and sweep % meas_every == 0:
            _append(series, taste_row(lat, model, fm, F))
            last = np.asarray(F).copy()
    out = _finish(series)
    out["acc"] = nacc / nprop
    out["worst_im"] = worst_im
    if record_trace:
        out["trace_sb"] = np.asarray(trace)
    if last is not None:
        out["fields_last"] = last
    return out


# ---- RHMC reference chains -------------------------------------------------------------------------------------
def rhmc_series(model, row_fn, fm, ntraj, seed, ntherm=200, tau_jitter=0.3, sign=False, measure=True):
    """RHMC chain (tau = 1, 8 Omelyan steps, jitter) from a hot start with the generator's stream; after ``ntherm``
    trajectories every configuration is measured with ``row_fn(lat, model, fm, fields)`` (plus the Pfaffian sign
    of the selective source when ``sign``). Returns the series, ``acc``, the dH of every trajectory (``dH``) and the
    last configuration (``fields_last``)."""
    lat = model.lat
    hh = HMC(model, tau=1.0, nsteps=8, seed=seed, tau_jitter=tau_jitter)
    F = model.start(hh.rng, "hot")
    series, acc, dH = {}, [], []
    for it in range(ntherm + ntraj):
        F, info = hh.trajectory(F)
        dH.append(info["dH"])
        if it < ntherm:
            continue
        acc.append(info["accepted"])
        if measure:
            row = row_fn(lat, model, fm, F)
            if sign:
                row["sign"] = float(pf_sign_config(model, np.asarray(F), [model.D.h_sel])["sign"][0])
            _append(series, row)
    out = _finish(series)
    out["acc"] = float(np.mean(acc)) if acc else float("nan")
    out["dH"] = np.asarray(dH)
    out["fields_last"] = np.asarray(F)
    return out


# ---- K4.1: composite takeover ------------------------------------------------------------------------------------
def takeover_row(series, meta, nb=8):
    """phi_T/h, phi_heavy/h, phi_light/h and |phi_T|/phi_heavy of one chain with the nodal source on flavour 4.

    First quarter cut; sign-weighted means (weights s / sum s; the Pfaffian sign where recorded, nan -> 0); errors
    from nb equal blocks (unweighted). phi_heavy = flavour 4 of phi_f, phi_light = mean of flavours 1-3 (error of
    the mean / sqrt 3)."""
    pT = np.asarray(series["ts_phi_T"])
    pf = np.asarray(series["ts_phi_f"])
    n = len(pT)
    cut = n // 4
    pT, pf = pT[cut:], pf[cut:]
    s = np.asarray(series["ts_pf_sign"])[cut:] if len(series.get("ts_pf_sign", [])) == n else np.ones(n - cut)
    s = np.where(np.isnan(s), 0.0, s)
    W = s / s.sum() if s.sum() > 0 else np.ones(len(s)) / len(s)

    def berr(x):
        x = np.asarray(x)
        return np.array([x[i * len(x) // nb : (i + 1) * len(x) // nb].mean(0) for i in range(nb)]).std(0) / np.sqrt(nb)

    mT = float((W * pT).sum())
    eT = float(berr(pT))
    mf = (W[:, None] * pf).sum(0)
    ef = berr(pf)
    h = float(meta["h"])
    return dict(
        L=int(meta["L"]),
        y=float(meta["y"]),
        h=h,
        n=n - cut,
        sign=float(s.mean()),
        phiT_h=mT / h,
        ephiT_h=eT / h,
        heavy_h=float(mf[3]) / h,
        eheavy_h=float(ef[3]) / h,
        light_h=float(mf[:3].mean()) / h,
        elight_h=float(ef[:3].mean() / np.sqrt(3)) / h,
        ratio=abs(mT / float(mf[3])),
    )


def parse_chain_name(name):
    """(L, y, h) from a chain file name L<L>_y<y>_..._h<h>_..."""
    L = int(re.match(r"L(\d+)_", name).group(1))
    y = float(re.search(r"_y([0-9.]+)_", name).group(1))
    h = float(re.search(r"_h([0-9.]+)_", name).group(1))
    return L, y, h


# ---- K5.2: box-shape effect -----------------------------------------------------------------------------------
def free_phi_L(L, Lt, h, bc=(-1, -1, -1, -1)):
    """Free light amplitude per flavour, (1/2V) sum_xz C_asd(0,3)[x,z] G_L[x,z] with G_L = P_L (K - h C_chi)^-1 P_L."""
    lat = Lattice((L, L, L, Lt), bc=tuple(bc))
    V = lat.V
    Minv = np.linalg.inv(kinetic_matrix(lat).toarray() - h * chiral_mass(lat).toarray())
    tp = TasteProjector(lat)
    GL = tp.PL(np.ascontiguousarray(tp.PL(Minv).T)).T
    del Minv
    Ca = chiral_mass(lat, (0, 3), -1).tocoo()
    return float(0.5 * np.sum(Ca.data * GL[Ca.row, Ca.col]) / V)


def stabiliser_elements(shape):
    """The point-group stabiliser of C_chi on an all-antiperiodic box and the action of each element on C_asd(0,3)
    and on P_L C_asd(0,3) P_L (+1 keep, -1 flip), with whether it moves the time axis."""
    lat = Lattice(shape, bc=(-1,) * 4)
    K = kinetic_matrix(lat)
    G = closure(generators(lat, K, translations=False), lat.V)
    S = stabiliser(G, chiral_mass(lat).toarray())
    Ca = chiral_mass(lat, (0, 3), -1).toarray()
    PL = TasteProjector(lat).dense_PL()
    CL = PL @ Ca @ PL
    maps = {}
    for perm in itertools.permutations(range(4)):
        for refl in itertools.product((1, -1), repeat=4):
            g = site_map(lat, perm, refl)
            if g is not None:
                maps[g.tobytes()] = (perm, refl)
    rows = []
    for g, s in S:
        perm, refl = maps[g.tobytes()]
        rows.append(
            dict(
                perm=perm,
                moves_time=perm[3] != 3,
                act=float(np.sum(transform(Ca, g, s) * Ca) / np.sum(Ca * Ca)),
                act_L=float(np.sum(transform(CL, g, s) * CL) / np.sum(CL * CL)),
            )
        )
    return len(G), rows


# ---- K6.2: loop readouts of a stored configuration ------------------------------------------------------------
def winding_record(sigma, meta):
    """Corner-block loop readouts (p_vec = 0) of a stored sigma configuration: D_c(sigma) with the stored y and bc,
    one sparse LU, the corner block at every p_0 of the antiperiodic loop."""
    shape = tuple(sigma.shape[1:])
    bc = tuple(meta.get("bc", (1, 1, 1, -1)))
    y = float(meta["y"])
    lat = Lattice(shape, bc=bc)
    t0 = time.time()
    r = analyse(CornerBlocks(doublet_operator(lat, y, sigma), shape, 2), bc)["all"]
    r.pop("det_abs", None)
    rec = dict(
        L=shape[0],
        Lt=shape[-1],
        bc=list(bc),
        y=y,
        kappa=float(meta.get("kappa", np.nan)),
        wall_s=time.time() - t0,
        reading=classify(r),
        quantised=bool(quantised(r, 32)),
    )
    rec.update({k: (float(v) if np.isscalar(v) else v) for k, v in r.items()})
    return rec


def dumps(obj):
    """JSON text with numpy scalars/arrays converted (stable formatting)."""

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    return json.dumps(obj, default=conv, indent=1)


def stored_config(name):
    """(Lattice, chain meta, fields) of data/configs/n1inst/<name>.npz (fields flat, or sigma (3,) + shape)."""
    from ..data import CONFIGS

    z = np.load(CONFIGS / "n1inst" / f"{name}.npz", allow_pickle=True)
    info = json.loads(str(z["meta"]))
    meta = info["chain"]
    shape = tuple(meta["shape"]) if meta.get("shape") else tuple(np.asarray(z["fields"]).shape[1:])
    lat = Lattice(shape, bc=tuple(meta.get("bc") or (1, 1, 1, -1)))
    return lat, meta, np.asarray(z["fields"], float)


def stored_model(lat, meta, **kw):
    """The model of a stored chain (epsilon model, optional link fields), no source."""
    return build_model(
        lat,
        meta["y"],
        meta["kappa"],
        meta.get("lam", 1.0),
        g10=float(meta.get("g10") or 0.0),
        g6=float(meta.get("g6") or 0.0),
        quads=meta.get("quads") or "all",
        **kw,
    )


# ---- the reference comparisons of I.8 and I.10 (parameters, seeds, models) ----------------------------------
# I.8: exact |Pf| Metropolis vs RHMC on 2^3 x 4, epsilon model (y, kappa, lambda) = (2, 0, 1), h = 0.3 on flavour 4
I8 = dict(shape=(2, 2, 2, 4), bc=(-1, -1, -1, -1), y=2.0, kappa=0.0, lam=1.0, h=0.3, mask=(0.0, 0.0, 0.0, 1.0))
I8_MET = [(3000, 1400), (3000, 1401)]
I8_RHMC = [(1200, 1402, False), (400, 1403, True)]


def i8_model(sabotage=False):
    """The RHMC model of I.8; the sabotage flips the sign of C_0 on the odd sublattice (operator and measure)."""
    p = I8
    lat = Lattice(p["shape"], bc=p["bc"])
    m = build_model(lat, p["y"], p["kappa"], p["lam"], h_sel=p["h"], mask=p["mask"], cg_tol=1e-10)
    if sabotage:
        eps = lat.eps_np.reshape(-1)
        C = source_pattern(lat).tocoo()
        C.data = C.data * np.where(eps[C.row] < 0, -1.0, 1.0)
        m.D.C = C.tocsr()
        m.D._csel = None
    return m


def i8_metropolis(nsweeps, seed, **kw):
    p = I8
    return metropolis_pfaffian(
        p["shape"], p["bc"], p["y"], p["kappa"], p["lam"], p["h"], p["mask"], nsweeps, seed, **kw
    )


def i8_rhmc(ntraj, seed, sabotage, **kw):
    m = i8_model(sabotage)
    return rhmc_series(m, selective_row, N1Measure(m, exact=True), ntraj, seed, sign=True, **kw)


# I.10: exact-determinant Metropolis vs RHMC on 4^4 aaaa, epsilon model (2, -0.01, 1), h = 0.3 C_chi
I10 = dict(shape=(4, 4, 4, 4), bc=(-1, -1, -1, -1), y=2.0, kappa=-0.01, lam=1.0, h=0.3)
I10_MET = [(5000, 1610), (5000, 1611)]
I10_RHMC = [(2000, 1612, False), (500, 1613, True)]


def i10_model(flipped=False):
    p = I10
    lat = Lattice(p["shape"], bc=p["bc"])
    pattern = flipped_pattern(lat) if flipped else "chiral"
    return build_model(lat, p["y"], p["kappa"], p["lam"], h=p["h"], pattern=pattern, cg_tol=1e-10)


def i10_metropolis(nsweeps, seed, **kw):
    p = I10
    return metropolis_woodbury(p["shape"], p["bc"], p["y"], p["kappa"], p["lam"], p["h"], nsweeps, seed, **kw)


def i10_rhmc(ntraj, seed, flipped, **kw):
    m = i10_model(flipped)
    return rhmc_series(m, taste_row, N1Measure(m, exact=True, taste=True), ntraj, seed, **kw)
