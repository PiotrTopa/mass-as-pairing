"""Common tools of the N1 stage analyses (stage 0, stage 1, stage 1b): file locations, free baselines, the error model
and the effective-mass estimators.

Model N1 is the epsilon model at kappa = -0.01 with the flavour-blind taste-chiral Majorana mass h C_chi on one taste
doublet ("R", heavy) of every flavour; the other doublet ("L", light) is the one the K5 criteria read. The chains store
per measurement the P_L / P_R-projected time-slice correlators CL_t, CR_t (all-source, zero momentum; CL_t is a
positive two-fermion (pair) channel), the light pair susceptibility chi_L_sum = V <phi_L^2> and its p_min version, the
momentum readouts GL_p0, GR_p0 (1 for a free massless doublet), the elementary amplitude phi_f (per flavour, in the
C_chi direction), the composite amplitudes phi_T_R, phi_T_L, and per trajectory the bosonic series (S(pi),
|Sigma_stag|, sigma^2).

Error model (fixed before the stage-1 data were read): sigma = max{sigma_block(nb), sigma_naive sqrt(2 tau_eff)} with
tau_eff = max(tau_int(series), tau_B / Delta), tau_B the Madras-Sokal time of the per-trajectory bosonic series and
Delta the measurement spacing; nb = 20 blocks for >= 150 measurements, else 10. Ratios and effective masses are
delete-one-block jackknifes, inflated by the error-model factor of the series they come from.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..data import DERIVED

N1DIR = DERIVED / "n1stage"
FREEDIR = DERIVED / "free"


# ------------------------------------------------------------ free baselines
def free_tables(names):
    """Concatenated rows of the free-baseline files data/derived/free/<name> (one row per (L, Lt, bc, h))."""
    rows = []
    for name in names:
        rows += json.loads((FREEDIR / name).read_text())
    return rows


def free_lookup(rows, L, Lt, bc, h):
    for r in rows:
        if r["L"] == L and r["Lt"] == Lt and r["bc"] == bc and abs(r["h"] - h) < 1e-12:
            return r
    return None


def open_chain(path):
    """(npz handle, meta dict) of a derived chain file."""
    d = np.load(path, allow_pickle=False)
    return d, json.loads(str(d["meta"]))


# ------------------------------------------------------------ statistics
def tau_int(x, c=6.0, wmax=None):
    """Madras-Sokal integrated autocorrelation time in units of the series spacing: the first window W >= c tau,
    W <= wmax (default n // 4); floor 0.5."""
    x = np.asarray(x, float)
    n = len(x)
    x = x - x.mean()
    v = (x * x).mean()
    if n < 8 or v == 0:
        return 0.5
    wmax = wmax or max(2, n // 4)
    rho = np.array([(x[:-t] * x[t:]).mean() / v for t in range(1, wmax + 1)])
    tau = 0.5
    for W in range(1, len(rho) + 1):
        tau = 0.5 + rho[:W].sum()
        if W >= c * tau:
            break
    return float(max(tau, 0.5))


def block_means(x, nb):
    x = np.asarray(x, float)
    k = len(x) // nb * nb
    return x[:k].reshape(nb, -1, *x.shape[1:]).mean(1)


def sigma_block(x, nb):
    b = block_means(x, nb)
    return float(b.std(ddof=1) / np.sqrt(nb))


def err_model(x, nb, tauB_over_D):
    """(mean, sigma, info) with sigma = max{sigma_block(nb), sigma_naive sqrt(2 tau_eff)}; info carries the parts, the
    inflation sigma/sigma_block, the half-split pull and a block-count scan."""
    x = np.asarray(x, float)
    n = len(x)
    sb = sigma_block(x, nb)
    sn = float(x.std(ddof=1) / np.sqrt(n))
    tx = tau_int(x)
    te = max(tx, tauB_over_D)
    s = max(sb, sn * np.sqrt(2 * te))
    h1, h2 = x[: n // 2].mean(), x[n // 2 :].mean()
    nh = max(2, nb // 2)
    hs = (h1 - h2) / np.sqrt(sigma_block(x[: n // 2], nh) ** 2 + sigma_block(x[n // 2 :], nh) ** 2 + 1e-300)
    scan = {k: sigma_block(x, k) for k in (5, 10, 20) if n >= 2 * k}
    info = dict(
        sigma_block=sb,
        sigma_naive=sn,
        tau_x=tx,
        tau_eff=te,
        infl=float(s / sb) if sb > 0 else 1.0,
        half_pull=float(hs),
        scan=scan,
    )
    return float(x.mean()), float(s), info


def jackknife(func, series, nb, infl=1.0):
    """Delete-one-block jackknife of func(*block-mean-reduced arrays): (value on the full means, error x infl)."""
    bms = [block_means(s, nb) for s in series]
    full = func(*[b.mean(0) for b in bms])
    jk = np.array([func(*[(b.sum(0) - b[i]) / (nb - 1) for b in bms]) for i in range(nb)], float)
    err = float(np.sqrt((nb - 1) / nb * np.sum((jk - jk.mean(0)) ** 2, axis=0)))
    return float(full), err * infl


def jackknife_fit(entries, fitfun):
    """Joint delete-one-block jackknife over several chains. entries: name -> dict(blocks=(nb, ...) block means,
    infl, reduce=callable(mean array) -> value); fitfun(values dict) -> dict of floats. Each chain's blocks are deleted
    in turn (the others at their full means); the variances add, each x that chain's infl^2."""
    full = {n: e["reduce"](e["blocks"].mean(0)) for n, e in entries.items()}
    out = fitfun(full)
    var = {k: 0.0 for k in out}
    for n, e in entries.items():
        B = e["blocks"]
        nb = len(B)
        vals = []
        for b in range(nb):
            v = dict(full)
            v[n] = e["reduce"]((B.sum(0) - B[b]) / (nb - 1))
            vals.append(fitfun(v))
        for k in out:
            arr = np.array([x[k] for x in vals], float)
            if np.all(np.isfinite(arr)):
                var[k] += (nb - 1) / nb * np.sum((arr - arr.mean()) ** 2) * e["infl"] ** 2
            else:
                var[k] = float("nan")
    return out, {k: float(np.sqrt(var[k])) for k in out}


# ------------------------------------------------------------ effective masses
def cosh_mass_point(C, t, Lt, hi=20.0):
    """The cosh effective mass at t: m solving C(t)/C(t+1) = cosh[m(t - Lt/2)]/cosh[m(t + 1 - Lt/2)] in (0, hi]
    (nan if none)."""
    r = C[t] / C[t + 1]
    a, b = t - Lt / 2, t + 1 - Lt / 2
    f = lambda m: np.cosh(m * a) / np.cosh(m * b) - r
    try:
        return brentq(f, 1e-6, hi)
    except ValueError:
        return float("nan")


def cosh_fit(C, sig, Lt, ts=(3, 4, 5, 6)):
    """Two-parameter fit C(t) = A cosh[m (t - Lt/2)] over ts with diagonal weights; returns (m, A, chi2)."""
    C = np.asarray(C, float)
    sig = np.asarray(sig, float)
    ts = np.asarray(ts)

    def chi2_of(m):
        g = np.cosh(m * (ts - Lt / 2))
        w = 1 / sig[ts] ** 2
        A = np.sum(w * g * C[ts]) / np.sum(w * g * g)
        return np.sum(w * (C[ts] - A * g) ** 2), A

    res = minimize_scalar(lambda m: chi2_of(m)[0], bounds=(1e-4, 5.0), method="bounded")
    chi2, A = chi2_of(res.x)
    return float(res.x), float(A), float(chi2)


def exponent(x6, e6, x8, e8, V6, V8):
    """d ln x / d ln V on a volume pair, with its error (independent chains)."""
    if x6 <= 0 or x8 <= 0:
        return float("nan"), float("nan")
    e = np.log(x8 / x6) / np.log(V8 / V6)
    return float(e), float(np.sqrt((e6 / x6) ** 2 + (e8 / x8) ** 2) / np.log(V8 / V6))


def to_json(o):
    """JSON-ready copy (numpy scalars and arrays converted)."""
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [to_json(v) for v in o]
    if isinstance(o, np.floating | float):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ------------------------------------------------------------ stored configurations
def stored_configs(name):
    """(arrays, chain meta) of data/configs/<name>.npz: fields (n, 3V), the chain's row index and trajectory of each
    configuration, and the chain's stored CL_t, CR_t, phi_f of these configurations."""
    from ..data import CONFIGS

    z = np.load(CONFIGS / f"{name}.npz", allow_pickle=False)
    info = json.loads(str(z["meta"]))
    return {k: z[k] for k in z.files if k != "meta"}, info["chain"]


def remeasure(chain, fields, h):
    """Dense N1 measurement (taste channels, no six-fermion composites) of stored configurations with the chain's
    action and the source strength h in the operator (h = the chain's h re-measures; another h is a quench at fixed
    sigma). Returns one bilinears dict per configuration."""
    from ..action import build_model
    from ..lattice import Lattice, parse_bc
    from ..measure import N1Measure

    L, Lt = chain["L"], chain.get("Lt") or chain["L"]
    lat = Lattice((L, L, L, Lt), bc=parse_bc(chain["bc_str"]))
    model = build_model(lat, chain["y"], chain["kappa"], chain["lam"], h=h, pattern=chain["h10_pattern"])
    meas = N1Measure(model, exact=True, seed=0, six_fermion=False)
    return [meas.bilinears(np.asarray(f, float)) for f in fields]
