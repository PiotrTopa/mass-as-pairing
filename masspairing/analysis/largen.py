"""Large-N (Gaussian) theory of the staggered sigma channel of model N1 with the taste-chiral mass h C_chi, its fits to
the stored P_c data, and the pentest of those fits (K8.6, K8.8).

For a staggered background sigma(x) = eps(x) sigma0 e_3 the doublet operator is
D = (K - hC) x 1 + i y sigma0 eps x tau_3, and (eps anticommutes with K, commutes with the same-parity C)
    D^dag D = A + y^2 sigma0^2 + 2 i h y sigma0 (C eps) x tau_3,     A = -(K - hC)^2.
With the weight det D_c = det(D^dag D)^(1/2) the Gaussian curvature of the staggered mode is
    S_eff''(0) = V [1 - y^2 Pi(h)],   Pi(h) = (2/V) [Tr A^-1 + 2 h^2 Tr(A^-1 C eps A^-1 C eps)]
(kappa and the sigma quartic do not enter); the large-N critical coupling is y_MF(h) = Pi(h)^(-1/2) and the relative
distance from criticality at fixed y is delta_L(h) = 1 - Pi_L(h)/Pi_L(0) on the box L.

Fits (weighted least squares on the P_c reference rows of stage 1b, h in {0, 0.5, 1, 1.5, 2, 3}):
  RPA   1/S(pi) = s0 + b delta_L(h)       (2 parameters, the shape has none)
  POW   1/S(pi) = s0 + b h^kappa          QUAD  1/S(pi) = s0 + b h^2
  MF_m  m^2 = m0^2 + a delta_L            NU_m  m^2 = m0^2 + a delta_L^p      POW_m  m^2 = m0^2 + (c h^kappa)^2
The production-operator path (``curvature``, ``ystar``) uses the chain runner's own DoubletOperator.dense and is
independent of the closed form.
"""

from __future__ import annotations

import functools
import json

import numpy as np
from scipy.optimize import least_squares

from ..data import DERIVED
from ..lattice import Lattice, kinetic_matrix
from ..patterns import all_planes_mass, chiral_mass
from . import stage1b as B

HS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
BOX = {"L6": (6, 6), "L8": (8, 8), "L6x12": (6, 12)}
LATS = ("L8", "L6", "L6x12")
MF_PROD = DERIVED / "largen" / "mf_prod.json"


# ------------------------------------------------------------ closed form
def operators(L, Lt=None, bc=(-1, -1, -1, -1)):
    lat = Lattice((L, L, L, Lt or L), bc=bc)
    K = kinetic_matrix(lat).toarray()
    C = chiral_mass(lat).toarray()
    eps = np.asarray(lat.eps_np, float).reshape(-1)
    return lat, K, C, eps


def bubble(K, C, eps, h):
    """Pi(h) = (2/V)[Tr A^-1 + 2h^2 T] and its two parts."""
    V = K.shape[0]
    M = K - h * C
    Ai = np.linalg.inv(M.T @ M)  # A = -M^2 = M^T M (M real antisymmetric)
    tA = float(np.trace(Ai))
    if h == 0.0:
        T = 0.0
    else:
        X = Ai @ (C * eps[None, :])
        T = float(np.sum(X * X.T))
    return 2.0 / V * (tA + 2.0 * h * h * T), 2.0 / V * tA, 4.0 * h * h * T / V


def half_logdet_DdD(K, C, eps, h, y, s0):
    """1/2 log det(D^dag D) of the explicit 2V-dimensional doublet operator for sigma = eps sigma0 e_3."""
    V = K.shape[0]
    M = K - h * C
    D = np.zeros((2 * V, 2 * V), complex)
    D[:V, :V] = M + 1j * y * s0 * np.diag(eps)
    D[V:, V:] = M - 1j * y * s0 * np.diag(eps)
    _, ld = np.linalg.slogdet(D.conj().T @ D)
    return 0.5 * ld


def finite_difference_check(L=4, h=0.7, y=1.3, d=1e-3):
    """(second sigma0-difference of 1/2 log det(D^dag D), V y^2 Pi(h))."""
    _, K, C, eps = operators(L)
    V = K.shape[0]

    def f(s):
        return half_logdet_DdD(K, C, eps, h, y, s)

    fd = (f(d) - 2 * f(0.0) + f(-d)) / d**2
    return fd, V * y * y * bubble(K, C, eps, h)[0]


@functools.cache
def _ops(L, Lt, pattern):
    l, K, C, eps = operators(L, Lt)
    if pattern == "full":
        C = all_planes_mass(l).toarray()
    return K, C, eps


@functools.cache
def pi(L, Lt, h, pattern="chiral"):
    """Pi(h) on the all-antiperiodic L^3 x Lt box (cached) for the taste-chiral mass, or "full": a mass on both
    doublets (all six planes)."""
    K, C, eps = _ops(L, Lt, pattern)
    return bubble(K, C, eps, h)[0]


def deltas(lat, hs=HS, pattern="chiral"):
    """(delta(h), Pi(h)) on the box of ``lat``."""
    L, Lt = BOX[lat]
    P = np.array([pi(L, Lt, h, pattern) for h in hs])
    return 1.0 - P / P[0], P


# ------------------------------------------------------------ fits to the stored P_c rows
def data_rows(rows=None):
    """{(lat, h): {"Spi", "m"}} of the P_c reference series (stored h in {0, 0.5, 1, 2}, new h in {1.5, 3})."""
    if rows is None:
        rows = B.read_all(B.free_tables(B.FREE_1B))
    return {(lat, h): dict(Spi=B.ref(rows, lat, h)["Spi"], m=B.ref(rows, lat, h)["m"]) for lat in LATS for h in HS}


def wls(fun, p0, x, yv, s):
    res = least_squares(lambda p: (fun(p, x) - yv) / s, p0, method="lm", max_nfev=20000)
    chi2 = float(np.sum(res.fun**2))
    dof = len(yv) - len(p0)
    J = res.jac
    try:
        err = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan)
    return dict(
        p=res.x.tolist(), err=err.tolist(), chi2=chi2, dof=dof, chi2_dof=chi2 / dof if dof > 0 else float("nan")
    )


def analyse(drows, delta=None):
    """Per lattice: delta, Pi, 1/S(pi), m and the six fits. ``delta(lat) -> (delta, Pi)`` overrides the shape."""
    out = {}
    for lat in LATS:
        dl, P = deltas(lat) if delta is None else delta(lat)
        h = np.array(HS)
        S = np.array([drows[(lat, hh)]["Spi"][0] for hh in HS])
        Se = np.array([drows[(lat, hh)]["Spi"][1] for hh in HS])
        m = np.array([drows[(lat, hh)]["m"][0] for hh in HS])
        me = np.array([drows[(lat, hh)]["m"][1] for hh in HS])
        iS, iSe = 1 / S, Se / S**2
        m2, m2e = m**2, 2 * m * me
        r = dict(h=HS, delta=dl.tolist(), Pi=P.tolist(), invS=iS.tolist(), invS_err=iSe.tolist())
        r["RPA"] = wls(lambda p, x: p[0] + p[1] * x, [iS[0], 1.0], dl, iS, iSe)
        r["POW"] = wls(lambda p, x: p[0] + p[1] * np.abs(x) ** p[2], [iS[0], 0.1, 1.0], h, iS, iSe)
        r["QUAD"] = wls(lambda p, x: p[0] + p[1] * x**2, [iS[0], 0.05], h, iS, iSe)
        r["MF_m"] = wls(lambda p, x: p[0] + p[1] * x, [m2[0], 1.0], dl, m2, m2e)
        with np.errstate(divide="ignore"):
            r["NU_m"] = wls(lambda p, x: p[0] + p[1] * np.abs(x) ** p[2], [m2[0], 1.0, 1.0], dl, m2, m2e)
        r["POW_m"] = wls(lambda p, x: p[0] + (p[1] * np.abs(x) ** p[2]) ** 2, [m2[0], 0.3, 0.7], h, m2, m2e)
        dS = iS - iS[0]
        r["local_exp_invS"] = [
            float(np.log(dS[i + 1] / dS[i]) / np.log(h[i + 1] / h[i])) if dS[i] > 0 and dS[i + 1] > 0 else None
            for i in range(1, 5)
        ]
        r["local_exp_delta"] = [float(np.log(dl[i + 1] / dl[i]) / np.log(h[i + 1] / h[i])) for i in range(1, 5)]
        out[lat] = r
    return out


def permuted(lat):
    """Sabotage: delta_L with permuted h labels."""
    d, P = deltas(lat)
    idx = [0, 3, 1, 5, 2, 4]
    return d[idx], P[idx]


def both_doublets(lat):
    return deltas(lat, pattern="full")


# ------------------------------------------------------------ pentest of the fits (K8.8)
GENERIC = ("Pade", "tanh2", "Gauss")


def pentest(drows):
    """(A) shape competitors for 1/S(pi) per lattice with AIC = chi^2 + 2k; (B) cross-lattice hold-out (shape and
    amplitude from one lattice, the target's h = 0 point as the only anchor; chi^2 over its 5 points h > 0); (C) the
    light pair-channel mass in the Gaussian sigma-mode form m^2 = m0^2 + a delta_L and a Pade form."""
    hs = np.array(HS)
    D = {}
    for lat in LATS:
        dl, _ = deltas(lat)
        S = np.array([drows[(lat, h)]["Spi"][0] for h in HS])
        Se = np.array([drows[(lat, h)]["Spi"][1] for h in HS])
        m = np.array([drows[(lat, h)]["m"][0] for h in HS])
        me = np.array([drows[(lat, h)]["m"][1] for h in HS])
        D[lat] = dict(dl=dl, df=deltas(lat, pattern="full")[0], y=1 / S, e=Se / S**2, m2=m**2, m2e=2 * m * me)
    shapes = {
        "RPA_deltaL": (lambda p, d: p[0] + p[1] * d["dl"], 2, [0.05, 1.0]),
        "both_doublets": (lambda p, d: p[0] + p[1] * d["df"], 2, [0.05, 0.5]),
        "Pade": (lambda p, d: p[0] + p[1] * hs**2 / (1 + hs**2 / p[2] ** 2), 3, [0.05, 0.2, 1.5]),
        "tanh2": (lambda p, d: p[0] + p[1] * np.tanh(hs / p[2]) ** 2, 3, [0.05, 0.5, 1.5]),
        "Gauss": (lambda p, d: p[0] + p[1] * (1 - np.exp(-(hs**2) / p[2] ** 2)), 3, [0.05, 0.5, 1.5]),
        "POW": (lambda p, d: p[0] + p[1] * hs ** p[2], 3, [0.05, 0.1, 1.0]),
    }
    out = {"A": {}, "B": {}, "C": {}}
    for lat, d in D.items():
        out["A"][lat] = {}
        for nm, (f, k, p0) in shapes.items():
            r = wls(lambda p, x, f=f, d=d: f(p, d), p0, None, d["y"], d["e"])
            r["AIC"] = r["chi2"] + 2 * k
            out["A"][lat][nm] = r
    for src in D:
        for tgt in D:
            if src == tgt:
                continue
            res = {}
            for nm in ("RPA_deltaL", "both_doublets", "Pade", "tanh2"):
                f = shapes[nm][0]
                rs = out["A"][src][nm]["p"]
                p = list(rs)
                p[0] = D[tgt]["y"][0] - (f(rs, D[tgt])[0] - rs[0])  # anchor: the target's h = 0 value
                pred = f(p, D[tgt])
                res[nm] = dict(chi2=float(np.sum(((pred - D[tgt]["y"]) / D[tgt]["e"])[1:] ** 2)))
            out["B"][f"{src}->{tgt}"] = res
    mshapes = {
        "Gauss_sigma_mode": (lambda p, d: p[0] + p[1] * d["dl"], [0.1, 1.0]),
        "Pade_m": (lambda p, d: p[0] + p[1] * hs**2 / (1 + hs**2 / p[2] ** 2), [0.1, 0.2, 1.5]),
    }
    for lat, d in D.items():
        out["C"][lat] = {
            nm: wls(lambda p, x, f=f, d=d: f(p, d), p0, None, d["m2"], d["m2e"]) for nm, (f, p0) in mshapes.items()
        }
    return out


def pade_permuted(drows):
    """Sabotage: the Pade crossover on permuted h labels, chi^2/dof per lattice."""
    hs = np.array(HS)
    idx = [0, 3, 1, 5, 2, 4]
    bad = []
    for lat in LATS:
        S = np.array([drows[(lat, h)]["Spi"][0] for h in HS])
        Se = np.array([drows[(lat, h)]["Spi"][1] for h in HS])
        y, e = (1 / S)[idx], (Se / S**2)[idx]
        r = wls(lambda p, x: p[0] + p[1] * hs**2 / (1 + hs**2 / p[2] ** 2), [0.05, 0.2, 1.5], None, y, e)
        bad.append(r["chi2"] / r["dof"])
    return bad


# ------------------------------------------------------------ production operator (K8.6)
def model(L, y, h):
    """Model N1 as the chain runner builds it: 4^4-type box L^4, all-antiperiodic, kappa = -0.01, lambda = 1, no link
    fields, pattern chiral."""
    from ..action import build_model

    lat = Lattice((L, L, L, L), bc=(-1, -1, -1, -1))
    return build_model(lat, y, -0.01, 1.0, h=h, pattern="chiral")


def fields_for(m, s0, a=2, stag=True):
    lat = m.lat
    V = lat.V
    eps = np.asarray(lat.eps_np, float).reshape(-1) if stag else np.ones(V)
    sig = np.zeros((3, V))
    sig[a] = eps * s0
    return m.pack(sig.reshape((3,) + tuple(lat.shape)), np.zeros(m.D.n_fields))


def logabsdet(m, f):
    _, ld = np.linalg.slogdet(m.D.dense(f))
    return float(ld)


def curvature(L, h, a=2, d=1e-3):
    """Second sigma0-difference of -(1/V) log|det D| at y = 1 (= -Pi_prod(h))."""
    m = model(L, 1.0, h)
    V = m.lat.V
    w = [-logabsdet(m, fields_for(m, s, a=a)) / V for s in (-d, 0.0, d)]
    return (w[0] - 2 * w[1] + w[2]) / d**2


def ystar(L, h, stag=True, bmass=1.0):
    """Full mean-field transition: the smallest y (grid 0.02) at which min_{sigma0 > 0} v(sigma0) < v(0),
    v = (bmass/2) sigma0^2 + sigma0^4/4 - (1/V) log|det D|; F(u) = -(1/V) log|det D(y sigma0 = u)| tabulated on 481
    points in [0, 12] and interpolated in u^2. Returns (y*, sigma0 at onset)."""
    m = model(L, 1.0, h)
    V = m.lat.V

    def F(u):
        return -logabsdet(m, fields_for(m, u, stag=stag)) / V

    F0 = F(0.0)
    ug = np.linspace(0.0, 12.0, 481)
    Fg = np.array([F(u) for u in ug]) - F0
    sg = np.linspace(0.005, 2.0, 400)
    bg = 0.5 * bmass * sg**2 + 0.25 * sg**4
    for y in np.round(np.arange(1.0, 6.01, 0.02), 3):
        u = y * sg
        okm = u <= ug[-1]
        dv = bg[okm] + np.interp(u[okm] ** 2, ug**2, Fg)
        if dv.size and dv.min() < -1e-9:
            return float(y), float(sg[okm][np.argmin(dv)])
    return None, None


def recorded_ystar():
    """The recorded full mean-field y*(h) of the stored scan (data/derived/largen/mf_prod.json):
    {key: {h: (y*, sigma0 at onset)}}."""
    z = json.loads(MF_PROD.read_text())["ystar"]
    return {k: {float(h): (v["ystar"], v["sigma_at"]) for h, v in d.items()} for k, d in z.items()}
