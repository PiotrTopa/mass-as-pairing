"""Estimators and verdicts of the K3 claims (the (10,3,1) channel on the sign-free link-field model).

Chain series (``ts_*`` keys of a run, or of a derived file) are analysed per ensemble:

* ``cut`` drops the first k0 trajectories of a stored stream (per-trajectory series directly, per-measurement
  series -- the fermion observables, one entry per True ``ferm_flag`` -- at the number of measurements made in
  those trajectories); ``ensemble`` concatenates several cut streams of one point (independent replicas) and
  analyses the result.
* ``analyse`` gives means with blocked-jackknife errors. Each listed series is summarised with its own block
  length ceil(2 tau_int); the derived quantities (V-scaled susceptibilities, corner structure factors, correlation
  lengths, the exact source derivative) use one common block length ceil(2 max tau_int) over the summarised
  series of the ensemble.
* ``exponent`` is the effective finite-size exponent d ln X / d ln V between two volumes with linear error
  propagation.

The verdict functions encode the criteria of the claims (fixed before the data were read): the volume scaling of the
complete 10-channel correlator (K3.1), the linear response of chi10 to the pure plaquette term (K3.3), the
spontaneous-breaking diagnostic under the same-parity pairing source (K3.4) and the exit of the U(1)_eps-exact
family (K3.5). ``rho_flat`` is the flatness of the zero-momentum fermion correlator on the odd time slices.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from ..data import DERIVED
from .stats import jackknife, summarize, xi2

# per-trajectory series that are summarised (sigma field, link fields)
SCALAR_KEYS = ("sigma2", "Sigma_abs", "Sigma_stag_abs", "S_0", "S_pi", "S_pmin", "S_pi_pmin", "s2") + tuple(
    f"{k}_{j}" for j in (0, 1) for k in ("s_mean", "s2", "Ss_0", "Ss_pi", "Ss_pmin", "Ss_pi_pmin")
)
# per-measurement series that are summarised (fermion observables of the complete-channel measure)
FERMION_KEYS = (
    "phi_abs",
    "phi_stag_abs",
    "phi_sq",
    "phi_stag_sq",
    "O4",
    "dimer_sq_par",
    "dimer_sq_perp",
    "chi10",
    "chi6",
    "E10",
    "E6",
    "phi_src",
    "phi_src_even",
    "phi_src_odd",
    "phi_src_dh",
)
# keys of the chain files the analyses read
ARRAY_KEYS = ("dH", "accepted", "ferm_flag", "chi10_corners", "chi10_pmin", "chi6_corners", "chi6_pmin", "Gf")
NCORNER = 8  # S(p + pi(1,1,1,1)) = -S(p): the first 8 of the 16 corners are independent


def corner_label(k, d=4):
    """'0'/'π' string of corner k (bit d-1-i of k is axis i)."""
    return "".join("π" if (k >> (d - 1 - i)) & 1 else "0" for i in range(d))


# ---- derived chain files ----------------------------------------------------------------------------------------
def derived_path(rel):
    """data/derived/k3/<set>/<file> of an archive chain path (results/xi_scan/<set>/<file>, results/laneW2/c072/n20/
    <file> -> c072_n20, results/laneW2/c073/<start>/<file> -> c073_<start>, results/laneR/c102/<tau>/<file>)."""
    parts = pathlib.Path(rel).parts
    sub = parts[2] if parts[1] == "xi_scan" else "_".join(parts[2:-1])
    return DERIVED / "k3" / sub / parts[-1]


def load(rel):
    """(series, meta) of the derived file of archive chain ``rel``; meta['k0_stored'] trajectories were dropped."""
    d = np.load(derived_path(rel), allow_pickle=True)
    return {k: d[k] for k in d.files if k.startswith("ts_")}, json.loads(str(d["meta"]))


def streams(spec):
    """[(archive path, first kept trajectory of the raw chain), ...] -> ([(series, k0 in the derived file)], metas)."""
    out, metas = [], []
    for rel, k0 in spec:
        s, meta = load(rel)
        assert k0 >= meta["k0_stored"], (rel, k0, meta["k0_stored"])
        out.append((s, k0 - meta["k0_stored"]))
        metas.append(meta)
    return out, metas


def ensemble_of(spec):
    """``ensemble`` of the derived streams of [(archive path, k0)], with 'streams' in raw trajectory numbers
    [(first kept, stored, kept)]."""
    st, metas = streams(spec)
    r = ensemble(st, metas[0])
    r["streams"] = [(k0, m["n_stored"], m["n_stored"] - k0) for (_, k0), m in zip(spec, metas, strict=True)]
    return r


# ---- streams ----------------------------------------------------------------------------------------------------
def n_traj(series):
    return len(series["ts_dH"])


def cut(series, k0):
    """Series of one stream with its first k0 trajectories dropped (see module docstring)."""
    nt = n_traj(series)
    ff = np.asarray(series["ts_ferm_flag"], bool) if "ts_ferm_flag" in series else None
    nf = int(ff.sum()) if ff is not None else None
    nf0 = int(ff[:k0].sum()) if ff is not None else 0
    out = {}
    for k, a in series.items():
        if not k.startswith("ts_"):
            continue
        if len(a) == nt:
            out[k] = a[k0:]
        elif nf is not None and len(a) == nf:
            out[k] = a[nf0:]
        else:
            raise ValueError(f"series {k} has length {len(a)} (trajectories {nt}, measurements {nf})")
    return out


def window(series, k0, k1=None):
    """Trajectories k0 <= t < k1 of a stream (per-measurement series cut accordingly)."""
    nt = n_traj(series)
    k1 = nt if k1 is None else k1
    head = cut(series, k0)
    if k1 >= nt:
        return head
    tail = cut(series, k1)
    return {k: v[: len(v) - len(tail[k])] for k, v in head.items()}


def concat(parts):
    """Concatenation of cut streams over their common keys."""
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    return {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}


def ensemble(streams, meta):
    """Analysis of the concatenation of [(series, k0), ...] (one point, independent streams)."""
    parts = [cut(s, k0) for s, k0 in streams]
    r = analyse(concat(parts), meta)
    r["streams"] = [(k0, n_traj(s), n_traj(s) - k0) for s, k0 in streams]
    return r


def stream_means(streams, key):
    """Plain mean of ts_<key> per cut stream."""
    out = []
    for s, k0 in streams:
        c = cut(s, k0)
        a = c.get("ts_" + key)
        out.append(float(np.mean(a)) if a is not None and len(a) else np.nan)
    return out


# ---- ensemble analysis ------------------------------------------------------------------------------------------
def analyse(series, meta):
    """Means, errors and derived quantities of one ensemble (series dict with ts_ keys, chain meta record)."""
    d = series
    shape = tuple(meta["shape"])
    V = int(np.prod(shape))
    L = shape[0]
    res = dict(shape=shape, V=V, y=meta["y"], g10=meta.get("g10"), g6=meta.get("g6"), ntraj=n_traj(d))
    res["acceptance"] = float(np.mean(d["ts_accepted"]))
    res["exp_mdH"] = float(np.mean(np.exp(-d["ts_dH"])))
    taus = {}
    for k in SCALAR_KEYS + FERMION_KEYS:
        a = d.get("ts_" + k)
        if a is None or len(a) == 0:
            continue
        s = summarize(a)
        taus[k] = s["tau_int"]
        res[k] = s
    blen = max(1, int(np.ceil(2 * max(taus.values())))) if taus else 1
    res["blen"] = blen
    for k in ("phi_sq", "phi_stag_sq", "dimer_sq_par", "dimer_sq_perp"):
        a = d.get("ts_" + k)
        if a is not None and len(a) > 1:
            v, e = jackknife(V * np.asarray(a, float), np.mean, blen=blen)
            res["V_" + k] = dict(mean=float(v), err=float(e))
    if "ts_chi10_corners" in d and len(d["ts_chi10_corners"]) > 1:
        for ch in ("chi10", "chi6"):
            c = np.asarray(d[f"ts_{ch}_corners"], float)[:, :NCORNER]
            v, e = jackknife(c, lambda a: a.mean(0), blen=blen)
            res[ch + "_corners"] = dict(mean=[float(x) for x in v], err=[float(x) for x in e])
            k = int(np.argmax(np.abs(v)))
            res[ch + "_corner_max"] = dict(mean=float(abs(v[k])), err=float(e[k]), corner=k, label=corner_label(k))
            if f"ts_{ch}_pmin" not in d:
                continue
            pm = np.asarray(d[f"ts_{ch}_pmin"], float)[:, :NCORNER]
            for nm, kk in (("xi" + ch[3:], 0), ("xi" + ch[3:] + "_cmax", k)):
                x, xe = xi2(c[:, kk], pm[:, kk], L, blen)
                res[nm] = dict(mean=float(x), err=float(xe), corner=kk)
    if "ts_phi_src_dh" in d and len(d["ts_phi_src_dh"]) > 1:
        # exact h-derivative of <Phi_src>: Wick-connected part + V var(Phi_src) (unbiased only with exact propagators)
        ps = np.asarray(d["ts_phi_src"], float)
        dh = np.asarray(d["ts_phi_src_dh"], float)
        f = lambda a: a[:, 1].mean() + V * (np.mean(a[:, 0] ** 2) - np.mean(a[:, 0]) ** 2)
        v, e = jackknife(np.stack([ps, dh], 1), f, blen=blen)
        res["dphi_src_dh"] = dict(mean=float(v), err=float(e))
    for j in (0, 1):
        if f"ts_Ss_0_{j}" in d and len(d[f"ts_Ss_0_{j}"]) > 1:
            v, e = xi2(d[f"ts_Ss_0_{j}"], d[f"ts_Ss_pmin_{j}"], L, blen)
            res[f"xi2_s_{j}"] = dict(mean=float(v), err=float(e))
            v, e = xi2(d[f"ts_Ss_pi_{j}"], d[f"ts_Ss_pi_pmin_{j}"], L, blen)
            res[f"xi2_s_dimer_{j}"] = dict(mean=float(v), err=float(e))
    if "ts_S_0" in d:
        v, e = xi2(d["ts_S_0"], d["ts_S_pmin"], L, blen)
        res["xi2_ferro"] = dict(mean=float(v), err=float(e))
        v, e = xi2(d["ts_S_pi"], d["ts_S_pi_pmin"], L, blen)
        res["xi2_stag"] = dict(mean=float(v), err=float(e))
    return res


def value(r, key):
    """(mean, error) of an analysed quantity; 'V_*' keys are the V-scaled susceptibilities, 'xi10L' etc. are xi/L."""
    L = r["shape"][0]
    if key.endswith("L") and key[:-1] in r:
        q = r[key[:-1]]
        return q["mean"] / L, q["err"] / L
    q = r[key]
    return q["mean"], q["err"]


def exponent(a, b, La, Lb, d=4):
    """d ln X / d ln V between volumes La^d and Lb^d from (mean, err) pairs (nan if a value is not positive)."""
    (x, ex), (y, ey) = a, b
    lv = np.log((Lb / La) ** d)
    if x <= 0 or y <= 0:
        return np.nan, np.nan
    return float(np.log(y / x) / lv), float(np.hypot(ex / x, ey / y) / lv)


def ratio(num, den):
    """q = a/b with linear error propagation."""
    (a, ea), (b, eb) = num, den
    q = a / b
    return q, abs(q) * np.hypot(ea / a, eb / b)


def freeze_ratio(series, meta, n=50, per_link=False):
    """Link-field equilibration check: min over pairs of <s_j^2>/(2 g_j) over the first n trajectories of a cut
    stream (per-link model: <s^2>/(2 c g), c = 6 on L >= 3 in d = 4)."""
    g10, g6 = meta["g10"], meta["g6"]
    if per_link:
        return float(np.mean(series["ts_s2_0"][:n])) / (12 * g10)
    vals = []
    for j, gj in ((0, g10 - g6), (1, g10 + g6)):
        if gj > 0 and f"ts_s2_{j}" in series:
            vals.append(float(np.mean(series[f"ts_s2_{j}"][:n])) / (2 * gj))
    return min(vals)


def stat(x):
    """(mean, blocked-jackknife error, tau_int, block) of one series with block ceil(2 tau_int)."""
    s = summarize(x)
    return s["mean"], s["err"], s["tau_int"], s["blen"]


def combine_streams(parts):
    """Streams of one chain: concatenated estimate and inverse-variance weighted estimate; the larger error is used.

    Returns (mean of the concatenation, error, tau_int of the concatenation, per-stream stats, chi2 of the streams).
    """
    per = [stat(p) for p in parts]
    mu_c, e_c, t_c, _ = stat(np.concatenate(parts))
    w = np.array([1 / p[1] ** 2 for p in per])
    mus = np.array([p[0] for p in per])
    mu_w = float((w * mus).sum() / w.sum())
    e_w = float(w.sum() ** -0.5)
    chi2 = float((w * (mus - mu_w) ** 2).sum())
    return mu_c, max(e_c, e_w), t_c, per, chi2


# ---- K3.1: volume scaling of the complete chi10 -------------------------------------------------------------------
def scaling_verdict(e68, xi6, xi8):
    """Kill criteria of the volume-scaling test: 'null' if exponent + 2 sigma < 0.5 on (6,8) and xi10/L not rising
    (xi/L(8) - xi/L(6) <= 2 sigma); 'alive' if exponent - 2 sigma >= 0.8 with xi10/L rising; else 'undecided'."""
    e, s = e68
    rising = xi8[0] - xi6[0] > 2 * np.hypot(xi6[1], xi8[1])
    if e + 2 * s < 0.5 and not rising:
        return "null"
    if e - 2 * s >= 0.8 and rising:
        return "alive"
    return "undecided"


# ---- K3.3: linear response to the pure plaquette term -------------------------------------------------------------
def response_verdict(R8, R6, V6=6**4, V8=8**4):
    """Response R(L) = [chi10(wedge) - chi10(completion)]/g, q = R(8)/R(6):
    relevant iff R(8) > 2 sigma and q - 1 > 2 sigma_q; irrelevant iff q <= 1 + 2 sigma_q and q + 2 sigma_q <
    (V8/V6)^0.5 (power against growth with exponent 0.5); undecided otherwise."""
    r8, s8 = R8
    q, sq = ratio(R8, R6)
    if r8 > 2 * s8 and q - 1 > 2 * sq:
        return "relevant"
    if q <= 1 + 2 * sq and q + 2 * sq < (V8 / V6) ** 0.5:
        return "irrelevant"
    return "undecided"


# ---- K3.4: SSB diagnostic under the same-parity pairing source ---------------------------------------------------
def source_point(M, hs, Ls=(4, 6, 8)):
    """Verdict and numbers for one point from the one-point functions M[(L, h)] = (mean, err) of <Phi_src>_h.

    m0(L): intercept of the weighted linear fit over h (two-point extrapolation 2 M(h0) - M(h1) as a cross-check);
    S(L): inverse-variance mean of <Phi_src>/h; e(6,8) = d ln S / d ln V; rho(L) = S(h_min)/S(h_max).
    SSB iff [m0(8) > 2 sigma and m0(8) - m0(6) > 2 sigma] or e(6,8) - 2 sigma > 0; no SSB iff |m0| < 2 sigma at every L,
    e(6,8) <= 2 sigma, e(6,8) + 2 sigma < 0.5 and rho <= 1 + 2 sigma at L = 6, 8; undecided otherwise.
    """
    hh = np.array(hs, float)
    out = dict(m0={}, m0_2pt={}, S={}, rho={})
    for L in Ls:
        mm = np.array([M[(L, h)][0] for h in hs])
        ee = np.array([M[(L, h)][1] for h in hs])
        A = np.stack([np.ones(len(hs)), hh], 1) / ee[:, None]
        cov = np.linalg.inv(A.T @ A)
        c = cov @ A.T @ (mm / ee)
        out["m0"][L] = (float(c[0]), float(np.sqrt(cov[0, 0])), float(c[1]), float(np.sqrt(cov[1, 1])))
        a, b = M[(L, hs[0])], M[(L, hs[1])]
        out["m0_2pt"][L] = (2 * a[0] - b[0], float(np.hypot(2 * a[1], b[1])))
        S, eS = mm / hh, ee / hh
        w = 1 / eS**2
        Sb = float((w * S).sum() / w.sum())
        out["S"][L] = (Sb, float(w.sum() ** -0.5), float((w * (S - Sb) ** 2).sum()))
        r = S[0] / S[-1]
        out["rho"][L] = (float(r), float(abs(r) * np.hypot(eS[0] / S[0], eS[-1] / S[-1])))
    (s6, e6, _), (s8, e8, _) = out["S"][6], out["S"][8]
    lnV = np.log((8 / 6) ** 4)
    e68, se68 = np.log(s8 / s6) / lnV, np.hypot(e8 / s8, e6 / s6) / lnV
    out["e68"] = (float(e68), float(se68))
    m8, sm8 = out["m0"][8][:2]
    m6, sm6 = out["m0"][6][:2]
    ssb = (m8 > 2 * sm8 and m8 - m6 > 2 * np.hypot(sm8, sm6)) or (e68 - 2 * se68 > 0)
    nossb = (
        all(abs(out["m0"][L][0]) < 2 * out["m0"][L][1] for L in Ls)
        and e68 <= 2 * se68
        and e68 + 2 * se68 < 0.5
        and all(out["rho"][L][0] <= 1 + 2 * out["rho"][L][1] for L in (6, 8))
    )
    out["verdict"] = "SSB" if ssb else ("no SSB" if nossb else "undecided")
    return out


# ---- K3.5: fermion correlator flatness ------------------------------------------------------------------------------
def rho_flat(G):
    """Mean of G over the middle odd time slices / mean over the edge odd slices (t = 1, Lt - 1); 1 for a massless
    fermion, about 1/cosh(m (Lt/2 - 1)) with a gap m. G: (..., Lt)."""
    Lt = G.shape[-1]
    odd = list(range(1, Lt, 2))
    edge = [1, Lt - 1] if Lt > 4 else [1]
    mid = [t for t in odd if t not in edge] or [Lt // 2 - 1 if (Lt // 2 - 1) % 2 else Lt // 2 + 1]
    return G[..., mid].mean(-1) / G[..., edge].mean(-1)


def rho_estimate(series, blen):
    """rho_flat of the ensemble-mean correlator ts_Gf with a blocked jackknife error."""
    G = np.asarray(series["ts_Gf"], float)
    v, e = jackknife(G, lambda a: rho_flat(a.mean(0)), blen=blen)
    return float(v), float(e)


# ---- the stored ensembles of the claims ----------------------------------------------------------------------------
XI = "results/xi_scan"
W2 = "results/laneW2"
KAPPA = -0.01
# K3.1: five link-field points (y, g10, g6) at kappa = -0.01, lambda = 1, L = 4, 6, 8
P1_POINTS = [(2.41, 0.05, 0.05), (2.41, 0.1, 0.1), (3.0, 0.1, 0.1), (2.41, 0.05, 0.0), (2.41, 0.02, 0.0)]
# K3.5: the g6 = 0 line and the -edge at y = 0, and the two -edge points at y = 2.41
X_POINTS = [
    (0.0, 0.05, 0.0),
    (0.0, 0.1, 0.0),
    (0.0, 0.05, -0.05),
    (0.0, 0.1, -0.1),
    (2.41, 0.05, -0.05),
    (2.41, 0.1, -0.1),
]
# free theory (y = 0, s = 0): complete chi10 at p = 0 and xi10/L at L = 4, 6, 8 (K3.2)
FREE_CHI10 = {4: (1.67664, 0.1607), 6: (1.08748, 0.0991), 8: (0.93653, 0.0674)}


def chain_name(L, y, g10, g6, suffix=""):
    return f"L{L}_y{y:g}_k{KAPPA:g}_g{g10:g}_g6{g6:g}{suffix}.npz"


def p1_spec(p, L, extra=0):
    """Streams of a K3.1 ensemble. The 8^4 chain at (2.41, 0.02, 0) ran its first 600 trajectories with a poorly
    accepting integrator (dropped) and was split at trajectory 1300 into the original stream and two re-seeded
    replicas (kept from 1400). ``extra`` drops that many more trajectories from every stream."""
    rel = f"{XI}/F8_P1_wedge/{chain_name(L, *p)}"
    if (L, p) == (8, (2.41, 0.02, 0.0)):
        rep = [rel.replace("F8_P1_wedge", "F8_P1_wedge_" + r) for r in ("rA", "rB")]
        return [(rel, 600 + extra)] + [(q, 1400 + extra) for q in rep]
    return [(rel, extra)]


def x_spec(p, L, extra=0):
    return [(f"{XI}/F9_X_y0exit/{chain_name(L, *p)}", extra)]


def p2_rel(L, g6, h):
    return f"{XI}/F8_P2_src/{chain_name(L, 2.41, 0.05, g6, f'_h{h:g}')}"


def p3_rel(model, g, L, stream=""):
    """K3.3 chains: 'compl' (per-link completion, --quads links) or 'wedge' (all squares, g6 = 0) at P_c; L = 8
    streams '' (original seed) or '_rA' (re-seeded continuation); L = 4, 6 the exact-propagator chains."""
    sub = f"F8_P3x_{model}{stream}" if L == 8 else f"F8_P3_{model}"
    return f"{XI}/{sub}/{chain_name(L, 2.41, g, 0.0, '_links' if model == 'compl' else '')}"


def c073_rel(L, y, g, start="main"):
    """K3.6 chains on the +edge (g6 = g10): L = 4 exact propagators, L = 6, 8 from the main runs; 6^4 starts."""
    tag = chain_name(L, y, g, g)[len(f"L{L}") :]
    if L == 4:
        return f"{W2}/c072/n20/L4{tag}" if (y, g) == (2.41, 0.05) else f"{W2}/c072/L4{tag}"
    return f"{W2}/c073/{start}/L{L}{tag}"
