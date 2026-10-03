"""Order of the transition at P_c = (2.41, -0.01) from the epsilon-model chains at L = 6 and 8.

The estimators and the decision rule of claims K7.1 (pilot set, 39 chains) and K7.2 (pilot set + the 8^4 three-start
set and the inner pair at L = 8). The criteria were fixed before the data were read; the code below implements them
as applied.

Per chain (``chain_stats``), after a cut of the first ``CUT`` trajectories: m = |Sigma_stag|, m^2, S(pi), S(pi + p_min),
sigma^2 (energy-like density), O4 (every second trajectory), R4 = <m^4>/<m^2>^2, xi_2,stag / L, the energy cumulant
V_e = 1 - <e^4>/(3 <e^2>^2) of e = sigma^2. Errors: blocked jackknife with block = ceil(2 tau_int(m^2)) (tau_int of
O4 for O4). A chain is *scored* iff its tau_int window converged, tau_int <= N/50 and its two halves agree within
3 sigma on m^2 and sigma^2.

Signals: H1 hysteresis (pulls between the afm / cold / hot starts at P_c), H2 double peak (kernel density of m,
sigma^2, O4 at the Scott bandwidth and 0.75 x Scott) with the V_e support ratio, H3 the finite-difference slope
exponent theta = ln(s_8/s_6)/ln(8/6) of xi_2/L (reported only: its first-order control shows no power on this grid),
H4 gamma/nu_eff = ln(S(pi)_8/S(pi)_6)/ln(8/6), H5 the Binder drift R4(8) - R4(6). Controls: injected jump, mixture /
AR(1) series, first-order and smooth finite-size-scaling mocks on the real grid with the real errors.

Records are dicts with the chain parameters (``tag`` = archive path below results/xi_scan, ``dir``, L, y, kappa, start,
seed, nsteps, ntraj_done, src, config_shape) and the float arrays ``SERIES``.
"""

from __future__ import annotations

import numpy as np

from ..data import derived, load_chain
from .stats import jackknife, tau_int

CUT = 100  # trajectories dropped at the start of every chain
YC = 2.41  # P_c on the kappa = -0.01 line
SERIES = ("Sigma_stag_abs", "S_pi", "S_pi_pmin", "sigma2", "O4", "ferm_flag", "accepted")
OBS = ("m2", "sig2", "O4", "Spi", "Spm")
POOLED_KEYS = ("mabs", "m2", "R4", "xiL", "Spi", "Spm", "sig2", "Ve", "O4", "VeO4")
CONTINUOUS = "CONTINUOUS (consistent with, L ≤ 8)"
SHARPENED = "CONTINUOUS (consistent with, L ≤ 8), sharpened"


# ---------------------------------------------------------------------------------------------- per-chain statistics
def xiL_f(L):
    """xi_2,stag / L from the columns (S(pi), S(pi + p_min))."""
    return lambda a: np.sqrt(max(np.mean(a[:, 0]) / np.mean(a[:, 1]) - 1.0, 0.0) / (4 * np.sin(np.pi / L) ** 2)) / L


def binder_e(a):
    """Energy cumulant V_e = 1 - <e^4>/(3 <e^2>^2) (2/3 for a delta peak)."""
    return 1.0 - np.mean(a**4) / (3 * np.mean(a**2) ** 2)


def chain_stats(r, cut=CUT):
    """Every number of one chain after the cut, with its scoring flags."""
    L = r["L"]
    mabs = r["Sigma_stag_abs"][cut:]
    m2 = mabs**2
    Spi, Spm, sig2 = r["S_pi"][cut:], r["S_pi_pmin"][cut:], r["sigma2"][cut:]
    acc = r["accepted"]
    n = len(m2)
    tau, _, ok = tau_int(m2)
    bl = max(1, int(np.ceil(2 * tau)))
    # fermion series: measured every 2nd trajectory; the cut drops the measurements inside the first `cut` trajectories
    nf_cut = int(r["ferm_flag"][:cut].sum())
    O4 = r["O4"][nf_cut:]
    tau_o, _, _ = tau_int(O4)
    bl_o = max(1, int(np.ceil(2 * tau_o)))
    st = dict(
        tag=r["tag"],
        L=L,
        y=r["y"],
        kappa=r["kappa"],
        start=r["start"],
        seed=r["seed"],
        n=n,
        nsteps=r["nsteps"],
        acc=float(acc.mean()),
        tau=float(tau),
        tau_ok=bool(ok),
        blen=bl,
    )
    st["mabs"], st["e_mabs"] = map(float, jackknife(mabs, np.mean, blen=bl))
    st["m2"], st["e_m2"] = map(float, jackknife(m2, np.mean, blen=bl))
    st["R4"], st["e_R4"] = map(float, jackknife(m2, lambda a: np.mean(a**2) / np.mean(a) ** 2, blen=bl))
    st["xiL"], st["e_xiL"] = map(float, jackknife(np.stack([Spi, Spm], 1), xiL_f(L), blen=bl))
    st["Spi"], st["e_Spi"] = map(float, jackknife(Spi, np.mean, blen=bl))
    st["Spm"], st["e_Spm"] = map(float, jackknife(Spm, np.mean, blen=bl))
    st["sig2"], st["e_sig2"] = map(float, jackknife(sig2, np.mean, blen=bl))
    st["Ve"], st["e_Ve"] = map(float, jackknife(sig2, binder_e, blen=bl))
    st["O4"], st["e_O4"] = map(float, jackknife(O4, np.mean, blen=bl_o))
    st["VeO4"], st["e_VeO4"] = map(float, jackknife(O4, binder_e, blen=bl_o))
    st["tau_O4"] = float(tau_o)
    # acceptance in 100-trajectory windows over the whole chain (before the cut too)
    win = [float(acc[i : i + 100].mean()) for i in range(0, len(acc), 100)]
    st["acc_windows"] = win
    st["low_acc"] = bool(min(win) < 0.5)
    # stationarity: first vs second half after the cut, blocked errors of each half
    h = n // 2
    pulls = {}
    for name, x in (("m2", m2), ("sig2", sig2)):
        a, ea = jackknife(x[:h], np.mean, blen=bl)
        b, eb = jackknife(x[h:], np.mean, blen=bl)
        pulls[name] = float(abs(a - b) / np.hypot(ea, eb)) if np.isfinite(ea) and np.isfinite(eb) else np.nan
    st["half_pulls"] = pulls
    st["stationary"] = bool(all(p < 3 for p in pulls.values() if np.isfinite(p)))
    st["few_samples"] = bool(tau > n / 50)
    st["scored"] = bool(ok and not st["few_samples"] and st["stationary"])
    flags = []
    if not ok:
        flags.append("tau-window")
    if st["few_samples"]:
        flags.append("tau>N/50")
    if not st["stationary"]:
        flags.append("non-stationary")
    if st["low_acc"]:
        flags.append("low-acc")
    st["flags"] = flags
    return st


def _inverse_variance(s, stats):
    for k in POOLED_KEYS:
        w = np.array([1 / x["e_" + k] ** 2 for x in stats])
        v = np.array([x[k] for x in stats])
        s[k] = float((w * v).sum() / w.sum())
        s["e_" + k] = float(1 / np.sqrt(w.sum()))


def combine(stats):
    """Inverse-variance combination of the replicas at one (kappa, L, y); replica pull on m^2 of the first two."""
    if len(stats) == 1:
        s = dict(stats[0])
        s["replicas"] = 1
        s["replica_pull"] = 0.0
        return s
    s = dict(stats[0])
    s["replicas"] = len(stats)
    s["tag"] = " + ".join(x["tag"] for x in stats)
    s["n"] = sum(x["n"] for x in stats)
    s["tau"] = max(x["tau"] for x in stats)
    s["scored"] = all(x["scored"] for x in stats)
    s["flags"] = sorted(set(sum((x["flags"] for x in stats), [])))
    a, b = stats[0], stats[1]
    s["replica_pull"] = float(abs(a["m2"] - b["m2"]) / np.hypot(a["e_m2"], b["e_m2"]))
    if s["replica_pull"] > 3:
        s["scored"] = False
        s["flags"].append("replica-inconsistent")
    _inverse_variance(s, stats)
    return s


def pool(stats):
    """Inverse-variance pooling of >= 2 replicas at P_c with the largest pairwise m^2 pull."""
    s = dict(stats[0])
    s["replicas"] = len(stats)
    s["tag"] = " + ".join(x["tag"] for x in stats)
    s["n"] = sum(x["n"] for x in stats)
    s["tau"] = max(x["tau"] for x in stats)
    s["scored"] = all(x["scored"] for x in stats)
    s["flags"] = sorted(set(sum((x["flags"] for x in stats), [])))
    s["replica_pull"] = float(max(pull(a, b, "m2") for i, a in enumerate(stats) for b in stats[i + 1 :]))
    if s["replica_pull"] > 3:
        s["scored"] = False
        s["flags"].append("replica-inconsistent")
    _inverse_variance(s, stats)
    return s


def pull(A, B, k):
    """|A_k - B_k| in units of the combined error."""
    return float(abs(A[k] - B[k]) / np.hypot(A["e_" + k], B["e_" + k]))


# ---------------------------------------------------------------------------------------------- double peaks (H2)
def kde_modes(x, factor=1.0):
    """Local maxima of a Gaussian kernel density (Scott bandwidth x factor) on a 400-point grid.

    Bimodal: two maxima, the lower >= 20 % of the higher, the valley between them <= 50 % of the lower.
    Returns (bimodal, detail).
    """
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std()
    if sd == 0 or n < 50:
        return False, dict(peaks=0)
    h = factor * sd * n ** (-1 / 5)
    g = np.linspace(x.min() - 2 * h, x.max() + 2 * h, 400)
    dens = np.exp(-0.5 * ((g[:, None] - x[None, :]) / h) ** 2).sum(1) / (n * h * np.sqrt(2 * np.pi))
    pk = [
        i
        for i in range(1, len(g) - 1)
        if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] > 0.02 * dens.max()
    ]
    if len(pk) < 2:
        return False, dict(peaks=len(pk))
    i1, i2 = sorted(sorted(pk, key=lambda i: -dens[i])[:2])
    hi, lo = max(dens[i1], dens[i2]), min(dens[i1], dens[i2])
    valley = dens[i1 : i2 + 1].min()
    bim = lo >= 0.2 * hi and valley <= 0.5 * lo
    return bool(bim), dict(
        peaks=len(pk),
        lo_over_hi=float(lo / hi),
        valley_over_lo=float(valley / lo),
        at=(float(g[i1]), float(g[i2])),
    )


def bimodal(x):
    """Bimodal at both bandwidths (Scott and 0.75 x Scott)."""
    b1, d1 = kde_modes(x, 1.0)
    b2, d2 = kde_modes(x, 0.75)
    return bool(b1 and b2), dict(scott=d1, scott075=d2)


def chain_bimodality(r):
    """{m, sig2, O4: (bimodal, detail)} of one chain after the cut."""
    out = {}
    for name, x in (
        ("m", r["Sigma_stag_abs"][CUT:]),
        ("sig2", r["sigma2"][CUT:]),
        ("O4", r["O4"][int(r["ferm_flag"][:CUT].sum()) :]),
    ):
        out[name] = bimodal(x)
    return out


# ---------------------------------------------------------------------------------------------- exponents (H3, H4)
def slope_exponent(P6, P8, side, delta):
    """theta = ln(s_8/s_6)/ln(8/6) from finite differences of xi_2/L between y* = YC and y* -/+ delta."""
    yo = round(YC + (delta if side == "SMG" else -delta), 3)
    if yo not in P6 or yo not in P8 or YC not in P6 or YC not in P8:
        return dict(
            side=side, delta=delta, y=yo, s6=np.nan, s8=np.nan, theta=np.nan, e_theta=np.nan, valid=False, missing=True
        )
    a, c = P6[YC], P6[yo]
    b, e = P8[YC], P8[yo]
    s6 = (a["xiL"] - c["xiL"]) / delta
    s8 = (b["xiL"] - e["xiL"]) / delta
    es6 = np.hypot(a["e_xiL"], c["e_xiL"]) / delta
    es8 = np.hypot(b["e_xiL"], e["e_xiL"]) / delta
    if s6 <= 0 or s8 <= 0:
        return dict(side=side, delta=delta, y=yo, s6=s6, s8=s8, theta=np.nan, e_theta=np.nan, valid=False)
    th = np.log(s8 / s6) / np.log(8 / 6)
    eth = np.hypot(es8 / s8, es6 / s6) / np.log(8 / 6)
    return dict(
        side=side,
        delta=delta,
        y=yo,
        s6=float(s6),
        es6=float(es6),
        s8=float(s8),
        es8=float(es8),
        theta=float(th),
        e_theta=float(eth),
        valid=bool(all(P[yy]["scored"] for P in (P6, P8) for yy in (YC, yo))),
    )


def ratio_exponent(v6, e6, v8, e8):
    """ln(v_8/v_6)/ln(8/6) with the error from the two relative errors."""
    ex = np.log(v8 / v6) / np.log(8 / 6)
    ee = np.hypot(e8 / v8, e6 / v6) / np.log(8 / 6)
    return float(ex), float(ee)


def classify_theta(th, eth):
    if not np.isfinite(th):
        return "undecided"
    if th - 2 * eth >= 2.5:
        return "first-order sign"
    if th + 2 * eth < 4:
        return "continuous-consistent"
    return "undecided"


def classify_gamma(g, eg):
    if not np.isfinite(g):
        return "undecided"
    if g - 2 * eg >= 3.5:
        return "first-order sign"
    if g + 2 * eg < 4:
        return "continuous-consistent"
    return "undecided"


def h3_letter(th):
    """H3 by the letter of the rule from the four slope exponents {SYM, SMG} x {0.042, 0.127}."""
    cls = {k: classify_theta(t["theta"], t["e_theta"]) if t["valid"] else "undecided" for k, t in th.items()}
    sides = {s: [cls[k] for k in cls if k.startswith(s)] for s in ("SYM", "SMG")}
    fo_both = all(any(c == "first-order sign" for c in v) for v in sides.values())
    cc_one = any(c == "continuous-consistent" for v in sides.values() for c in v)
    contradict = any(c == "first-order sign" for v in sides.values() for c in v)
    return "first-order sign" if fo_both else ("continuous-consistent" if (cc_one and not contradict) else "undecided")


# ---------------------------------------------------------------------------------------------- controls
def ar1(n, tau, rng, mean=0.0, sd=1.0):
    """Unit-variance AR(1) series with autocorrelation time tau."""
    phi = np.exp(-1 / max(tau, 0.51))
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.normal()
    return mean + sd * x


def injected_jump(A, B):
    """Pull after shifting A's m^2 by 6 x the combined error away from B (must exceed 4)."""
    se = np.hypot(A["e_m2"], B["e_m2"])
    shift = 6 * se * np.sign(A["m2"] - B["m2"] or 1.0)
    p = float(abs(A["m2"] + shift - B["m2"]) / se)
    return p, bool(p > 4)


def mixture_control(tau, n=900, nseeds=40):
    """Bimodality rates: AR(1) (false positives), equal-weight AR(1) mixtures 5 and 4 standard deviations apart."""
    det5 = det4 = fp = 0
    for seed in range(nseeds):
        r2 = np.random.default_rng(1000 + seed)
        fp += bimodal(ar1(n, tau, r2))[0]
        lab = r2.random(n) < 0.5
        det5 += bimodal(ar1(n, tau, r2) + 5.0 * lab)[0]
        det4 += bimodal(ar1(n, tau, r2) + 4.0 * lab)[0]
    return fp, det5, det4


def mock_grid(P6, P8, nu, gn, peak_growth):
    """Finite-size-scaling mock on the real grid with the real relative errors.

    xi_2/L = 0.4 (L/6)^peak_growth F(x), S(pi) = 3 L^(gamma/nu) F(x), F = 1/(1 + (x/3)^2), x = (y - y_c) L^(1/nu).
    Continuous: nu = 0.61, gamma/nu = 3.08, no peak growth. First order: a pseudo-transition of width 1/V (nu = 1/4),
    S(pi) ~ V, xi_2/L peak ~ L.
    """
    Pm = {6: {}, 8: {}}
    for L in (6, 8):
        src = P6 if L == 6 else P8
        for y in sorted(set(P6) & set(P8)):
            x = (y - YC) * L ** (1 / nu)
            F = 1 / (1 + (x / 3.0) ** 2)
            xiL = 0.4 * (L / 6) ** peak_growth * F
            Spi = 3.0 * L**gn * F
            Pm[L][y] = dict(
                xiL=xiL,
                e_xiL=xiL * src[y]["e_xiL"] / src[y]["xiL"],
                Spi=Spi,
                e_Spi=Spi * src[y]["e_Spi"] / src[y]["Spi"],
                scored=True,
            )
    th = {f"{s}-{d}": slope_exponent(Pm[6], Pm[8], s, d) for s in ("SYM", "SMG") for d in (0.042, 0.127)}
    g, eg = ratio_exponent(Pm[6][YC]["Spi"], Pm[6][YC]["e_Spi"], Pm[8][YC]["Spi"], Pm[8][YC]["e_Spi"])
    return th, g, eg


def _mock_pilot(P6, P8, nu, gn, peak_growth):
    th, g, eg = mock_grid(P6, P8, nu, gn, peak_growth)
    th = [t for t in th.values() if not t.get("missing")]
    return dict(
        theta=[(round(t["theta"], 2), round(t["e_theta"], 2)) if np.isfinite(t["theta"]) else None for t in th],
        theta_class=[classify_theta(t["theta"], t["e_theta"]) for t in th],
        gamma=(round(g, 2), round(eg, 2)),
        gamma_class=classify_gamma(g, eg),
    )


def _mock_sharpened(P6, P8, nu, gn, peak_growth):
    th, g, eg = mock_grid(P6, P8, nu, gn, peak_growth)
    res = dict(
        theta={
            k: (round(t["theta"], 2), round(t["e_theta"], 2), classify_theta(t["theta"], t["e_theta"]))
            for k, t in th.items()
        },
        drift={s: round(th[f"{s}-0.127"]["theta"] - th[f"{s}-0.042"]["theta"], 2) for s in ("SYM", "SMG")},
        gamma=(round(g, 2), round(eg, 2)),
        gamma_class=classify_gamma(g, eg),
    )
    res["inner_class"] = {s: res["theta"][f"{s}-0.042"][2] for s in ("SYM", "SMG")}
    return res


# ---------------------------------------------------------------------------------------------- phase labels
def phase_label(s):
    """AFM if <|Sigma_stag|> >= 0.15; else SMG if <O4> >= 0.04; else SYM."""
    if s["mabs"] >= 0.15:
        return "AFM"
    return "SMG" if s["O4"] >= 0.04 else "SYM"


def _label_entry(r, s, lab, near, conflict, bim):
    return dict(
        file=r["src"],
        config_array="sigma_final",
        config_shape=r["config_shape"],
        L=s["L"],
        y=s["y"],
        kappa=s["kappa"],
        start=s["start"],
        seed=s["seed"],
        ntraj=r["ntraj_done"],
        nsteps=s["nsteps"],
        acceptance=round(s["acc"], 3),
        phase=lab,
        near_critical=bool(near),
        Sigma_stag_abs=[round(s["mabs"], 5), round(s["e_mabs"], 5)],
        O4=[round(s["O4"], 5), round(s["e_O4"], 5)],
        xiL=[round(s["xiL"], 4), round(s["e_xiL"], 4)],
        tau_int_m2=round(s["tau"], 2),
        scored=s["scored"],
        flags=s["flags"],
        conflict=conflict,
        bimodal={n: b for n, (b, _) in bim.items()},
    )


def _grid(stats, hyst_prefixes):
    grid = {}
    for s in stats:
        if any(s["tag"].startswith(p) for p in hyst_prefixes):
            continue
        grid.setdefault((s["kappa"], s["L"], round(s["y"], 3)), []).append(s)
    return grid


def sort_records(recs):
    return sorted(recs, key=lambda r: (r["kappa"], r["L"], r["y"], r["dir"]))


# ---------------------------------------------------------------------------------------------- pilot set (K7.1)
def analyse_pilot(recs):
    """The 39-chain pilot set: FSS table, H1 at 6^4, H2, H3, H4, H5, crossing, controls, verdict, phase labels.

    Returns dict(verdict=..., fss=[combined grid rows], labels=[...], stats=[per chain], comb={key: row}).
    """
    recs = sort_records(recs)
    stats = [chain_stats(r) for r in recs]
    comb = {k: combine(v) for k, v in _grid(stats, ("F_hyst",)).items()}
    # (6,8) differences per kappa line
    lines = {}
    for kap in (-0.04, -0.01, 0.02):
        P6 = {k[2]: v for k, v in comb.items() if k[0] == kap and k[1] == 6}
        P8 = {k[2]: v for k, v in comb.items() if k[0] == kap and k[1] == 8}
        ys = sorted(set(P6) & set(P8))
        d = {}
        for y in ys:
            a, b = P6[y], P8[y]
            dx = b["xiL"] - a["xiL"]
            ex = np.hypot(a["e_xiL"], b["e_xiL"])
            g, eg = ratio_exponent(a["Spi"], a["e_Spi"], b["Spi"], b["e_Spi"])
            d[y] = dict(
                dxiL=float(dx),
                e_dxiL=float(ex),
                sig=float(dx / ex),
                gamma_nu=g,
                e_gamma_nu=eg,
                eta=2 - g,
                e_eta=eg,
                dR4=float(b["R4"] - a["R4"]),
                e_dR4=float(np.hypot(a["e_R4"], b["e_R4"])),
                ratio_mabs=float(a["mabs"] / b["mabs"]),
                scored=bool(a["scored"] and b["scored"]),
            )
        cross = []
        for i in range(len(ys)):
            if abs(d[ys[i]]["sig"]) < 1:
                cross.append(("zero-within-1σ", ys[i]))
            if i + 1 < len(ys) and np.sign(d[ys[i]]["dxiL"]) != np.sign(d[ys[i + 1]]["dxiL"]):
                cross.append(("sign-change", (ys[i], ys[i + 1])))
        vmin = {}
        for L, PP in ((6, P6), (8, P8)):
            cand = [(2 / 3 - v["Ve"], v["e_Ve"], y) for y, v in PP.items() if v["scored"]]
            vmin[L] = max(cand) if cand else (np.nan, np.nan, None)
        lines[kap] = dict(ys=ys, diff=d, crossings=cross, Ve_min=vmin, P6=P6, P8=P8)

    # H1 hysteresis at 6^4
    hyst = {s["start"]: s for s in stats if s["tag"].startswith("F_hyst")}
    hyst_pulls = {}
    for a, b in (("afm", "cold"), ("afm", "hot"), ("cold", "hot")):
        for k in OBS:
            hyst_pulls[f"{a}-{b}:{k}"] = pull(hyst[a], hyst[b], k)
    hmax = max(hyst_pulls.values())
    hyst_scored = all(s["scored"] for s in hyst.values())
    H1 = "positive" if (hmax > 4 and hyst_scored) else ("negative" if hmax < 3 else "undecided")

    # H2 double peaks on every chain; a first-order sign needs the same (kappa, y) bimodal on both volumes
    bim = {r["tag"]: chain_bimodality(r) for r in recs}
    pos = [(t, n) for t, v in bim.items() for n, (b, _) in v.items() if b]
    st_of = {s["tag"]: s for s in stats}
    byky = {}
    for r, s in zip(recs, stats, strict=True):
        for n, (b, _) in bim[r["tag"]].items():
            if b and s["scored"] and not r["tag"].startswith("F_hyst"):
                byky.setdefault((s["kappa"], round(s["y"], 3), n), set()).add(s["L"])
    H2_pos = [k for k, Ls in byky.items() if Ls >= {6, 8}]
    near = [
        (t, n)
        for (t, n) in pos
        if st_of[t]["kappa"] == -0.01 and abs(st_of[t]["y"] - YC) <= 0.13 and st_of[t]["scored"]
    ]
    L8_scored_bim = [(t, n) for (t, n) in pos if st_of[t]["L"] == 8 and st_of[t]["scored"]]
    H2 = "positive" if H2_pos else ("negative" if not (L8_scored_bim or near) else "indicative")
    ve6, ve8 = lines[-0.01]["Ve_min"][6], lines[-0.01]["Ve_min"][8]
    ve_ratio = ve8[0] / ve6[0]
    e_ve_ratio = ve_ratio * np.hypot(ve8[1] / ve8[0], ve6[1] / ve6[0])
    H2s = (
        "first-order sign"
        if ve_ratio - 2 * e_ve_ratio > 0.5
        else ("continuous-consistent" if ve_ratio + 2 * e_ve_ratio < 0.5 else "undecided")
    )

    # H3 slope exponents on the kappa_c line
    P6, P8 = lines[-0.01]["P6"], lines[-0.01]["P8"]
    th = {f"{s}-{d}": slope_exponent(P6, P8, s, d) for s in ("SYM", "SMG") for d in (0.042, 0.127)}
    H3 = h3_letter(th)
    drift = {
        s: (th[f"{s}-0.127"]["theta"] - th[f"{s}-0.042"]["theta"]) if not th[f"{s}-0.042"].get("missing") else np.nan
        for s in ("SYM", "SMG")
    }

    # H4 at y = 2.41 and from the grid maxima; H5; the crossing
    g, eg = ratio_exponent(P6[YC]["Spi"], P6[YC]["e_Spi"], P8[YC]["Spi"], P8[YC]["e_Spi"])
    m6 = max(P6.values(), key=lambda v: v["Spi"])
    m8 = max(P8.values(), key=lambda v: v["Spi"])
    gm, egm = ratio_exponent(m6["Spi"], m6["e_Spi"], m8["Spi"], m8["e_Spi"])
    H4 = classify_gamma(g, eg)
    if classify_gamma(gm, egm) != H4:
        H4 = "undecided"
    dR = lines[-0.01]["diff"][YC]
    H5 = (
        "first-order sign"
        if (dR["dR4"] / dR["e_dR4"] > 3 and H2 == "positive")
        else ("continuous-consistent" if abs(dR["dR4"]) < 2 * dR["e_dR4"] else "undecided")
    )
    crossing = abs(dR["sig"]) <= 3 and lines[-0.01]["diff"][2.537]["dxiL"] < 0

    # controls
    ctl = {}
    ctl["H1_injected_pull"], ctl["H1_injected_flagged"] = injected_jump(hyst["afm"], hyst["cold"])
    fp, det5, det4 = mixture_control(P6[YC]["tau"])
    ctl["H2_ar1_false_positive"] = f"{fp}/40"
    ctl["H2_mixture5_detected"] = f"{det5}/40"
    ctl["H2_mixture4_detected"] = f"{det4}/40"
    ctl["H2_pass"] = bool(fp <= 2 and det5 >= 38)
    fo = ctl["H3H4_firstorder_mock"] = _mock_pilot(P6, P8, 0.25, 4.0, 1.0)
    co = ctl["H3H4_continuous_mock"] = _mock_pilot(P6, P8, 0.61, 3.08, 0.0)
    ctl["H4_firstorder_mock_not_continuous"] = bool(fo["gamma_class"] != "continuous-consistent")
    ctl["H4_firstorder_mock_flagged"] = bool(fo["gamma_class"] == "first-order sign")
    ctl["H4_continuous_mock_clean"] = bool(co["gamma_class"] == "continuous-consistent")
    ctl["H3_firstorder_mock_not_continuous"] = bool(not any(c == "continuous-consistent" for c in fo["theta_class"]))
    ctl["H3_continuous_mock_clean"] = bool(not any(c == "first-order sign" for c in co["theta_class"]))
    ctl["H3_has_power"] = bool(ctl["H3_firstorder_mock_not_continuous"])
    ctl["prereg_all_pass"] = bool(
        ctl["H1_injected_flagged"]
        and ctl["H2_pass"]
        and ctl["H4_firstorder_mock_flagged"]
        and ctl["H4_continuous_mock_clean"]
        and ctl["H3_firstorder_mock_not_continuous"]
        and ctl["H3_continuous_mock_clean"]
    )
    ctl["exclusion_pass"] = bool(
        ctl["H1_injected_flagged"]
        and ctl["H2_pass"]
        and ctl["H4_firstorder_mock_not_continuous"]
        and ctl["H4_continuous_mock_clean"]
        and ctl["H3_continuous_mock_clean"]
    )

    # decision
    pc_scored = P6[YC]["scored"] and P8[YC]["scored"] and hyst_scored
    first = (
        (H1 == "positive" or H2 == "positive")
        and (H3 == "first-order sign" or H4 == "first-order sign")
        and not any(x == "continuous-consistent" for x in ((H3 if ctl["H3_has_power"] else "undecided"), H4))
        and ctl["prereg_all_pass"]
    )
    cont = (
        H1 == "negative"
        and H2 == "negative"
        and (H3 == "continuous-consistent" or not ctl["H3_has_power"])
        and H4 == "continuous-consistent"
        and H5 != "first-order sign"
        and pc_scored
        and ctl["exclusion_pass"]
    )
    cont_letter = (
        H1 == "negative"
        and H2 == "negative"
        and H3 == "continuous-consistent"
        and H4 == "continuous-consistent"
        and H5 != "first-order sign"
        and pc_scored
    )
    verdict = "FIRST ORDER" if first else (CONTINUOUS if cont else "UNDECIDED")

    # phase labels (the volume ratio overrides a contradicted label; P_c chains are "critical")
    raw = {}
    for r, s in zip(recs, stats, strict=True):
        lab = phase_label(s)
        key = (s["kappa"], round(s["y"], 3))
        other = [v for k, v in comb.items() if (k[0], k[2]) == key and k[1] != s["L"]]
        conflict = None
        if other:
            o = other[0]
            ratio = (s["mabs"] / o["mabs"]) if s["L"] == 6 else (o["mabs"] / s["mabs"])
            if (abs(ratio - 1) < 0.15 and lab != "AFM") or (ratio > 1.5 and lab == "AFM"):
                conflict = f"volume ratio {ratio:.2f} contradicts {lab}"
                lab = "AFM" if abs(ratio - 1) < 0.15 else ("SMG" if s["O4"] >= 0.04 else "SYM")
        raw[r["tag"]] = (lab, conflict)
    labels = []
    for r, s in zip(recs, stats, strict=True):
        lab, conflict = raw[r["tag"]]
        line = sorted(
            {(round(st["y"], 3), raw[st["tag"]][0]) for st in stats if st["kappa"] == s["kappa"] and st["L"] == s["L"]}
        )
        ys_line = [y for y, _ in line]
        i = ys_line.index(round(s["y"], 3))
        neighbours = [line[j][1] for j in (i - 1, i + 1) if 0 <= j < len(line)]
        nearc = any(nb != lab for nb in neighbours)
        if s["kappa"] == -0.01:
            nearc = abs(s["y"] - YC) <= 0.05
            if abs(s["y"] - YC) < 1e-6:
                lab = "critical"
        labels.append(_label_entry(r, s, lab, nearc, conflict, bim[r["tag"]]))

    v = dict(
        verdict=verdict,
        verdict_prereg_letter=("CONTINUOUS" if cont_letter else ("FIRST ORDER" if first else "UNDECIDED")),
        crossing_persists=crossing,
        H1=H1,
        H1_pulls=hyst_pulls,
        H1_max=hmax,
        H2=H2,
        H2_Ve=H2s,
        H2_Ve_ratio=[ve_ratio, e_ve_ratio],
        H2_bimodal=pos,
        H3=H3,
        H3_theta=th,
        H3_drift=drift,
        H4=H4,
        H4_gamma_nu=[g, eg],
        H4_eta=[2 - g, eg],
        H4_gridmax=[gm, egm],
        H5=H5,
        H5_dR4=[dR["dR4"], dR["e_dR4"]],
        pc_scored=pc_scored,
        controls=ctl,
        lines={
            k: dict(ys=x["ys"], diff=x["diff"], crossings=x["crossings"], Ve_min=x["Ve_min"]) for k, x in lines.items()
        },
        hyst={k: {kk: vv for kk, vv in x.items() if kk != "acc_windows"} for k, x in hyst.items()},
    )
    fss = [{kk: vv for kk, vv in x.items() if kk != "acc_windows"} for _, x in sorted(comb.items())]
    return dict(verdict=v, fss=fss, labels=labels, stats=stats, comb=comb)


# ---------------------------------------------------------------------------------------------- pilot + 8^4 set (K7.2)
def last_half_means(r, s):
    """Blocked means of the last half (after the cut) of the five observables of one chain."""
    bl = s["blen"]
    out = {}
    m2 = r["Sigma_stag_abs"][CUT:] ** 2
    h = len(m2) // 2
    for k, x in (("m2", m2), ("sig2", r["sigma2"][CUT:]), ("Spi", r["S_pi"][CUT:]), ("Spm", r["S_pi_pmin"][CUT:])):
        out[k] = tuple(map(float, jackknife(x[h:], np.mean, blen=bl)))
    O4 = r["O4"][int(r["ferm_flag"][:CUT].sum()) :]
    h4 = len(O4) // 2
    out["O4"] = tuple(map(float, jackknife(O4[h4:], np.mean, blen=max(1, int(np.ceil(2 * s["tau_O4"]))))))
    return out


def analyse_sharpened(pilot_recs, new_recs, pilot):
    """Pilot set + the five L = 8 chains on the kappa_c line (three starts at P_c, inner pair y = 2.368 / 2.452).

    ``pilot`` is the output of ``analyse_pilot(pilot_recs)`` (its H1 at 6^4 and H2 enter the combined signals).
    Returns dict(verdict=..., fss=[kappa_c rows], labels=[new chains], stats=[per chain]).
    """
    v051 = pilot["verdict"]
    new_tags = {r["tag"] for r in new_recs}
    recs = sort_records(list(pilot_recs) + list(new_recs))
    stats = [chain_stats(r) for r in recs]
    by_tag = {s["tag"]: (r, s) for r, s in zip(recs, stats, strict=True)}

    comb = {k: combine(v) for k, v in _grid(stats, ("F_hyst", "F12_hyst")).items()}
    P6 = {k[2]: v for k, v in comb.items() if k[0] == -0.01 and k[1] == 6}
    P8 = {k[2]: v for k, v in comb.items() if k[0] == -0.01 and k[1] == 8}

    # H1 at 8^4 (with the last-half condition of a positive), combined with H1 at 6^4
    hyst8 = {s["start"]: s for s in stats if s["tag"].startswith("F12_hyst")}
    hyst6 = {s["start"]: s for s in stats if s["tag"].startswith("F_hyst")}
    lh = {st: last_half_means(*by_tag[hyst8[st]["tag"]]) for st in hyst8}
    pulls8, lh_pulls = {}, {}
    for a, b in (("afm", "cold"), ("afm", "hot"), ("cold", "hot")):
        for k in OBS:
            pulls8[f"{a}-{b}:{k}"] = pull(hyst8[a], hyst8[b], k)
            (ma, ea), (mb, eb) = lh[a][k], lh[b][k]
            lh_pulls[f"{a}-{b}:{k}"] = float(abs(ma - mb) / np.hypot(ea, eb))
    hmax8 = max(pulls8.values())
    hkey8 = max(pulls8, key=pulls8.get)
    scored8 = all(s["scored"] for s in hyst8.values())
    H1_8 = "positive" if (hmax8 > 4 and scored8 and lh_pulls[hkey8] > 3) else ("negative" if hmax8 < 3 else "undecided")
    H1_6, hmax6 = v051["H1"], v051["H1_max"]
    H1 = "positive" if "positive" in (H1_6, H1_8) else ("negative" if H1_6 == H1_8 == "negative" else "undecided")
    en8 = max(v for k, v in pulls8.items() if k.endswith(":sig2") or k.endswith(":O4"))

    # the four 8^4 P_c chains: the pilot chain (GPU path without the fused kernels) + the three starts (fused path)
    old8 = by_tag["F_L8_k-0.01/L8_y2.41_k-0.01.npz"][1]
    four = dict(old=old8, **hyst8)
    names = list(four)
    rp = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for k in OBS:
                rp[f"{a}-{b}:{k}"] = pull(four[a], four[b], k)
    old_vs_new = {k: v for k, v in rp.items() if k.startswith("old-")}
    new_only = {k: v for k, v in rp.items() if not k.startswith("old-")}
    old_outlier = max(old_vs_new.values()) > 4 and max(new_only.values()) < 3

    # pooled P_c at both volumes
    l6_pc = by_grid(comb, stats, 6) + [hyst6[s] for s in ("afm", "cold", "hot")]
    l8_pc = [old8] + [hyst8[s] for s in ("afm", "cold", "hot")]
    pool6, pool8 = pool(l6_pc), pool(l8_pc)
    P6p, P8p = dict(P6), dict(P8)
    P6p[YC], P8p[YC] = pool6, pool8

    # H2 on the new chains, V_e support on the complete kappa_c grid
    bim = {r["tag"]: chain_bimodality(r) for r in recs if r["tag"] in new_tags}
    pos_new = [(t, n) for t, v in bim.items() for n, (b, _) in v.items() if b]
    pos_new_scored = [(t, n) for t, n in pos_new if by_tag[t][1]["scored"]]
    H2 = (
        "negative"
        if (v051["H2"] == "negative" and not pos_new_scored)
        else ("indicative" if pos_new_scored else v051["H2"])
    )
    vmin = {}
    for L, PP in ((6, P6), (8, P8)):
        vmin[L] = max((2 / 3 - v["Ve"], v["e_Ve"], y) for y, v in PP.items() if v["scored"])
    ve_ratio = vmin[8][0] / vmin[6][0]
    e_ve_ratio = ve_ratio * np.hypot(vmin[8][1] / vmin[8][0], vmin[6][1] / vmin[6][0])
    H2s = (
        "first-order sign"
        if ve_ratio - 2 * e_ve_ratio > 0.5
        else ("continuous-consistent" if ve_ratio + 2 * e_ve_ratio < 0.5 else "undecided")
    )

    # H3 with the inner pair; its weight from the first-order mock at the inner pair
    th = {f"{s}-{d}": slope_exponent(P6, P8, s, d) for s in ("SYM", "SMG") for d in (0.042, 0.127)}
    thp = {f"{s}-{d}": slope_exponent(P6p, P8p, s, d) for s in ("SYM", "SMG") for d in (0.042, 0.127)}
    drift = {
        s: (
            th[f"{s}-0.127"]["theta"] - th[f"{s}-0.042"]["theta"],
            np.hypot(th[f"{s}-0.127"]["e_theta"], th[f"{s}-0.042"]["e_theta"]),
        )
        for s in ("SYM", "SMG")
    }
    ymax = {L: max(PP.values(), key=lambda v: v["xiL"])["y"] for L, PP in ((6, P6), (8, P8))}
    fo = _mock_sharpened(P6p, P8p, 0.25, 4.0, 1.0)
    co = _mock_sharpened(P6p, P8p, 0.61, 3.08, 0.0)
    H3_power = not any(c == "continuous-consistent" for c in fo["inner_class"].values())
    H3 = h3_letter(th)

    # H4, H5, the crossing: unpooled and pooled must agree
    def h4(P6x, P8x):
        g, eg = ratio_exponent(P6x[YC]["Spi"], P6x[YC]["e_Spi"], P8x[YC]["Spi"], P8x[YC]["e_Spi"])
        m6 = max(P6x.values(), key=lambda v: v["Spi"])
        m8 = max(P8x.values(), key=lambda v: v["Spi"])
        gm, egm = ratio_exponent(m6["Spi"], m6["e_Spi"], m8["Spi"], m8["e_Spi"])
        c, cm = classify_gamma(g, eg), classify_gamma(gm, egm)
        return dict(
            g=g,
            eg=eg,
            gm=gm,
            egm=egm,
            y6=m6["y"],
            y8=m8["y"],
            cls=c if c == cm else "undecided",
            cls_fixed=c,
            cls_max=cm,
        )

    H4u, H4p = h4(P6, P8), h4(P6p, P8p)
    H4 = H4p["cls"] if H4p["cls"] == H4u["cls"] else "undecided"

    def h5(P6x, P8x):
        d = P8x[YC]["R4"] - P6x[YC]["R4"]
        e = np.hypot(P6x[YC]["e_R4"], P8x[YC]["e_R4"])
        cls = (
            "first-order sign"
            if (d / e > 3 and H2 == "positive")
            else ("continuous-consistent" if abs(d) < 2 * e else "undecided")
        )
        return d, e, cls

    H5u, H5p = h5(P6, P8), h5(P6p, P8p)
    H5 = H5p[2] if H5p[2] == H5u[2] else "undecided"

    def crossing(P6x, P8x):
        dx = P8x[YC]["xiL"] - P6x[YC]["xiL"]
        ex = np.hypot(P6x[YC]["e_xiL"], P8x[YC]["e_xiL"])
        d537 = P8x[2.537]["xiL"] - P6x[2.537]["xiL"]
        return dx / ex, d537, (abs(dx / ex) <= 3 and d537 < 0)

    Ku, Kp = crossing(P6, P8), crossing(P6p, P8p)
    crossing_persists = Ku[2] and Kp[2]

    # controls at 8^4
    ctl = {}
    ctl["H1_injected_pull"], ctl["H1_injected_flagged"] = injected_jump(hyst8["afm"], hyst8["cold"])
    tau8 = max(s["tau"] for s in hyst8.values())
    fp, det5, det4 = mixture_control(tau8)
    ctl["H2_tau_used"] = float(tau8)
    ctl["H2_ar1_false_positive"] = f"{fp}/40"
    ctl["H2_mixture5_detected"] = f"{det5}/40"
    ctl["H2_mixture4_detected"] = f"{det4}/40"
    ctl["H2_pass"] = bool(fp <= 2 and det5 >= 38)
    ctl["H3H4_firstorder_mock"] = fo
    ctl["H3H4_continuous_mock"] = co
    ctl["H4_firstorder_mock_not_continuous"] = bool(fo["gamma_class"] != "continuous-consistent")
    ctl["H4_firstorder_mock_flagged"] = bool(fo["gamma_class"] == "first-order sign")
    ctl["H4_continuous_mock_clean"] = bool(co["gamma_class"] == "continuous-consistent")
    ctl["H3_continuous_mock_clean"] = bool(not any(c[2] == "first-order sign" for c in co["theta"].values()))
    ctl["H3_has_power"] = bool(H3_power)
    ctl["exclusion_pass"] = bool(
        ctl["H1_injected_flagged"]
        and ctl["H2_pass"]
        and ctl["H4_firstorder_mock_not_continuous"]
        and ctl["H4_continuous_mock_clean"]
        and ctl["H3_continuous_mock_clean"]
    )
    ctl["prereg_all_pass"] = bool(ctl["exclusion_pass"] and ctl["H4_firstorder_mock_flagged"] and ctl["H3_has_power"])

    # decision
    pc_scored = all(s["scored"] for s in l6_pc + l8_pc)
    H3v = H3 if H3_power else "undecided"
    first = (
        (H1 == "positive" or H2 == "positive")
        and (H4 == "first-order sign" or H3v == "first-order sign")
        and not any(x == "continuous-consistent" for x in (H3v, H4))
        and ctl["prereg_all_pass"]
    )
    cont = (
        H1 == "negative"
        and H2 == "negative"
        and (H3v == "continuous-consistent" or not H3_power)
        and H4 == "continuous-consistent"
        and H5 != "first-order sign"
        and pc_scored
        and ctl["exclusion_pass"]
    )
    verdict = "FIRST ORDER" if first else (SHARPENED if cont else "UNDECIDED")

    # phase labels of the new chains
    labels = []
    for r, s in zip(recs, stats, strict=True):
        if r["tag"] not in new_tags:
            continue
        lab = phase_label(s)
        conflict = None
        other = [v for k, v in comb.items() if (k[0], k[2]) == (s["kappa"], round(s["y"], 3)) and k[1] == 6]
        if other and abs(s["y"] - YC) > 1e-6:
            ratio = other[0]["mabs"] / s["mabs"]
            if (abs(ratio - 1) < 0.15 and lab != "AFM") or (ratio > 1.5 and lab == "AFM"):
                conflict = f"volume ratio {ratio:.2f} contradicts {lab}"
                lab = "AFM" if abs(ratio - 1) < 0.15 else ("SMG" if s["O4"] >= 0.04 else "SYM")
        if abs(s["y"] - YC) < 1e-6:
            lab = "critical"
        labels.append(_label_entry(r, s, lab, abs(s["y"] - YC) <= 0.05, conflict, bim[r["tag"]]))

    strip = lambda d: {k: v for k, v in d.items() if k != "acc_windows"}
    v = dict(
        verdict=verdict,
        crossing_persists=crossing_persists,
        H1=H1,
        H1_6=H1_6,
        H1_6_max=hmax6,
        H1_8=H1_8,
        H1_8_max=hmax8,
        H1_8_pulls=pulls8,
        H1_8_lasthalf_pulls=lh_pulls,
        H1_8_energy_max=en8,
        replica4_pulls=rp,
        replica4_max=max(rp.values()),
        old_chain_outlier=old_outlier,
        old_vs_new_max=max(old_vs_new.values()),
        new_vs_new_max=max(new_only.values()),
        H2=H2,
        H2_bimodal_new=pos_new,
        H2_Ve=H2s,
        H2_Ve_ratio=[ve_ratio, e_ve_ratio],
        H2_Ve_max={str(L): list(x) for L, x in vmin.items()},
        H3=H3,
        H3_power=H3_power,
        H3_theta=th,
        H3_theta_pooled=thp,
        H3_drift={k: list(x) for k, x in drift.items()},
        xiL_max_y=ymax,
        H4=H4,
        H4_unpooled=H4u,
        H4_pooled=H4p,
        H5=H5,
        H5_unpooled=list(H5u),
        H5_pooled=list(H5p),
        crossing=dict(unpooled=list(Ku), pooled=list(Kp)),
        pc_scored=pc_scored,
        pooled=dict(L6=strip(pool6), L8=strip(pool8)),
        hyst8={k: strip(x) for k, x in hyst8.items()},
        old8=strip(old8),
        controls=ctl,
    )
    fss = [strip(x) for k, x in sorted(comb.items()) if k[0] == -0.01]
    return dict(verdict=v, fss=fss, labels=labels, stats=stats, comb=comb, by_tag=by_tag)


def by_grid(comb, stats, L):
    """The grid (non-start-set) replicas at P_c on the kappa_c line at volume L."""
    return [
        s for s in stats if s["kappa"] == -0.01 and s["L"] == L and round(s["y"], 3) == YC and "hyst" not in s["tag"]
    ]


# ---------------------------------------------------------------------------------------------- data
def load_record(path):
    """One derived chain file (data/derived/k7/t3a) as a record: chain parameters, provenance, float series."""
    s, m = load_chain(path)
    rec = dict(
        tag=m["tag"],
        dir=m["dir"],
        set=m["set"],
        L=int(m["L"]),
        y=float(m["y"]),
        kappa=float(m["kappa"]),
        start=m["start"],
        seed=int(m["seed"]),
        ntraj_done=int(m["ntraj_done"]),
        nsteps=int(m["nsteps"]),
        fast=bool(m.get("fast", False)),
        config_shape=m["config_shape"],
        src=m["source"],
    )
    for k in SERIES:
        rec[k] = np.asarray(s["ts_" + k]).astype(float)
    return rec


def load_records(which):
    """'pilot' (the 39 chains of K7.1) or 'sharpened' (the five L = 8 chains on the kappa_c line added in K7.2)."""
    recs = [load_record(p) for p in sorted(derived("k7", "t3a").glob("*/*.npz"))]
    return [r for r in recs if r["set"] == which]


# ---------------------------------------------------------------------------------------------- output helpers
def jsonable(o):
    """Plain JSON types (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
