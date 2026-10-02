"""Stage 1b of the N1 test: the order K-N1-S1b applied to the NEW trajectories only (four stage-1 chains extended to
2000 trajectories, new rows at trajectory >= 1000; six new chains at P_c, h in {1.5, 3}, cut at 100), with the stored
stage-1 chains (cut at 100) as fixed reference rows: the y = 3.0 clauses (C2), the anti-seesaw tests T1-T5 (C3), the
integrity table, the stored-vs-new consistency S7', the controls S1'-S6', and the recorded tables E1, E2, E3, E8.

Estimators (fixed before any stage-1b observable was printed):
  m        primary gap: the cosh effective mass of the mean CL_t at the midpoint t = L_t/2 - 1 (delete-one-block
           jackknife x the error-model inflation of the CL_t midpoint series) -- the light pair-channel cosh mass
  fit3     secondary: C(t) = A cosh[m (t - L_t/2)] over t in [L_t/2 - 2, L_t/2], accepted iff chi^2/dof <= 3 and
           m > 2 sigma
  r_L      [m(h)/m(0)] / [m^free(h)/m^free(0)] on 6^4/8^4 ("cosh/free"); raw m(h)/m(0) on 6^3x12 (no free cosh
           solution at the 6^3x12 midpoint)
  g        GL_p0 / free;  e = d ln chi_L / d ln V on (6^4, 8^4), read against the free exponent e_free
  R_peak   [chi_L(0)/chi_L(p_min)] / free (a p = 0 peak sharper than free is > 1)
  S(pi), |Sigma_stag|   the epsilon/sigma channel of the selection's trajectories (20 blocks,
           tau_eff = max(tau_int, tau_B))
Clauses at y = 3.0 (C2, 8^4 rows, stored h = 0 denominators): (a) r_L >= 0.7 - 2 sigma and g <= 0.1 + 2 sigma, not (b);
(b) = (b1) e - e_free - 2 sigma > 0.5 and (b2) R_peak(8^4) - 2 sigma > 1 and sharper than 6^4 and (b3) chi_L/free rising
with V at 2 sigma; (c) r_L + 2 sigma < 0.3 with g >= 0.8 - 2 sigma. Partial outcome "growth relative to free, no
SSB-scale exponent": e - e_free > 3 sigma with (b1) failing.
Anti-seesaw (C3, y = 2.41): T1 monotone rise at 3 sigma (6^3x12, 6^4: h = 1 -> 1.5 and 2 -> 3; 8^4: m(1.5) between m(1)
and m(2) at 2 sigma); T2 m(h)^2 = m0^2 + (c h^kappa)^2 over h in {0.5, 1, 1.5, 2, 3} on 6^3x12 (PASS iff kappa in
[0.7, 1.3] at 2 sigma with chi^2/dof <= 3, UNDECIDED if chi^2/dof > 3); T3 S(pi)(h) - S_inf = a h^-x with x =
kappa (2 - eta), and saturation of S(pi), |Sigma_stag| at h = 3 within 25 % (2 sigma) of the y = 3.0 h = 0 value; T4 no
SSB (e(1.5) - e_free < 0.5 - 2 sigma, R_peak within 2 sigma of 1, S(pi) never rising); T5 V-check.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize_scalar

from .n1stage import (
    N1DIR,
    block_means,
    cosh_fit,
    cosh_mass_point,
    err_model,
    free_lookup,
    free_tables,
    jackknife,
    jackknife_fit,
    open_chain,
    sigma_block,
    tau_int,
)
from .n1stage import exponent as _exponent

CHAINS_1B = {  # tag: (file under data/derived/n1stage/S1b, kind)
    "R1": ("L8/L8_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", "ext"),
    "R3": ("L8/L8_y3_k-0.01_g0_g60_h1_chiral_bcaaaa.npz", "ext"),
    "L5": ("L6/L6_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", "ext"),
    "L6": ("L6/L6_y3_k-0.01_g0_g60_h1_chiral_bcaaaa.npz", "ext"),
    "R2": ("L8/L8_y2.41_k-0.01_g0_g60_h1.5_chiral_bcaaaa.npz", "new"),
    "D-1b-3": ("L8/L8_y2.41_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "new"),
    "L1": ("L6x12/L6x12_y2.41_k-0.01_g0_g60_h1.5_chiral_bcaaaa.npz", "new"),
    "L2": ("L6x12/L6x12_y2.41_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "new"),
    "L3": ("L6/L6_y2.41_k-0.01_g0_g60_h1.5_chiral_bcaaaa.npz", "new"),
    "L4": ("L6/L6_y2.41_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "new"),
}
MD5_DELIVERED = {  # md5 of the chain files as delivered by the stage-1b runs
    "R1": "390ffa3c67e10be806aecdcbfe31c470",
    "R2": "123de3547d9f240d333c57f8f21e8295",
    "R3": "ec35b0889756cf62053f8f111b66d644",
    "D-1b-3": "0bee68b8c148b2c0f762d1343cae60f3",
    "L1": "485492b58db411068744321796fe04ae",
    "L2": "731f04a19dcd45083bf86e63966dd652",
    "L3": "a3f16170c39e5be7d95a8cf91c3d6274",
    "L4": "b042a1ef420f69fd5ec3e07c9c661b22",
    "L5": "077e84bbd0de345f725042c0f41e97fd",
    "L6": "37535675c5c33183a408c5c9843b90dc",
}
FREE_1B = (
    "free_L8_aaaa.json",
    "free_L8_aaaa_h1.5.json",
    "free_L8_aaaa_h3.json",
    "free_L6x12_aaaa.json",
    "free_L6x12_aaaa_h1.5_3.json",
    "free_baselines.json",
    "free_L6_aaaa_h1.5_3.json",
)
LAT = {(8, 8): "L8", (6, 12): "L6x12", (6, 6): "L6"}
FIT_H = (0.5, 1.0, 1.5, 2.0, 3.0)  # T2
T3_H = (0.5, 1.0, 1.5, 2.0)  # T3 (h = 3 is the saturation point)
TWO_MINUS_ETA = (1.58, 0.16)  # the staggered-channel exponent at P_c (K7)
KAPPA_WIN = (0.7, 1.3)
STORED = (0.0, 0.5, 1.0, 2.0)


def coshmass(C, t, Lt):
    return cosh_mass_point(np.asarray(C, float), t, Lt)


def cosh_fit3(C, sig, Lt):
    """The secondary estimator: the cosh fit over the three points t = L_t/2 - 2 ... L_t/2."""
    return cosh_fit(C, sig, Lt, ts=(Lt // 2 - 2, Lt // 2 - 1, Lt // 2))


# ------------------------------------------------------------ reader
def selection(d, sel):
    """(fermion row indices, trajectory mask, lo, hi) for sel in {new, stored, pooled}; duplicates dropped."""
    ntr = len(d["ts_dH"])
    ct = np.asarray(d["ts_cfg_traj"])
    ext = ntr >= 2000
    if sel == "new":
        lo, hi = (1000 if ext else 100), 10**9
    elif sel == "stored":
        lo, hi = 100, (1000 if ext else 10**9)
    else:
        lo, hi = 100, 10**9
    seen, keep = set(), []
    for i, t in enumerate(ct):
        t = int(t)
        if t in seen:
            continue
        seen.add(t)
        if lo <= t < hi:
            keep.append(i)
    idx = np.arange(ntr)
    return np.array(keep, int), (idx >= lo) & (idx < hi), lo, hi


def read_row(path, sel, bl, tag=None):
    d, meta = open_chain(path)
    L, Lt = meta["L"], meta.get("Lt") or meta["L"]
    V = L**3 * Lt
    h = float(meta["h10"])
    keep, tmask, lo, hi = selection(d, sel)
    ct = np.asarray(d["ts_cfg_traj"])[keep]
    n = len(keep)
    nb = 20 if n >= 150 else 10
    D = float(np.median(np.diff(ct))) if n > 1 else float("nan")
    ntr_sel = int(tmask.sum())
    tauB = max(
        tau_int(np.asarray(d["ts_Sigma_stag_abs"], float)[tmask], wmax=ntr_sel // 10),
        tau_int(np.asarray(d["ts_sigma2"], float)[tmask], wmax=ntr_sel // 10),
    )
    tBD = tauB / D
    acc = np.asarray(d["ts_accepted"], bool)[tmask]
    acc50 = [float(acc[i : i + 50].mean()) for i in range(0, len(acc) - 49, 50)]
    row = dict(
        tag=tag,
        file=str(path),
        name=path.stem,
        sel=sel,
        lat=LAT[(L, Lt)],
        y=float(meta["y"]),
        L=L,
        Lt=Lt,
        bc=meta["bc_str"],
        h=h,
        V=V,
        n=n,
        nb=nb,
        cut=[lo, hi],
        cadence=np.unique(np.diff(ct)).tolist(),
        Delta=D,
        ntraj_sel=ntr_sel,
        tau_B=tauB,
        tau_B_over_D=tBD,
        acc_min50=float(min(acc50)) if acc50 else float("nan"),
        git_hash=meta.get("git_hash"),
        comp=meta.get("h10_comp"),
        dual_sign=meta.get("h10_dual_sign"),
        pattern=meta.get("h10_pattern"),
        mts=meta.get("mts"),
        ferm_every=meta.get("ferm_every"),
        cfg_traj_range=[int(ct.min()), int(ct.max())] if n else None,
    )
    CL = np.asarray(d["ts_CL_t"], float)[keep]
    CR = np.asarray(d["ts_CR_t"], float)[keep]
    t = Lt // 2 - 1
    ser = {
        "chi_L": np.asarray(d["ts_chi_L_sum"], float)[keep],
        "chi_L_pmin": np.asarray(d["ts_chi_L_sum_pmin"], float)[keep],
        "GL_p0": np.asarray(d["ts_GL_p0"], float)[keep],
        "GR_p0": np.asarray(d["ts_GR_p0"], float)[keep],
        "chi_R": np.asarray(d["ts_chi_f_sum"], float)[keep],
        "phi_R": np.asarray(d["ts_phi_f"], float)[keep].sum(1),
        "phi_L": np.asarray(d["ts_phi_L"], float)[keep].sum(1),
        "phi_T_R": np.asarray(d["ts_phi_T_R"], float)[keep],
        "phi_T_L": np.asarray(d["ts_phi_T_L"], float)[keep],
        "phi_stag_sq": np.asarray(d["ts_phi_stag_sq"], float)[keep] * V,
        "O4": np.asarray(d["ts_O4"], float)[keep],
        "CL_mid": CL[:, t],
        "CL_ratio": CL[:, t] / CL[:, t + 1],
    }
    stats = {}
    for k, x in ser.items():
        m, s, info = err_model(x, nb, tBD)
        row[k] = (m, s)
        stats[k] = info
    row["stats"] = stats
    infl = stats["CL_mid"]["infl"]
    row["infl_CL"] = infl
    Bm = block_means(CL, nb)
    row["CL_blocks"] = Bm
    BmR = block_means(CR, nb)
    CLm = CL.mean(0)
    CLe = np.array([sigma_block(CL[:, i], nb) for i in range(Lt)]) * infl
    row["CL_t"], row["CL_t_err"], row["CR_t"] = CLm.tolist(), CLe.tolist(), CR.mean(0).tolist()

    def jk_mass(B, tt):
        v, e = jackknife_fit(
            {"c": dict(blocks=B, infl=infl, reduce=lambda c: coshmass(c, tt, Lt))}, lambda v: {"m": v["c"]}
        )
        return v["m"], e["m"]

    mc = {tt: jk_mass(Bm, tt) for tt in range(Lt // 2)}
    row["mcosh_t"] = {str(k): v for k, v in mc.items()}
    row["m"] = mc[t]
    row["mR"] = jk_mass(BmR, t)[0]
    row["mL_eff"] = [float(np.log(CLm[i] / CLm[i + 1])) for i in range(Lt // 2)]
    mf, Af, chi2 = cosh_fit3(CLm, CLe, Lt)
    _, ef = jackknife_fit(
        {"c": dict(blocks=Bm, infl=infl, reduce=lambda c: cosh_fit3(c, CLe, Lt)[0])}, lambda v: {"m": v["c"]}
    )
    row["fit3"] = dict(m=mf, err=ef["m"], A=Af, chi2=chi2, dof=1, accepted=bool(chi2 / 1 <= 3.0 and mf > 2 * ef["m"]))
    row["A"] = jackknife(lambda a, b: a / b, [(CL - CR).sum(1), CL.sum(1)], nb, infl)
    row["peak"] = jackknife(lambda a, b: a / b, [ser["chi_L"], ser["chi_L_pmin"]], nb, stats["chi_L"]["infl"])
    for key, name in (("ts_S_pi", "Spi"), ("ts_Sigma_stag_abs", "Sab")):
        x = np.asarray(d[key], float)[tmask]
        nbB = 20 if len(x) >= 150 else 10
        m, s, info = err_model(x, nbB, tauB)
        row[name] = (m, s)
        stats[name] = info
        row[name + "_blocks"] = block_means(x, nbB)
        row["infl_" + name] = info["infl"]
    if h == 0.0:
        pulls = [jackknife(lambda a: a, [CL[:, tt] - CR[:, tt]], nb, infl) for tt in range(Lt)]
        row["h0_max_pull"] = float(max(abs(p[0] / p[1]) if p[1] > 0 else 0 for p in pulls))
    fr = free_lookup(bl, L, Lt, row["bc"], h)
    if fr is not None:
        row["free"] = dict(
            chi_L=fr["chi_L_sum"],
            chi_L_pmin=fr["chi_L_sum_pmin"],
            GL_p0=fr["GL_p0"],
            GR_p0=fr["GR_p0"],
            chi_R=fr["chi_f_sum"],
            phi_R=float(np.sum(fr["phi_f"])),
            phi_L=float(np.sum(fr["phi_L"])),
            phi_T_R=fr.get("phi_T_R"),
            CL_t=fr["CL_t"],
            m=coshmass(fr["CL_t"], t, Lt),
            mcosh_t={str(tt): coshmass(fr["CL_t"], tt, Lt) for tt in range(Lt // 2)},
            peak=fr["chi_L_sum"] / fr["chi_L_sum_pmin"],
            A=(np.sum(fr["CL_t"]) - np.sum(fr["CR_t"])) / np.sum(fr["CL_t"]),
        )
    return row


def stored_files():
    """The stage-1 chains on 8^4, 6^3x12, 6^4 aaaa; the four extended ones from their extended files (whose stored
    part, trajectories < 1000, is the stage-1 chain)."""
    ext = {f for f, kind in CHAINS_1B.values() if kind == "ext"}
    out = []
    for sub in ("L8", "L6x12", "L6"):
        for p in sorted((N1DIR / "S1" / sub).glob("*.npz")):
            rel = f"{sub}/{p.name}"
            out.append(N1DIR / "S1b" / rel if rel in ext else p)
    return out


def read_all(bl):
    """rows: (lat, y, h, sel) -> row; sel = 'stored' for every stage-1 chain, 'new' (and 'pooled' for the
    extensions) for the ten stage-1b chains."""
    rows = {}
    for f in stored_files():
        r = read_row(f, "stored", bl)
        rows[(r["lat"], r["y"], r["h"], "stored")] = r
    for tag, (f, kind) in CHAINS_1B.items():
        for sel in ("new", "pooled") if kind == "ext" else ("new",):
            r = read_row(N1DIR / "S1b" / f, sel, bl, tag=tag)
            rows[(r["lat"], r["y"], r["h"], sel)] = r
    return rows


def R(rows, lat, y, h, sel="stored"):
    return rows.get((lat, y, h, sel))


def ref(rows, lat, h, y=2.41):
    """The P_c reference series: stored rows at h in {0, 0.5, 1, 2}, new rows at h in {1.5, 3}."""
    return R(rows, lat, y, h, "stored" if h in STORED else "new")


# ------------------------------------------------------------ derived
def q(a, b):
    """Ratio of two (value, error) pairs, errors in quadrature."""
    v = a[0] / b[0]
    return float(v), float(abs(v) * np.hypot(a[1] / a[0], b[1] / b[0]))


def r_L(row, row0):
    raw = q(row["m"], row0["m"])
    if row["lat"] == "L6x12":
        return dict(value=raw, estimator="raw", free_ratio=None)
    fr = row["free"]["m"] / row0["free"]["m"]
    return dict(value=(raw[0] / fr, raw[1] / fr), estimator="cosh/free", free_ratio=float(fr))


def g_of(row):
    return row["GL_p0"][0] / row["free"]["GL_p0"], row["GL_p0"][1] / row["free"]["GL_p0"]


def exponent(r6, r8):
    e, ee = _exponent(r6["chi_L"][0], r6["chi_L"][1], r8["chi_L"][0], r8["chi_L"][1], r6["V"], r8["V"])
    ef = float(np.log(r8["free"]["chi_L"] / r6["free"]["chi_L"]) / np.log(r8["V"] / r6["V"]))
    return (e, ee), ef


def R_peak(row):
    return row["peak"][0] / row["free"]["peak"], row["peak"][1] / row["free"]["peak"]


def chi_over_free(row):
    return row["chi_L"][0] / row["free"]["chi_L"], row["chi_L"][1] / row["free"]["chi_L"]


def cond_bound(row):
    """|<phi_L>| <= sqrt(chi_L/V), and the same bound for the free theory."""
    return float(np.sqrt(max(row["chi_L"][0], 0) / row["V"])), float(np.sqrt(row["free"]["chi_L"] / row["V"]))


# ------------------------------------------------------------ C2: y = 3.0
def verdict_y3(rows, sel="new"):
    """The y = 3.0 clauses on the (sel) rows at h = 1, 2 with the stored h = 0 denominators."""
    out = dict(sel=sel, rows={})
    r0_8 = R(rows, "L8", 3.0, 0.0)
    r0_6 = R(rows, "L6", 3.0, 0.0)
    a_all, b_any, c_any = True, False, False
    for h in (1.0, 2.0):
        r8, r6 = R(rows, "L8", 3.0, h, sel), R(rows, "L6", 3.0, h, sel)
        if r8 is None or r6 is None:
            continue
        rl, rl6 = r_L(r8, r0_8), r_L(r6, r0_6)
        g, g6 = g_of(r8), g_of(r6)
        (e, ee), ef = exponent(r6, r8)
        Rp8, Rp6 = R_peak(r8), R_peak(r6)
        cf8, cf6 = chi_over_free(r8), chi_over_free(r6)
        d = dict(
            h=h,
            r_L8=rl["value"],
            r_L8_est=rl["estimator"],
            r_L6=rl6["value"],
            g8=g,
            g6=g6,
            e=(e, ee),
            e_free=ef,
            e_minus_free=(e - ef, ee),
            R_peak8=Rp8,
            R_peak6=Rp6,
            chiLf8=cf8,
            chiLf6=cf6,
            peak_raw8=r8["peak"],
            peak_free8=r8["free"]["peak"],
            cond_bound8=cond_bound(r8),
            cond_bound6=cond_bound(r6),
            phi_R8=r8["phi_R"],
            phi_R8_free=r8["free"]["phi_R"],
            m8=r8["m"],
            m8_free=r8["free"]["m"],
            m0_8=r0_8["m"],
            m6=r6["m"],
            fit3_8=r8["fit3"],
            fit3_6=r6["fit3"],
            n8=r8["n"],
            n6=r6["n"],
        )
        d["a_rL"] = bool(rl["value"][0] >= 0.7 - 2 * rl["value"][1])
        d["a_g"] = bool(g[0] <= 0.1 + 2 * g[1])
        d["b1"] = bool(e - ef - 2 * ee > 0.5)
        d["b2"] = bool(Rp8[0] - 2 * Rp8[1] > 1.0 and Rp8[0] - Rp6[0] > 2 * np.hypot(Rp8[1], Rp6[1]))
        d["b3"] = bool(cf8[0] - cf6[0] > 2 * np.hypot(cf8[1], cf6[1]))
        d["b"] = d["b1"] and d["b2"] and d["b3"]
        d["partial_growth"] = bool((e - ef) > 3 * ee and not d["b1"])
        d["null_e"] = bool(abs(e - ef) <= 3 * ee)
        d["c"] = bool(rl["value"][0] + 2 * rl["value"][1] < 0.3 and g[0] >= 0.8 - 2 * g[1])
        d["a"] = d["a_rL"] and d["a_g"] and not d["b"]
        a_all &= d["a"]
        b_any |= d["b"]
        c_any |= d["c"]
        out["rows"][str(h)] = d
    out["class"] = "(b)" if b_any else ("(c)" if c_any else ("(a)" if a_all and out["rows"] else "UNDECIDED"))
    r0_12 = R(rows, "L6x12", 3.0, 0.0)
    out["floor_6x12_stored"] = dict(
        m=r0_12["m"], passes=bool(r0_12["m"][0] - 2 * r0_12["m"][1] >= 0.3), fit3=r0_12["fit3"]
    )
    out["S4_free_gate"] = s4_free_gate(rows)
    out["pooled_usable"] = {k: v["ok"] for k, v in s7_consistency(rows).items() if k != "all_ok"}
    return out


def s4_free_gate(rows):
    """GL_p0^free(h) = 1 to 1e-9 (applied on 6^4/8^4, recorded on 6^3x12) and the free midpoint shift m^free(h) -
    m^free(0)."""
    out = {}
    for lat in ("L8", "L6", "L6x12"):
        for y in (3.0, 2.41):
            for h in (1.0, 1.5, 2.0, 3.0):
                r = R(rows, lat, y, h, "new") or R(rows, lat, y, h, "stored")
                if r is None or (lat, h) in out:
                    continue
                f, f0 = r["free"], R(rows, lat, y, 0.0)["free"]
                out[(lat, h)] = dict(
                    GL_p0_free=f["GL_p0"],
                    gate_ok=bool(abs(f["GL_p0"] - 1) < 1e-9) if lat != "L6x12" else None,
                    dm_free=(f["m"] - f0["m"]) if np.isfinite(f["m"]) and np.isfinite(f0["m"]) else None,
                )
    out["ok_6_8"] = bool(all(v["gate_ok"] for k, v in out.items() if isinstance(k, tuple) and k[0] != "L6x12"))
    return {("|".join(map(str, k)) if isinstance(k, tuple) else k): v for k, v in out.items()}


# ------------------------------------------------------------ T2 / T3 fits
def _scan_min(chi2, lo, hi, n):
    grid = np.linspace(lo, hi, n)
    k0 = grid[int(np.argmin([chi2(k)[0] for k in grid]))]
    return minimize_scalar(lambda k: chi2(k)[0], bounds=(max(0.02, k0 - 0.1), k0 + 0.1), method="bounded").x


def fit_m2(hs, m, sig, m0):
    """m(h)^2 = m0^2 + (c h^kappa)^2, weights 1/(2 m sigma_m)^2, c^2 linear. Returns (kappa, c^2, chi^2)."""
    hs, m, sig = np.asarray(hs, float), np.asarray(m, float), np.asarray(sig, float)
    yv = m**2 - m0**2
    w = 1.0 / (2 * m * sig) ** 2

    def chi2(k):
        X = hs ** (2 * k)
        c2 = np.sum(w * yv * X) / np.sum(w * X * X)
        return float(np.sum(w * (yv - c2 * X) ** 2)), float(c2)

    k = _scan_min(chi2, 0.05, 4.0, 400)
    c2v, c2 = chi2(k)
    return float(k), c2, c2v


def fit_decay(hs, S, sig, Sinf):
    """S(h) - S_inf = a h^-x, a linear. Returns (x, a, chi^2); nan if no point lies above S_inf."""
    hs = np.asarray(hs, float)
    yv = np.asarray(S, float) - Sinf
    w = 1.0 / np.asarray(sig, float) ** 2
    if not np.any(yv > 0):
        return float("nan"), float("nan"), float("nan")

    def chi2(x):
        X = hs ** (-x)
        a = np.sum(w * yv * X) / np.sum(w * X * X)
        return float(np.sum(w * (yv - a * X) ** 2)), float(a)

    x = _scan_min(chi2, 0.05, 6.0, 600)
    c, a = chi2(x)
    return float(x), a, c


def fit_additive(hs, m, sig, m0):
    """Post hoc, recorded only: m(h) = m0 + c h^kappa, c linear."""
    hs = np.asarray(hs, float)
    yv = np.asarray(m, float) - m0
    w = 1.0 / np.asarray(sig, float) ** 2

    def chi2(k):
        X = hs**k
        c = np.sum(w * yv * X) / np.sum(w * X * X)
        return float(np.sum(w * (yv - c * X) ** 2)), float(c)

    k = _scan_min(chi2, 0.05, 4.0, 400)
    c2v, c = chi2(k)
    return float(k), c, c2v


def t2_entries(rows, lat, hs=FIT_H):
    """Jackknife entries for T2 on one lattice: the stored h = 0 chain and the fit rows."""
    ent = {}
    Lt = ref(rows, lat, 0.0)["Lt"]
    t = Lt // 2 - 1
    for h in (0.0,) + tuple(hs):
        r = ref(rows, lat, h)
        ent[f"m{h:g}"] = dict(blocks=r["CL_blocks"], infl=r["infl_CL"], reduce=lambda c, t=t, Lt=Lt: coshmass(c, t, Lt))
    return ent, {h: ref(rows, lat, h)["m"][1] for h in hs}


def T2(rows, lat, hs=FIT_H, entries=None, sig=None):
    ent, sg = (entries, sig) if entries is not None else t2_entries(rows, lat, hs)
    hs = tuple(hs)

    def fitfun(v):
        k, c2, chi2 = fit_m2(hs, [v[f"m{h:g}"] for h in hs], [sg[h] for h in hs], v["m0"])
        return dict(kappa=k, c2=c2, chi2=chi2)

    full, err = jackknife_fit(ent, fitfun)
    k, sk = full["kappa"], err["kappa"]
    dof = len(hs) - 2
    red = full["chi2"] / dof
    if red > 3.0 or not np.isfinite(sk) or sk > 0.3:
        status = "UNDECIDED"
    elif k - 2 * sk > KAPPA_WIN[1] or k + 2 * sk < KAPPA_WIN[0]:
        status = "FAIL"
    else:
        status = "PASS"
    vals = {n: ent[n]["reduce"](ent[n]["blocks"].mean(0)) for n in ent}
    return dict(
        lat=lat,
        hs=list(hs),
        kappa=(k, sk),
        c=(float(np.sqrt(max(full["c2"], 0))), None),
        c2=full["c2"],
        chi2=full["chi2"],
        dof=dof,
        chi2_dof=red,
        status=status,
        m_values={n: float(v) for n, v in vals.items()},
        m_errs={f"m{h:g}": sg[h] for h in hs},
    )


def T3(rows, lat, kappa, hs=T3_H, swap=None):
    """sigma channel: S(pi)(h) - S_inf = a h^-x, S_inf = S(pi) of the stored y = 3.0 h = 0 chain of the lattice.
    swap: None | 'series' (y = 3.0 rows throughout) | 'h2' (y = 3.0 at h = 2) -- the label-swap control S4'."""

    def srow(h):
        yy = 3.0 if swap == "series" or (swap == "h2" and h == 2.0) else 2.41
        return R(rows, lat, yy, h, "stored" if h in STORED else "new")

    hs = tuple(h for h in hs if srow(h) is not None)
    s_inf = R(rows, lat, 3.0, 0.0)
    ent = {"Sinf": dict(blocks=s_inf["Spi_blocks"], infl=s_inf["infl_Spi"], reduce=lambda x: float(x))}
    sig = {}
    for h in hs:
        r = srow(h)
        ent[f"S{h:g}"] = dict(blocks=r["Spi_blocks"], infl=r["infl_Spi"], reduce=lambda x: float(x))
        sig[h] = r["Spi"][1]

    def fitfun(v):
        x, a, chi2 = fit_decay(hs, [v[f"S{h:g}"] for h in hs], [sig[h] for h in hs], v["Sinf"])
        return dict(x=x, a=a, chi2=chi2)

    full, err = jackknife_fit(ent, fitfun)
    x, sx = full["x"], err["x"]
    dof = len(hs) - 2
    red = full["chi2"] / dof if dof > 0 else float("nan")
    kx = kappa[0] * TWO_MINUS_ETA[0]
    skx = float(np.hypot(kappa[1] * TWO_MINUS_ETA[0], kappa[0] * TWO_MINUS_ETA[1]))
    if not np.isfinite(x) or not np.isfinite(sx) or not np.isfinite(red) or red > 3.0:
        status = "UNDECIDED"
    elif abs(x - kx) <= 2 * np.hypot(sx, skx):
        status = "PASS"
    else:
        status = "FAIL"
    return dict(
        lat=lat,
        hs=list(hs),
        swap=swap,
        x=(x, sx),
        a=full["a"],
        chi2=full["chi2"],
        dof=dof,
        chi2_dof=red,
        predicted_x=(kx, skx),
        status=status,
        S_values={h: srow(h)["Spi"] for h in hs},
        Sinf=s_inf["Spi"],
    )


def saturation(rows, lat):
    """S(pi)(3) and |Sigma_stag|(3) at P_c within 25 % (2 sigma) of the y = 3.0 h = 0 values."""
    r3, s = R(rows, lat, 2.41, 3.0, "new"), R(rows, lat, 3.0, 0.0)
    out = {}
    for key in ("Spi", "Sab"):
        v, e = r3[key]
        vi, ei = s[key]
        dev, sd = abs(v - vi), float(np.hypot(e, ei))
        out[key] = dict(
            value=(v, e),
            ref=(vi, ei),
            ratio=v / vi,
            dev_minus_2sigma_over_ref=(dev - 2 * sd) / vi,
            ok=bool(dev - 2 * sd <= 0.25 * vi),
        )
    out["ok"] = out["Spi"]["ok"] and out["Sab"]["ok"]
    return out


def T1(rows, P=None):
    """Monotone rise; P: optional {(lat, h): (m, err)} override (the controls). Also records the 8^4 h = 3 row and the
    fall alert C4 (a fall from h = 1 to 1.5 at > 3 sigma on 6^3x12 and 6^4)."""

    def m(lat, h):
        return P[(lat, h)] if P is not None and (lat, h) in P else ref(rows, lat, h)["m"]

    out, ok = {}, True
    for lat in ("L6x12", "L6"):
        steps = {}
        for ha, hb in ((1.0, 1.5), (2.0, 3.0)):
            a, b = m(lat, ha), m(lat, hb)
            pull = (b[0] - a[0]) / np.hypot(a[1], b[1])
            steps[f"{ha:g}->{hb:g}"] = dict(m_lo=a, m_hi=b, pull=float(pull), ok=bool(pull > 3))
        out[lat] = steps
        ok &= all(s["ok"] for s in steps.values())
    a, b, c = m("L8", 1.0), m("L8", 1.5), m("L8", 2.0)
    lo = (b[0] - a[0]) / np.hypot(a[1], b[1])
    hi = (c[0] - b[0]) / np.hypot(b[1], c[1])
    out["L8"] = dict(m1=a, m1_5=b, m2=c, pull_above_1=float(lo), pull_below_2=float(hi), ok=bool(lo > -2 and hi > -2))
    ok &= out["L8"]["ok"]
    d3 = m("L8", 3.0)
    out["L8_h3_recorded"] = dict(m3=d3, m2=c, pull=float((d3[0] - c[0]) / np.hypot(c[1], d3[1])))
    falls = {
        lat: (lambda a, b: (a[0] - b[0]) / np.hypot(a[1], b[1]))(m(lat, 1.0), m(lat, 1.5)) for lat in ("L6x12", "L6")
    }
    out["C4_alert"] = dict(pulls_fall_1_to_1_5=falls, fires=bool(all(p > 3 for p in falls.values())))
    out["ok"] = bool(ok)
    return out


def T4(rows):
    r8, r6 = R(rows, "L8", 2.41, 1.5, "new"), R(rows, "L6", 2.41, 1.5, "new")
    (e, ee), ef = exponent(r6, r8)
    Rp = R_peak(r8)
    out = dict(
        h=1.5,
        e=(e, ee),
        e_free=ef,
        e_minus_free=(e - ef, ee),
        ok_e=bool(e - ef < 0.5 - 2 * ee),
        R_peak8=Rp,
        ok_peak=bool(abs(Rp[0] - 1) <= 2 * Rp[1]),
    )
    r83, r63 = R(rows, "L8", 2.41, 3.0, "new"), R(rows, "L6", 2.41, 3.0, "new")
    (e3, ee3), ef3 = exponent(r63, r83)
    Rp3 = R_peak(r83)
    out["h3_recorded"] = dict(
        e=(e3, ee3),
        e_free=ef3,
        e_minus_free=(e3 - ef3, ee3),
        ok_e=bool(e3 - ef3 < 0.5 - 2 * ee3),
        R_peak8=Rp3,
        ok_peak=bool(abs(Rp3[0] - 1) <= 2 * Rp3[1]),
    )
    mono = {}
    for lat in ("L6x12", "L6", "L8"):
        seq = [(h, ref(rows, lat, h)["Spi"]) for h in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0) if ref(rows, lat, h) is not None]
        steps = [
            dict(
                step=f"{seq[i][0]:g}->{seq[i + 1][0]:g}",
                rise_pull=float((seq[i + 1][1][0] - seq[i][1][0]) / np.hypot(seq[i][1][1], seq[i + 1][1][1])),
            )
            for i in range(len(seq) - 1)
        ]
        mono[lat] = dict(steps=steps, ok=bool(all(s["rise_pull"] <= 2 for s in steps)), S=list(seq))
    out["S_never_rises"] = mono
    out["ok"] = bool(out["ok_e"] and out["ok_peak"] and mono["L6x12"]["ok"] and mono["L6"]["ok"])
    return out


def T5(rows):
    """V-check: the t = 2 cosh mass 8^4 / 6^4 within 15 % at h = 2 (stored), 1.5 and 3 (new); the h = 0 midpoint gap
    halving from 6^4 to 8^4."""
    out = {}
    for h, sel in ((2.0, "stored"), (1.5, "new"), (3.0, "new")):
        a, b = R(rows, "L8", 2.41, h, sel)["mcosh_t"]["2"], R(rows, "L6", 2.41, h, sel)["mcosh_t"]["2"]
        out[f"h{h:g}_t2"] = dict(m8=a, m6=b, ratio=a[0] / b[0], within15=bool(abs(a[0] / b[0] - 1) <= 0.15))
    a, b = R(rows, "L8", 2.41, 0.0)["m"], R(rows, "L6", 2.41, 0.0)["m"]
    out["h0_midpoint"] = dict(m8=a, m6=b, ratio=a[0] / b[0], halves=bool(0.4 <= a[0] / b[0] <= 0.6))
    out["ok"] = bool(out["h2_t2"]["within15"] and out["h0_midpoint"]["halves"])
    return out


def post_hoc_shape(rows, lat, hs=FIT_H):
    """Recorded only: the additive-form exponent, the quadrature form with m0 free, and the local exponents between
    adjacent h."""
    ent, sg = t2_entries(rows, lat, hs)

    def fit_add(v):
        k, c, chi2 = fit_additive(hs, [v[f"m{h:g}"] for h in hs], [sg[h] for h in hs], v["m0"])
        return dict(kappa=k, c=c, chi2=chi2)

    fa, ea = jackknife_fit(ent, fit_add)

    def fit_free(v):
        m = np.array([v[f"m{h:g}"] for h in hs])
        best = None
        for m0 in np.linspace(0.0, 0.98 * m.min(), 50):
            k, c2, chi2 = fit_m2(hs, m, [sg[h] for h in hs], m0)
            if best is None or chi2 < best[2]:
                best = (k, m0, chi2)
        return dict(kappa=best[0], m0=best[1], chi2=best[2])

    ff, ef = jackknife_fit(ent, fit_free)
    vals = {n: ent[n]["reduce"](ent[n]["blocks"].mean(0)) for n in ent}
    m0 = vals["m0"]
    mm = np.array([vals[f"m{h:g}"] for h in hs])
    H = np.asarray(hs)
    q2, a = mm**2 - m0**2, mm - m0
    return dict(
        post_hoc_recorded=True,
        additive=dict(kappa=(fa["kappa"], ea["kappa"]), c=fa["c"], chi2_dof=fa["chi2"] / (len(hs) - 2)),
        quadrature_m0_free=dict(
            kappa=(ff["kappa"], ef["kappa"]), m0=(ff["m0"], ef["m0"]), chi2_dof=ff["chi2"] / (len(hs) - 3)
        ),
        local_kappa_quadrature=[float(x) for x in np.log(q2[1:] / q2[:-1]) / (2 * np.log(H[1:] / H[:-1]))],
        local_kappa_additive=[float(x) for x in np.log(a[1:] / a[:-1]) / np.log(H[1:] / H[:-1])],
        steps=[f"{H[i]:g}->{H[i + 1]:g}" for i in range(len(H) - 1)],
    )


def anti_seesaw(rows):
    lats = ("L6x12", "L6", "L8")
    out = dict(T1=T1(rows))
    out["post_hoc_shape"] = {lat: post_hoc_shape(rows, lat) for lat in lats}
    out["T2"] = {lat: T2(rows, lat) for lat in lats}
    out["T3"] = {lat: T3(rows, lat, out["T2"][lat]["kappa"]) for lat in lats}
    out["saturation"] = {lat: saturation(rows, lat) for lat in lats}
    out["T4"] = T4(rows)
    out["T5"] = T5(rows)
    t1, t2, t3, t4 = out["T1"]["ok"], out["T2"]["L6x12"]["status"], out["T3"]["L6x12"]["status"], out["T4"]["ok"]
    sat = out["saturation"]["L6x12"]["ok"] and out["saturation"]["L6"]["ok"]
    t3ok = t3 == "PASS" and sat
    if t1 and t2 == "PASS" and t3ok and t4:
        k = out["T2"]["L6x12"]["kappa"]
        reading = f"anti-seesaw certified at L ≤ 8 with κ = {k[0]:.2f}({k[1]:.2f})"
    elif t1 and t2 == "FAIL":
        reading = "anti-seesaw holds, scaling prediction wrong (T1 PASS, T2 FAIL)"
    elif not t1:
        reading = "T1 FAIL: the rise is not monotone — the stored h ≤ 2 rows stand, K8 tier 2 not claimed"
    else:
        reading = (
            f"T1 PASS; T2 {t2}; T3 {t3} (saturation {'ok' if sat else 'FAIL'}); T4 {'PASS' if t4 else 'FAIL'}"
            " — no certified exponent"
        )
    out["reading"] = reading
    out["T3_with_saturation_ok"] = bool(t3ok)
    return out


# ------------------------------------------------------------ S7' and integrity
def s7_consistency(rows):
    """New-only vs stored-only values of the extended chains (3 sigma rule) for m, chi_L, GL_p0."""
    out = {}
    for tag, (_, kind) in CHAINS_1B.items():
        if kind != "ext":
            continue
        new = next(r for r in rows.values() if r.get("tag") == tag and r["sel"] == "new")
        st = R(rows, new["lat"], new["y"], new["h"], "stored")
        pulls = {k: float((new[k][0] - st[k][0]) / np.hypot(new[k][1], st[k][1])) for k in ("m", "chi_L", "GL_p0")}
        out[tag] = dict(
            lat=new["lat"],
            h=new["h"],
            pulls=pulls,
            ok=bool(all(abs(p) <= 3 for p in pulls.values())),
            new={k: new[k] for k in ("m", "chi_L", "GL_p0")},
            stored={k: st[k] for k in ("m", "chi_L", "GL_p0")},
        )
    out["all_ok"] = bool(all(v["ok"] for k, v in out.items() if k != "all_ok"))
    return out


def integrity(rows):
    """Per stage-1b chain: counts, cadence, acceptance, duplicates, git hash, the delivered md5 (recorded in the
    derived file's meta), the kill A(h) < 0 beyond 3 sigma, and for the extensions the seam: the first rows of every
    stored series (and the configuration digests) equal the pre-extension copy."""
    out = {}
    for tag, (f, kind) in CHAINS_1B.items():
        p = N1DIR / "S1b" / f
        d, meta = open_chain(p)
        ct = np.asarray(d["ts_cfg_traj"])
        new = next(r for r in rows.values() if r.get("tag") == tag and r["sel"] == "new")
        rec = dict(
            file=f,
            kind=kind,
            ntraj=len(d["ts_dH"]),
            n_new=new["n"],
            cut=new["cut"],
            cadence=new["cadence"],
            ferm_every=new["ferm_every"],
            mts=new["mts"],
            acc_min50_new=new["acc_min50"],
            dup=int(len(ct) - len(np.unique(ct))),
            git=str(new["git_hash"])[:7],
            md5=meta["source_md5"],
            md5_matches_delivered=bool(meta["source_md5"] == MD5_DELIVERED[tag]),
            A=new["A"],
            A_pull=float(new["A"][0] / new["A"][1]) if new["A"][1] > 0 else None,
            tau_B_over_D=new["tau_B_over_D"],
            infl_CL=new["infl_CL"],
            half_pulls={k: new["stats"][k]["half_pull"] for k in ("CL_mid", "chi_L", "GL_p0", "Spi")},
        )
        if kind == "ext":
            pre, pmeta = open_chain(N1DIR / "S1" / f)
            keys = [k for k in pre.files if k.startswith("ts_")] + ["cfg_digest"]
            rec["seam_ok"] = bool(all(np.array_equal(pre[k], np.asarray(d[k])[: len(pre[k])]) for k in keys))
            rec["pre_md5"] = pmeta["source_md5"]
        rec["kill_A_neg_3sigma"] = bool(new["A"][0] < -3 * new["A"][1])
        rec["acc_ok"] = bool(new["acc_min50"] >= 0.5)
        out[tag] = rec
    return out


# ------------------------------------------------------------ E3: 8^4 quench
def e3_reading(rows):
    """The 8^4 P_c h = 0 chain's stored configurations re-measured with h = 1, 2 in the operator (fixed sigma):
    the light midpoint cosh mass against the h = 0 measurement of the same configurations and the annealed value."""
    z = np.load(N1DIR / "e3_quench_L8_Pc.npz", allow_pickle=False)
    r0 = R(rows, "L8", 2.41, 0.0)
    d, _ = open_chain(r0["file"])
    ct = np.asarray(d["ts_cfg_traj"])
    CL0 = np.asarray(d["ts_CL_t"], float)
    out = dict(
        n_rows=int(len(z["h"])), identity=[float(v) for v, h in zip(z["identity_CL_t"], z["h"], strict=True) if h == 0]
    )
    Lt, t, nb = 8, 3, 10
    for h in (1.0, 2.0):
        sel = np.where(z["h"] == h)[0]
        sel = sel[np.argsort(z["traj"][sel], kind="stable")]
        trajs = z["traj"][sel]
        CLq, GLq = z["CL_t"][sel], z["GL_p0"][sel]
        i0 = [int(np.where(ct == tr)[0][0]) for tr in trajs]
        CL_0 = CL0[i0]

        def jk_m(C):
            v, e = jackknife_fit(
                {"c": dict(blocks=block_means(C, nb), infl=r0["infl_CL"], reduce=lambda c: coshmass(c, t, Lt))},
                lambda v: {"m": v["c"]},
            )
            return v["m"], e["m"]

        mq, eq = jk_m(CLq)
        m0, e0 = jk_m(CL_0)
        per = np.array([coshmass(c, t, Lt) for c in CLq]) - np.array([coshmass(c, t, Lt) for c in CL_0])
        ann = R(rows, "L8", 2.41, h)["m"]
        frac = (mq - m0) / (ann[0] - m0)
        efrac = frac * np.hypot(np.hypot(eq, e0) / (mq - m0), np.hypot(ann[1], e0) / (ann[0] - m0))
        gq = jackknife(lambda a: a, [GLq], nb, 1.0)
        gfree = R(rows, "L8", 2.41, h)["free"]["GL_p0"]
        se = per.std(ddof=1) / np.sqrt(len(per))
        out[f"h{h:g}"] = dict(
            n=len(sel),
            m_quench=(mq, eq),
            m_h0_same_cfgs=(m0, e0),
            m_annealed=ann,
            paired_dm=(float(per.mean()), float(se)),
            paired_pull=float(per.mean() / se),
            n_positive=int((per > 0).sum()),
            fraction=(float(frac), float(efrac)),
            GL_p0_quench_over_free=(gq[0] / gfree, gq[1] / gfree),
            GL_p0_annealed_over_free=g_of(R(rows, "L8", 2.41, h)),
            GL_p0_h0_over_free=g_of(r0),
        )
    return out


# ------------------------------------------------------------ E1 / E2 / E8 tables
def e1_table(rows):
    """The stored stage-1 rows with the stage-1b estimators."""
    out = []
    for key in sorted(k for k in rows if k[3] == "stored"):
        r = rows[key]
        r0 = R(rows, r["lat"], r["y"], 0.0)
        d = dict(
            lat=r["lat"],
            y=r["y"],
            h=r["h"],
            n=r["n"],
            m=r["m"],
            m_free=r["free"]["m"],
            fit3=r["fit3"],
            g=g_of(r),
            chi_L_over_free=chi_over_free(r),
            R_peak=R_peak(r),
            peak_raw=r["peak"],
            cond_bound=cond_bound(r),
            phi_L=r["phi_L"],
            phi_L_free=r["free"]["phi_L"],
            Spi=r["Spi"],
            Sab=r["Sab"],
            phi_stag_sq_V=r["phi_stag_sq"],
            O4=r["O4"],
        )
        if r["h"] > 0:
            d["r_L"] = r_L(r, r0)
        if r["lat"] == "L8":
            r6 = R(rows, "L6", r["y"], r["h"])
            if r6:
                (e, ee), ef = exponent(r6, r)
                d["e"], d["e_free"] = (e, ee), ef
        out.append(d)
    return out


def e2_table(rows):
    """The epsilon/sigma channel of every row."""
    return [
        dict(
            lat=r["lat"],
            y=r["y"],
            h=r["h"],
            sel=r["sel"],
            tag=r.get("tag"),
            Spi=r["Spi"],
            Sab=r["Sab"],
            phi_stag_sq_V=r["phi_stag_sq"],
            O4=r["O4"],
            m=r["m"],
            g=g_of(r),
            chi_L_over_free=chi_over_free(r),
        )
        for r in (rows[k] for k in sorted(rows))
    ]


def two_state_fit(C, sig, Lt, ts):
    """C(t) = A1 cosh[m1(t - Lt/2)] + A2 cosh[m2(t - Lt/2)], A's linear, (m1, m2) on a grid."""
    ts = np.asarray(ts)
    C = np.asarray(C, float)[ts]
    w = 1 / np.asarray(sig, float)[ts] ** 2
    best = None
    for m1 in np.linspace(0.02, 1.2, 60):
        for m2 in np.linspace(m1 + 0.05, 3.0, 60):
            X = np.stack([np.cosh(m1 * (ts - Lt / 2)), np.cosh(m2 * (ts - Lt / 2))], 1)
            sol, *_ = np.linalg.lstsq(np.sqrt(w)[:, None] * X, np.sqrt(w) * C, rcond=None)
            chi2 = float(np.sum(w * (C - X @ sol) ** 2))
            if best is None or chi2 < best[4]:
                best = (float(m1), float(m2), float(sol[0]), float(sol[1]), chi2)
    return best


def e8_table(rows):
    """Fit-window table of the light correlator: every single-cosh window and a two-state fit."""
    out = []
    for lat in ("L6x12", "L8"):
        for y in (2.41, 3.0):
            for h in (0.0, 2.0):
                for sel in ("stored", "new", "pooled"):
                    r = R(rows, lat, y, h, sel)
                    if r is None:
                        continue
                    Lt = r["Lt"]
                    C, E = np.asarray(r["CL_t"]), np.asarray(r["CL_t_err"])
                    wins = []
                    for t1 in range(0, Lt // 2 - 1):
                        for t2 in range(t1 + 2, Lt // 2 + 1):
                            ts = tuple(range(t1, t2 + 1))
                            m, _, chi2 = cosh_fit(C, E, Lt, ts=ts)
                            wins.append(dict(window=[t1, t2], m=m, chi2_dof=chi2 / (len(ts) - 2)))
                    two = two_state_fit(C, E, Lt, range(1, Lt // 2 + 1))
                    out.append(
                        dict(
                            lat=lat,
                            y=y,
                            h=h,
                            sel=sel,
                            n=r["n"],
                            mL_eff=r["mL_eff"],
                            mcosh_t=r["mcosh_t"],
                            windows=wins,
                            two_state=dict(
                                m1=two[0],
                                m2=two[1],
                                A1=two[2],
                                A2=two[3],
                                chi2_dof=two[4] / (Lt // 2 - 4) if Lt // 2 > 4 else None,
                            ),
                        )
                    )
    return out


# ------------------------------------------------------------ controls
def inject_s1(rows):
    """S1': m(h) -> m(0)/(1 + 3h) on the new P_c rows -> T1 must fail and the C4 alert fire."""
    P = {}
    for lat in ("L6x12", "L6", "L8"):
        m0 = R(rows, lat, 2.41, 0.0)["m"]
        for h in (1.5, 3.0):
            f = 1 / (1 + 3 * h)
            P[(lat, h)] = (m0[0] * f, R(rows, lat, 2.41, h, "new")["m"][1] * f)
    return T1(rows, P)


def inject_s2(rows):
    """S2': chi_L(8^4, h = 2) x V_8/V_6 and chi_L(p_min) x 0.5 -> (b) must fire."""
    Rc = copy.deepcopy(rows)
    r = Rc[("L8", 3.0, 2.0, "new")]
    f = 4096 / 1296
    r["chi_L"] = (r["chi_L"][0] * f, r["chi_L"][1] * f)
    r["chi_L_pmin"] = (r["chi_L_pmin"][0] * 0.5, r["chi_L_pmin"][1] * 0.5)
    r["peak"] = (r["peak"][0] * f / 0.5, r["peak"][1] * f / 0.5)
    return verdict_y3(Rc, "new")


def inject_s3(rows):
    """S3': g(2) := 1, m(2) := 0.25 m(0) at y = 3.0 on 8^4 -> (c) must fire."""
    Rc = copy.deepcopy(rows)
    r = Rc[("L8", 3.0, 2.0, "new")]
    m0 = R(rows, "L8", 3.0, 0.0)["m"]
    r["m"] = (0.25 * m0[0], r["m"][1])
    r["GL_p0"] = (r["free"]["GL_p0"], r["GL_p0"][1])
    return verdict_y3(Rc, "new")


def control_s5(rows, lat="L6x12"):
    """S5': synthetic m(h)^2 = m0^2 + (0.3h)^2 with the real per-block scatter -> T2 kappa = 1; synthetic saturating
    m0 + 0.5(1 - e^-h) -> never PASS."""
    ent, sig = t2_entries(rows, lat)
    Lt = R(rows, lat, 2.41, 0.0)["Lt"]
    t = Lt // 2 - 1
    m0 = R(rows, lat, 2.41, 0.0)["m"][0]
    out = {}
    for name, fn in (
        ("linear", lambda h: np.sqrt(m0**2 + (0.3 * h) ** 2)),
        ("saturating", lambda h: m0 + 0.5 * (1 - np.exp(-h))),
    ):
        ent2 = {}
        for key, e in ent.items():
            h = float(key[1:])
            per_block = np.array([coshmass(c, t, Lt) for c in e["blocks"]])
            synth = per_block - per_block.mean() + (m0 if h == 0 else fn(h))
            ent2[key] = dict(blocks=synth, infl=e["infl"], reduce=lambda x: float(x))
        out[name] = T2(rows, lat, entries=ent2, sig=sig)
    out["linear_ok"] = bool(abs(out["linear"]["kappa"][0] - 1.0) <= max(out["linear"]["kappa"][1], 1e-3) + 0.02)
    out["saturating_ok"] = bool(out["saturating"]["status"] != "PASS")
    return out


def inject_free_null(rows):
    """S6': new rows at their free values (r_L = 1, chi_L = free, g = 1, S(pi) = S_inf) -> not (a), not (b), and T1
    fails."""
    Rc = copy.deepcopy(rows)
    for r in Rc.values():
        if r["sel"] != "new":
            continue
        r0 = R(rows, r["lat"], r["y"], 0.0)
        f = r["free"]
        fr = 1.0 if r["lat"] == "L6x12" else f["m"] / r0["free"]["m"]
        r["m"] = (r0["m"][0] * fr, r["m"][1])
        r["chi_L"] = (f["chi_L"], r["chi_L"][1])
        r["chi_L_pmin"] = (f["chi_L_pmin"], r["chi_L_pmin"][1])
        r["peak"] = (f["peak"], r["peak"][1])
        r["GL_p0"] = (f["GL_p0"], r["GL_p0"][1])
        r["Spi"] = (R(rows, r["lat"], 3.0, 0.0)["Spi"][0], r["Spi"][1])
    v, t1 = verdict_y3(Rc, "new"), T1(Rc)
    return dict(verdict=v, T1=t1, ok=bool(v["class"] not in ("(a)", "(b)") and not t1["ok"]))


def selftest(rows):
    """Every stage-1b control (S7' reported beside: a consistency check, not a control)."""
    out = {}
    s1 = inject_s1(rows)
    out["S1'"] = dict(
        T1_ok=s1["ok"], C4_fires=s1["C4_alert"]["fires"], ok=bool(not s1["ok"] and s1["C4_alert"]["fires"])
    )
    s2 = inject_s2(rows)
    d = s2["rows"]["2.0"]
    out["S2'"] = dict(
        b1=d["b1"],
        b2=d["b2"],
        b3=d["b3"],
        cls=s2["class"],
        ok=bool(s2["class"] == "(b)" and d["b1"] and d["b2"] and d["b3"]),
    )
    s3 = inject_s3(rows)
    out["S3'"] = dict(c=s3["rows"]["2.0"]["c"], cls=s3["class"], ok=bool(s3["class"] == "(c)"))
    k = T2(rows, "L6x12")["kappa"]
    sw = {s: T3(rows, "L6x12", k, swap=s) for s in ("series", "h2")}
    out["S4'"] = {
        s: dict(status=v["status"], x=v["x"], a=v["a"], chi2_dof=v["chi2_dof"], hs=v["hs"]) for s, v in sw.items()
    }
    out["S4'"]["ok"] = bool(sw["h2"]["status"] in ("UNDECIDED", "FAIL"))
    out["S4'"]["series_as_written_ok"] = bool(sw["series"]["status"] in ("UNDECIDED", "FAIL"))
    s5 = control_s5(rows)
    out["S5'"] = dict(
        linear_kappa=s5["linear"]["kappa"],
        linear_status=s5["linear"]["status"],
        saturating_kappa=s5["saturating"]["kappa"],
        saturating_status=s5["saturating"]["status"],
        saturating_chi2_dof=s5["saturating"]["chi2_dof"],
        ok=bool(s5["linear_ok"] and s5["saturating_ok"]),
    )
    s6 = inject_free_null(rows)
    out["S6'"] = dict(cls=s6["verdict"]["class"], T1_ok=s6["T1"]["ok"], ok=s6["ok"])
    out["S7'"] = s7_consistency(rows)
    out["all_ok"] = bool(all(out[k]["ok"] for k in ("S1'", "S2'", "S3'", "S4'", "S5'", "S6'")))
    return out


def analyse():
    """(rows, results) of the stage-1b analysis."""
    rows = read_all(free_tables(FREE_1B))
    res = dict(
        verdict_y3_new=verdict_y3(rows, "new"),
        verdict_y3_pooled=verdict_y3(rows, "pooled"),
        anti_seesaw=anti_seesaw(rows),
        S7=s7_consistency(rows),
        integrity=integrity(rows),
        E3=e3_reading(rows),
    )
    return rows, res


def fmt(p, d=4):
    return f"{p[0]:.{d}f}({p[1]:.{d}f})" if p[1] is not None and np.isfinite(p[1]) else f"{p[0]:.{d}f}"


def to_json(o):
    """JSON-ready copy; non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): to_json(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [to_json(v) for v in o]
    if isinstance(o, np.floating | float):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ------------------------------------------------------------ E9: box shape
def element_table(shape):
    """The stabiliser of C_chi among the box's signed hypercubic maps and the action of each element on the (0,3)
    anti-self-dual pattern C_asd(0,3) (+1 keeps, -1 flips) and on its light projection; whether flipping elements
    exchange a spatial axis with the time axis."""
    import itertools

    from ..lattice import Lattice, kinetic_matrix
    from ..patterns import TasteProjector, chiral_mass
    from ..symmetry import closure, generators, site_map, stabiliser, transform

    lat = Lattice(shape, bc=(-1,) * 4)
    G = closure(generators(lat, kinetic_matrix(lat), translations=False), lat.V)
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
        r = float(np.sum(transform(Ca, g, s) * Ca) / np.sum(Ca * Ca))
        rl = float(np.sum(transform(CL, g, s) * CL) / np.sum(CL * CL))
        rows.append(dict(perm=list(perm), refl=list(refl), moves_time=bool(perm[3] != 3), action=r, action_PL=rl))
    flips = [e for e in rows if e["action"] < -0.5]
    keeps = [e for e in rows if e["action"] > 0.5]
    return dict(
        G=len(G),
        stab=len(S),
        n_flip=len(flips),
        n_keep=len(keeps),
        all_flips_move_time=bool(all(e["moves_time"] for e in flips)),
        mean_action=float(np.mean([e["action"] for e in rows])),
        mean_action_PL=float(np.mean([e["action_PL"] for e in rows])),
    )


def free_phiL(L, Lt, h, bc=(-1, -1, -1, -1)):
    """Free per-flavour <phi_L> = (1/2V) sum_{xz} C_asd(0,3)[x,z] G_L[x,z], G_L = P_L (K - h C_chi)^-1 P_L (dense)."""
    from ..lattice import Lattice, kinetic_matrix
    from ..patterns import TasteProjector, chiral_mass

    lat = Lattice((L, L, L, Lt), bc=tuple(bc))
    M = kinetic_matrix(lat).toarray() - h * chiral_mass(lat).toarray()
    Minv = np.linalg.inv(M)
    tp = TasteProjector(lat)
    GL = tp.PL(np.ascontiguousarray(tp.PL(Minv).T)).T
    Ca = chiral_mass(lat, (0, 3), -1).tocoo()
    return float(0.5 * np.sum(Ca.data * GL[Ca.row, Ca.col]) / lat.V)
