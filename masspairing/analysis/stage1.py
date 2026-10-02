"""Stage 1 of the N1 test (26 chains: 8^4, 6^3x12, 6^4 aaaa x y in {2.41, 3.0} x h in {0, 0.5, 1, 2}; 6^4 pppa at
y = 2.41, h in {0, 0.5}): the reader, the estimators, the order's verdict with its controls, the K4 composite readout
and the alpha_L ensemble readout of the corner blocks.

Reader (fixed before any stage-1 observable was read): measurements with trajectory index < 100 cut; the error model
of n1stage (tau_B from the per-trajectory |Sigma_stag| and sigma^2 series). Estimators per chain:
  mL, mR        log-ratio ln C(t*)/C(t*+1) of CL_t, CR_t at t* = L_t/2 - 1 (the order's primary gap proxy)
  mcosh_L_t<t>  cosh effective mass of the mean CL_t at t = L_t/2 - 2, L_t/2 - 1 (jackknife)
  fit_L         on 6^3x12: the cosh fit over t in [3, 6]; "exists" iff chi^2/dof <= 2 and m > 2 sigma_m
  A             on-configuration asymmetry sum_t (CL_t - CR_t) / sum_t CL_t
  rho           |phi_T_R / phi_R| (composite / elementary), a lower bound where phi_R is unresolved
The order's criteria (fixed before the data, see the claims): r_L(h) = [m_L(h)/m_L(0)] / [free ratio] with the
primary estimator (the 6^3x12 fit if it exists at every h of that y, else t*); e(h) = d ln chi_L / d ln V on
(6^4, 8^4); g(h) = GL_p0/free; (a) r_L >= 0.7 - 2 sigma at every h on 8^4 and 6^3x12, e + 2 sigma < 0.5, g <= 0.8 +
2 sigma; (b) e - 2 sigma > 0.5 at some h; (c) r_L + 2 sigma < 0.3 with g >= 0.8 - 2 sigma; gap floor m_L(0) on
6^3x12 >= 0.3; V-stability between 6^4 and 8^4; the M trigger at y = 2.41 (r_L falling to r_L(2) + 2 sigma < 0.7 at both
volumes with a V-stable log-slope, absent at y = 3.0).
"""

from __future__ import annotations

import copy

import numpy as np

from ..corner import odd_exponent, parity_split
from . import stage0
from .n1stage import (
    N1DIR,
    cosh_fit,
    cosh_mass_point,
    err_model,
    exponent,
    free_lookup,
    free_tables,
    jackknife,
    open_chain,
    sigma_block,
    tau_int,
)

CUT_TRAJ = 100
S1_DIRS = ("L8", "L6x12", "L6", "L6pppa")
FREE_S1 = ("free_L8_aaaa.json", "free_L6x12_aaaa.json", "free_baselines.json")


# ------------------------------------------------------------ reader
def read_chain(path, bl=None):
    d, meta = open_chain(path)
    L, Lt = meta["L"], meta.get("Lt") or meta["L"]
    V = L**3 * Lt
    bc = meta["bc_str"]
    h = float(meta["h10"])
    y = float(meta["y"])
    traj = np.asarray(d["ts_cfg_traj"])
    dig = d["cfg_digest"]
    dup, seen, keep = [], {}, []
    for i, t in enumerate(traj):
        t = int(t)
        if t in seen:
            dup.append(dict(traj=t, first=seen[t], second=i, identical_sigma=bool(dig[seen[t]] == dig[i])))
            continue
        seen[t] = i
        keep.append(i)
    keep = np.array([i for i in keep if traj[i] >= CUT_TRAJ])
    n = len(keep)
    nb = 20 if n >= 150 else 10
    D = float(np.median(np.diff(traj[keep])))
    ntr = len(d["ts_sigma2"])
    tauB = max(tau_int(d["ts_Sigma_stag_abs"], wmax=ntr // 10), tau_int(d["ts_sigma2"], wmax=ntr // 10))
    tBD = tauB / D
    acc = np.asarray(d["ts_accepted"], bool)
    acc50 = [float(acc[i : i + 50].mean()) for i in range(50, ntr, 50)]
    row = dict(
        file=str(path),
        name=path.stem,
        y=y,
        L=L,
        Lt=Lt,
        bc=bc,
        h=h,
        V=V,
        n_raw=len(traj),
        n=n,
        nb=nb,
        cadence=np.unique(np.diff(traj[keep])).tolist(),
        Delta=D,
        tau_B=tauB,
        tau_B_over_D=tBD,
        duplicates=dup,
        acc_min50=float(min(acc50)) if acc50 else float("nan"),
        git_hash=meta.get("git_hash"),
        comp=meta.get("h10_comp"),
        dual_sign=meta.get("h10_dual_sign"),
        pattern=meta.get("h10_pattern"),
    )
    CL = np.asarray(d["ts_CL_t"], float)[keep]
    CR = np.asarray(d["ts_CR_t"], float)[keep]
    t = Lt // 2 - 1
    ser = {
        "mL": np.log(CL[:, t] / CL[:, t + 1]),
        "mR": np.log(CR[:, t] / CR[:, t + 1]),
        "chi_L": np.asarray(d["ts_chi_L_sum"], float)[keep],
        "GL_p0": np.asarray(d["ts_GL_p0"], float)[keep],
        "GR_p0": np.asarray(d["ts_GR_p0"], float)[keep],
        "chi_R": np.asarray(d["ts_chi_f_sum"], float)[keep],
        "phi_R": np.asarray(d["ts_phi_f"], float)[keep].sum(1),
        "phi_T_R": np.asarray(d["ts_phi_T_R"], float)[keep],
        "phi_T_L": np.asarray(d["ts_phi_T_L"], float)[keep],
        "CL_tstar": CL[:, t],
        "A_num": (CL - CR).sum(1),
        "A_den": CL.sum(1),
    }
    stats = {}
    for k in ("mL", "mR", "chi_L", "GL_p0", "GR_p0", "chi_R", "phi_R", "phi_T_R", "phi_T_L", "CL_tstar"):
        m, s, info = err_model(ser[k], nb, tBD)
        row[k] = (m, s)
        stats[k] = info
    row["stats"] = stats
    infl_C = stats["CL_tstar"]["infl"]
    row["A"] = jackknife(lambda a, b: a / b, [ser["A_num"], ser["A_den"]], nb, infl_C)
    CLm, CRm = CL.mean(0), CR.mean(0)
    CLe = np.array([sigma_block(CL[:, i], nb) for i in range(Lt)]) * infl_C
    CRe = np.array([sigma_block(CR[:, i], nb) for i in range(Lt)]) * infl_C
    row["CL_t"], row["CL_t_err"] = CLm.tolist(), CLe.tolist()
    row["CR_t"], row["CR_t_err"] = CRm.tolist(), CRe.tolist()
    row["mL_eff"] = [float(np.log(CLm[i] / CLm[i + 1])) for i in range(Lt // 2)]
    row["mR_eff"] = [float(np.log(CRm[i] / CRm[i + 1])) for i in range(Lt // 2)]
    for key, Cs in (("L", CL), ("R", CR)):
        for tt in (Lt // 2 - 2, Lt // 2 - 1):
            row[f"mcosh_{key}_t{tt}"] = jackknife(lambda c, tt=tt: cosh_mass_point(c, tt, Lt), [Cs], nb, infl_C)
    if Lt == 12:
        for key, Cs, Ce in (("L", CL, CLe), ("R", CR, CRe)):
            m, A, chi2 = cosh_fit(Cs.mean(0), Ce, Lt)
            mj = jackknife(lambda c, Ce=Ce: cosh_fit(c, Ce, Lt)[0], [Cs], nb, infl_C)
            row[f"fit_{key}"] = dict(
                m=m, err=mj[1], A=A, chi2=chi2, dof=2, exists=bool(chi2 / 2 <= 2.0 and m > 2 * mj[1])
            )
    if h == 0.0:
        pulls = []
        for tt in range(Lt):
            m_, e_ = jackknife(lambda a: a, [CL[:, tt] - CR[:, tt]], nb, infl_C)
            pulls.append(m_ / e_ if e_ > 0 else 0.0)
        dm, edm = jackknife(lambda a: a, [ser["mL"] - ser["mR"]], nb, infl_C)
        row["h0_check"] = dict(
            max_pull=float(max(abs(p) for p in pulls)), dm_pull=float(abs(dm / edm)) if edm > 0 else 0.0
        )
    if bl is not None:
        fr = free_lookup(bl, L, Lt, bc, h)
        if fr is not None:
            f = dict(
                mL=fr["mL_eff"][t],
                mR=fr["mR_eff"][t],
                chi_L=fr["chi_L_sum"],
                GL_p0=fr["GL_p0"],
                GR_p0=fr["GR_p0"],
                chi_R=fr["chi_f_sum"],
                phi_R=float(np.sum(fr["phi_f"])),
                phi_T_R=fr.get("phi_T_R"),
                CL_t=fr["CL_t"],
                CR_t=fr["CR_t"],
                A=(np.sum(fr["CL_t"]) - np.sum(fr["CR_t"])) / np.sum(fr["CL_t"]),
                mL_eff=fr["mL_eff"],
                mR_eff=fr["mR_eff"],
            )
            f[f"mcosh_L_t{Lt // 2 - 1}"] = cosh_mass_point(np.asarray(fr["CL_t"]), Lt // 2 - 1, Lt)
            f[f"mcosh_L_t{Lt // 2 - 2}"] = cosh_mass_point(np.asarray(fr["CL_t"]), Lt // 2 - 2, Lt)
            if Lt == 12:
                for key in ("L", "R"):  # the data's relative weights
                    Ce = np.asarray(row[f"C{key}_t_err"]) / np.asarray(row[f"C{key}_t"]) * np.asarray(fr[f"C{key}_t"])
                    f[f"fit_{key}"] = cosh_fit(np.asarray(fr[f"C{key}_t"]), Ce, Lt)[0]
            row["free"] = f
    return row


def s1_files():
    return [p for sub in S1_DIRS for p in sorted((N1DIR / "S1" / sub).glob("*.npz"))]


def stage1_rows():
    """The 26 stage-1 rows with the derived quantities attached."""
    return derived([read_chain(f, free_tables(FREE_S1)) for f in s1_files()])


# ------------------------------------------------------------ derived quantities
def get(rows, y, L, Lt, h, bc="aaaa"):
    return next(
        (
            r
            for r in rows
            if abs(r["y"] - y) < 1e-9 and r["L"] == L and r["Lt"] == Lt and r["bc"] == bc and abs(r["h"] - h) < 1e-9
        ),
        None,
    )


def primary_mass(r, key="L"):
    """(m, err, estimator): the 6^3x12 cosh fit where the caller set r['use_fit'], else the t* log-ratio."""
    if r.get("use_fit"):
        return r[f"fit_{key}"]["m"], r[f"fit_{key}"]["err"], "fit"
    return r[f"m{key}"][0], r[f"m{key}"][1], "tstar"


def primary_free_mass(r, key="L"):
    return r["free"][f"fit_{key}"] if r.get("use_fit") else r["free"][f"m{key}"]


def decide_fit(rows):
    """6^3x12: the fit is used for a y iff it exists at every h (incl. 0) of that y."""
    for y in sorted({r["y"] for r in rows}):
        g = [r for r in rows if r["Lt"] == 12 and abs(r["y"] - y) < 1e-9]
        ok = bool(g) and all(r["fit_L"]["exists"] for r in g)
        for r in g:
            r["use_fit"] = ok


def r_L(r, r0):
    m, e, est = primary_mass(r)
    m0, e0, _ = primary_mass(r0)
    fr = primary_free_mass(r) / primary_free_mass(r0)
    v = m / m0 / fr
    err = abs(v) * np.sqrt((e / m) ** 2 + (e0 / m0) ** 2) if m > 0 and m0 > 0 else float("nan")
    return float(v), float(err), fr, est


def _rl_tstar(r, r0):
    m, e = r["mL"]
    m0, e0 = r0["mL"]
    v = m / m0 / (r["free"]["mL"] / r0["free"]["mL"])
    return v, abs(v) * np.sqrt((e / m) ** 2 + (e0 / m0) ** 2)


def derived(rows):
    """Attach r_L, g, the volume exponents, A pulls and the K4 ratios (h = 0 partner at the same y, L, Lt, bc)."""
    decide_fit(rows)
    nan2 = (float("nan"), float("nan"))
    for r in rows:
        r0 = get(rows, r["y"], r["L"], r["Lt"], 0.0, r["bc"])
        f = r.get("free", {})
        r["g"] = (r["GL_p0"][0] / f["GL_p0"], r["GL_p0"][1] / f["GL_p0"]) if f.get("GL_p0") else nan2
        r["chi_L_over_free"] = r["chi_L"][0] / f["chi_L"] if f else float("nan")
        r["chi_R_over_free"] = r["chi_R"][0] / f["chi_R"] if f and f["chi_R"] else float("nan")
        if r["h"] > 0:
            r["r_L"] = r_L(r, r0) if r0 else None
            r["rL_tstar_only"] = _rl_tstar(r, r0) if r0 else None
            if r0:
                q = r["chi_L"][0] / r0["chi_L"][0]
                r["chi_L_over_h0"] = (q, q * np.hypot(r["chi_L"][1] / r["chi_L"][0], r0["chi_L"][1] / r0["chi_L"][0]))
            else:
                r["chi_L_over_h0"] = None
            r["phi_R_per_h"] = (r["phi_R"][0] / r["h"], r["phi_R"][1] / r["h"])
            r["phi_T_R_per_h"] = (r["phi_T_R"][0] / r["h"], r["phi_T_R"][1] / r["h"])
            r["phi_R_over_free"] = r["phi_R"][0] / f["phi_R"] if f and f["phi_R"] else float("nan")
            r["phi_T_R_over_free"] = r["phi_T_R"][0] / f["phi_T_R"] if f and f.get("phi_T_R") else float("nan")
            pr, epr = r["phi_R"]
            pt, ept = r["phi_T_R"]
            if abs(pr) > 2 * epr:
                rho = abs(pt / pr)
                r["rho"] = (float(rho), float(rho * np.hypot(ept / pt if pt else 0, epr / pr)), "resolved")
            else:
                lb = float(abs(pt) / (abs(pr) + epr)) if (abs(pr) + epr) > 0 else float("inf")
                r["rho"] = (lb, float("nan"), "phi_R unresolved: lower bound |phi_T_R|/(|phi_R|+σ)")
            r["A_pull"] = r["A"][0] / r["A"][1] if r["A"][1] > 0 else float("inf")
            r["phi_R_pull"] = pr / epr if epr > 0 else float("inf")
            r["phi_T_R_pull"] = pt / ept if ept > 0 else float("inf")
    for r in rows:
        if r["L"] == 8:
            r6 = get(rows, r["y"], 6, 6, r["h"])
            ok = r6 is not None
            r["e"] = (
                exponent(r6["chi_L"][0], r6["chi_L"][1], r["chi_L"][0], r["chi_L"][1], r6["V"], r["V"]) if ok else nan2
            )
            r["e_R"] = (
                exponent(r6["chi_R"][0], r6["chi_R"][1], r["chi_R"][0], r["chi_R"][1], r6["V"], r["V"]) if ok else nan2
            )
    return rows


# ------------------------------------------------------------ the verdict
def class_reading(rows, y, L, Lt, use_g=True):
    """The (a)/(c) clauses from r_L (and g if use_g) on one lattice."""
    hs = sorted(
        {
            r["h"]
            for r in rows
            if abs(r["y"] - y) < 1e-9 and r["L"] == L and r["Lt"] == Lt and r["bc"] == "aaaa" and r["h"] > 0
        }
    )
    a_r, c_r, a_g, notes = True, False, True, []
    for h in hs:
        r = get(rows, y, L, Lt, h)
        v, e = r["r_L"][0], r["r_L"][1]
        if not (v >= 0.7 - 2 * e):
            a_r = False
        g, ge = r["g"]
        c_here = v + 2 * e < 0.3
        if use_g:
            if not (g <= 0.8 + 2 * ge):
                a_g = False
            c_here = c_here and (g >= 0.8 - 2 * ge)
        if c_here:
            c_r = True
        notes.append(dict(h=h, r_L=(v, e), g=(g, ge)))
    return dict(a_rL=a_r, a_g=a_g, c=c_r, rows=notes, hs=hs)


def verdict(rows, y):
    """The order's (a)/(b)/(c)/no-verdict at one y with every clause."""
    out = dict(y=y)
    r8 = class_reading(rows, y, 8, 8)
    r6 = class_reading(rows, y, 6, 6)
    r12 = class_reading(rows, y, 6, 12, use_g=False)
    es = {h: get(rows, y, 8, 8, h)["e"] for h in r8["hs"]}
    b_flag = any(e - 2 * ee > 0.5 for e, ee in es.values())
    a_e = all(e + 2 * ee < 0.5 for e, ee in es.values())
    out.update(e=es, b_flag=b_flag, a_e=a_e, L8=r8, L6=r6, L6x12=r12)
    r0 = get(rows, y, 6, 12, 0.0)
    m0, e0, est = primary_mass(r0)
    gap_ok = bool(m0 >= 0.3 and m0 > e0)
    out.update(mL0_6x12=(m0, e0, est), gap_readable=gap_ok)
    a_8 = r8["a_rL"] and r8["a_g"] and a_e and r12["a_rL"]
    c_8 = r8["c"] or r12["c"]
    v8 = "(b)" if b_flag else ("(c)" if c_8 else ("(a)" if a_8 else "UNDECIDED"))
    a_6 = r6["a_rL"] and r6["a_g"] and a_e
    v6 = "(b)" if b_flag else ("(c)" if r6["c"] else ("(a)" if a_6 else "UNDECIDED"))
    out.update(class_L8=v8, class_L6=v6, V_stable=(v8 == v6))
    if not gap_ok:
        final = "NO VERDICT at L ≤ 8 (gap channel: m_L(0) on 6³×12 below the 0.3 floor)" + (
            "" if not b_flag else " — (b) flagged (SSB channel unaffected)"
        )
    elif v8 == v6 and v8 != "UNDECIDED":
        final = v8
    elif v8 == "UNDECIDED" and v6 == "UNDECIDED":
        final = "NO VERDICT (none of (a)/(b)/(c) at both volumes)"
    else:
        final = f"UNDECIDED at L ≤ 8 (L = 8 reads {v8}, L = 6 reads {v6})"
    out["verdict"] = final
    return out


def m_trigger(rows, y=2.41, y_smg=3.0):
    """The M trigger: r_L falling monotonically (2 sigma) with r_L(2) + 2 sigma < 0.7 on 6^4 and 8^4, a V-stable
    log-slope e_r on the last two h, and absent at y = 3.0."""

    def fall(L, Lt, yy):
        hs = sorted(
            {
                r["h"]
                for r in rows
                if abs(r["y"] - yy) < 1e-9 and r["L"] == L and r["Lt"] == Lt and r["bc"] == "aaaa" and r["h"] > 0
            }
        )
        vals = [get(rows, yy, L, Lt, h)["r_L"][:2] for h in hs]
        mono = all(
            vals[i + 1][0] <= vals[i][0] + 2 * np.hypot(vals[i][1], vals[i + 1][1]) for i in range(len(vals) - 1)
        )
        last = vals[-1][0] + 2 * vals[-1][1] < 0.7
        lh = np.log(hs[-1] / hs[-2])
        er = np.log(vals[-1][0] / vals[-2][0]) / lh
        ere = np.sqrt((vals[-1][1] / vals[-1][0]) ** 2 + (vals[-2][1] / vals[-2][0]) ** 2) / lh
        return dict(
            hs=hs,
            r_L=vals,
            mono=bool(mono),
            last_below=bool(last),
            fall=bool(mono and last),
            e_r=(float(er), float(ere)),
        )

    f6, f8 = fall(6, 6, y), fall(8, 8, y)
    vstab = abs(f6["e_r"][0] - f8["e_r"][0]) < 2 * np.hypot(f6["e_r"][1], f8["e_r"][1])
    s8 = get(rows, y_smg, 8, 8, 2.0)["r_L"]
    absent = s8[0] >= 0.7 - 2 * s8[1]
    trig = f6["fall"] and f8["fall"] and vstab and absent
    return dict(
        L6=f6, L8=f8, e_r_Vstable=bool(vstab), absent_at_smg=bool(absent), rL2_smg_L8=s8[:2], triggered=bool(trig)
    )


# ------------------------------------------------------------ K4 on 8^4, 6^4, 6^3x12
def k4(rows):
    """The composite readout: (i) SMG rho > 4 and phi_R/h <= 0.05 x free; (ii) P_c rho < 0.2; (iii) phi_T_R/h at
    y = 3.0 h-independent on 8^4 and equal on 6^4 / 8^4 within max(10 %, 2 sigma); (iv) no reversal with the volume;
    the y-label swap must fail."""
    res = {}
    for L, Lt in ((8, 8), (6, 6), (6, 12)):
        for y in (2.41, 3.0):
            for h in (0.5, 1.0, 2.0):
                r = get(rows, y, L, Lt, h)
                if r is None:
                    continue
                res[f"{L}x{Lt}_y{y}_h{h}"] = dict(
                    L=L,
                    Lt=Lt,
                    y=y,
                    h=h,
                    phi_R_per_h=r["phi_R_per_h"],
                    phi_R_over_free=r["phi_R_over_free"],
                    phi_R_pull=r["phi_R_pull"],
                    phi_T_R_per_h=r["phi_T_R_per_h"],
                    phi_T_R_over_free=r["phi_T_R_over_free"],
                    phi_T_R_pull=r["phi_T_R_pull"],
                    rho=r["rho"],
                    A=r["A"],
                    A_pull=r["A_pull"],
                )

    def ok_smg(e):
        lo = e["rho"][0] - e["rho"][1] if e["rho"][2] == "resolved" else e["rho"][0]
        return (e["rho"][0] > 4 if e["rho"][2] == "resolved" else lo > 4) and (abs(e["phi_R_over_free"]) <= 0.05)

    def ok_sym(e):
        res_ = e["rho"][2] == "resolved"
        return res_ and e["rho"][0] + 2 * e["rho"][1] < 0.2 or (res_ and e["rho"][0] < 0.2)

    crit = {}
    for L, Lt in ((8, 8), (6, 6), (6, 12)):
        smg = [v for v in res.values() if v["L"] == L and v["Lt"] == Lt and v["y"] == 3.0]
        sym = [v for v in res.values() if v["L"] == L and v["Lt"] == Lt and v["y"] == 2.41]
        crit[f"{L}x{Lt}"] = dict(
            i_smg=all(ok_smg(e) for e in smg), ii_sym=all(ok_sym(e) for e in sym), n=len(smg) + len(sym)
        )
    p8 = {v["h"]: v["phi_T_R_per_h"] for v in res.values() if v["L"] == 8 and v["y"] == 3.0}
    p6 = {v["h"]: v["phi_T_R_per_h"] for v in res.values() if v["L"] == 6 and v["Lt"] == 6 and v["y"] == 3.0}
    mean8 = np.mean([v[0] for v in p8.values()])
    lin = all(abs(v[0] - mean8) <= max(0.10 * abs(mean8), 2 * v[1]) for v in p8.values())
    vol = all(
        abs(p8[h][0] - p6[h][0]) <= max(0.10 * abs(p6[h][0]), 2 * np.hypot(p8[h][1], p6[h][1])) for h in p8 if h in p6
    )
    crit["iii_linear_8"] = bool(lin)
    crit["iii_volume_6_8"] = bool(vol)
    crit["iv_no_reversal"] = bool(all(crit[k]["i_smg"] and crit[k]["ii_sym"] for k in ("8x8", "6x6", "6x12")))
    swap_i = all(ok_smg(e) for e in [v for v in res.values() if v["L"] == 8 and v["y"] == 2.41])
    swap_ii = all(ok_sym(e) for e in [v for v in res.values() if v["L"] == 8 and v["y"] == 3.0])
    crit["control_swap_fails"] = not (swap_i and swap_ii)
    crit["holds_at_8"] = bool(crit["8x8"]["i_smg"] and crit["8x8"]["ii_sym"] and lin and vol and crit["iv_no_reversal"])
    return dict(rows=res, criteria=crit)


# ------------------------------------------------------------ controls
def inject_s1(rows):
    """S1 as written: m_L(h) -> m_L(h)/(1 + 3h) on every h > 0 row (all estimators)."""
    R = copy.deepcopy(rows)
    for r in R:
        if r["h"] > 0:
            f = 1 / (1 + 3 * r["h"])
            r["mL"] = (r["mL"][0] * f, r["mL"][1] * f)
            if "fit_L" in r:
                r["fit_L"]["m"] *= f
                r["fit_L"]["err"] *= f
    return derived(R)


def inject_s1_prime(rows):
    """S1': S1 with g set to free."""
    R = inject_s1(rows)
    for r in R:
        if r["h"] > 0:
            r["g"] = (1.0, r["g"][1])
    return R


def inject_s1_double_prime(rows, ys=(2.41, 3.0)):
    """S1'': r_L(h) := 1/(1 + 3h) at the y in ys (m_L(h) = m_L(0) x free ratio / (1 + 3h), every estimator);
    g -> free."""
    R = copy.deepcopy(rows)
    decide_fit(R)
    for r in R:
        if r["h"] > 0 and any(abs(r["y"] - yy) < 1e-9 for yy in ys):
            r0 = get(R, r["y"], r["L"], r["Lt"], 0.0, r["bc"])
            f = 1 / (1 + 3 * r["h"])
            fr = r["free"]["mL"] / r0["free"]["mL"]
            rel = r["mL"][1] / r["mL"][0]
            r["mL"] = (r0["mL"][0] * fr * f, abs(r0["mL"][0] * fr * f) * rel)
            if "fit_L" in r:
                frf = r["free"]["fit_L"] / r0["free"]["fit_L"] if r0["free"]["fit_L"] else 1.0
                r["fit_L"]["m"] = r0["fit_L"]["m"] * frf * f
                r["fit_L"]["err"] = r["fit_L"]["m"] * rel
    R = derived(R)
    for r in R:
        if r["h"] > 0 and any(abs(r["y"] - yy) < 1e-9 for yy in ys):
            r["g"] = (1.0, r["g"][1])
    return R


def inject_s2(rows):
    """S2: injected SSB, chi_L(8^4, h > 0.3) x V_8/V_6."""
    R = copy.deepcopy(rows)
    for r in R:
        if r["L"] == 8 and r["h"] > 0.3:
            r["chi_L"] = (r["chi_L"][0] * 4096 / 1296, r["chi_L"][1] * 4096 / 1296)
    return derived(R)


def inject_free_null(rows):
    """Null injection: every light number at its free value (r_L = 1 within its error, chi_L = free, g = 1)."""
    R = copy.deepcopy(rows)
    for r in R:
        f = r.get("free")
        if not f:
            continue
        rel = r["mL"][1] / r["mL"][0]
        r["mL"] = (f["mL"], abs(f["mL"]) * rel)
        if "fit_L" in r:
            r["fit_L"]["m"] = f["fit_L"]
            r["fit_L"]["err"] = f["fit_L"] * rel
        r["chi_L"] = (f["chi_L"], f["chi_L"] * r["chi_L"][1] / r["chi_L"][0])
        r["GL_p0"] = (f["GL_p0"], r["GL_p0"][1])
    return derived(R)


def control_summary(R):
    """The clause outcomes of a (possibly injected) row set."""
    out = {}
    for y in (2.41, 3.0):
        v = verdict(R, y)
        out[f"y{y}"] = dict(
            verdict=v["verdict"],
            class_L8=v["class_L8"],
            class_L6=v["class_L6"],
            b_flag=v["b_flag"],
            a_e=v["a_e"],
            gap=v["gap_readable"],
            L8=dict(a_rL=v["L8"]["a_rL"], a_g=v["L8"]["a_g"], c=v["L8"]["c"]),
            L6=dict(a_rL=v["L6"]["a_rL"], a_g=v["L6"]["a_g"], c=v["L6"]["c"]),
            L6x12=dict(a_rL=v["L6x12"]["a_rL"], c=v["L6x12"]["c"]),
        )
    m = m_trigger(R)
    out["M"] = dict(
        triggered=m["triggered"],
        fall6=m["L6"]["fall"],
        fall8=m["L8"]["fall"],
        Vstable=m["e_r_Vstable"],
        absent=m["absent_at_smg"],
    )
    return out


def controls(rows):
    """Every control on the real rows: S1, S1', S1'', S1_M (S1'' at y = 2.41 only), S2, the free null, S4 (the stage-0
    y = 2.0 rows read with max(2 sigma, 3 %); the free instrument gate on every stage-1 lattice) and the reader's
    synthetic self-test."""
    res = dict(
        real=control_summary(rows),
        S1=control_summary(inject_s1(rows)),
        S1_prime=control_summary(inject_s1_prime(rows)),
        S1_double_prime=control_summary(inject_s1_double_prime(rows)),
        S1_M=control_summary(inject_s1_double_prime(rows, ys=(2.41,))),
        S2=control_summary(inject_s2(rows)),
        null_free=control_summary(inject_free_null(rows)),
    )
    free0 = free_tables(("free_L4_aaaa.json",))
    s0 = [stage0.load_chain(f, 100) for f in sorted((N1DIR / "S0" / "L4").glob("L4_y2_*.npz"))]
    b0 = next(r for r in s0 if r["h"] == 0)
    fr0 = free_lookup(free0, 4, 4, "aaaa", 0.0)
    s4 = []
    for r in sorted(s0, key=lambda r: r["h"]):
        fr = free_lookup(free0, 4, 4, "aaaa", r["h"])
        row = dict(h=r["h"], chi_L_over_free=r["chi_L"][0] / fr["chi_L_sum"], GL_over_free=r["GL_p0"][0] / fr["GL_p0"])
        if r["h"] > 0:
            v = r["mL"][0] / b0["mL"][0] / (fr["mL_eff"][1] / fr0["mL_eff"][1])
            e = v * np.hypot(r["mL"][1] / r["mL"][0], b0["mL"][1] / b0["mL"][0])
            row.update(r_L=v, err=e, ok=bool(abs(v - 1) <= max(2 * e, 0.03)))
        row["ok_chi"] = bool(0.5 <= row["chi_L_over_free"] <= 2)
        row["ok_g"] = bool(0.7 <= row["GL_over_free"] <= 1.3)
        s4.append(row)
    bl = free_tables(FREE_S1)
    gate = []
    for L, Lt, bc in ((6, 6, "aaaa"), (8, 8, "aaaa"), (6, 12, "aaaa"), (6, 6, "pppa")):
        f0 = free_lookup(bl, L, Lt, bc, 0.0)
        t = Lt // 2 - 1
        for h in (0.5, 1.0, 2.0):
            f = free_lookup(bl, L, Lt, bc, h)
            if f is not None:
                gate.append(
                    dict(
                        L=L,
                        Lt=Lt,
                        bc=bc,
                        h=h,
                        GL_p0=f["GL_p0"],
                        dmL=f["mL_eff"][t] - f0["mL_eff"][t],
                        mL_free=f["mL_eff"][t],
                    )
                )
    res["S4"] = dict(rows=s4, gate=gate)
    res["selftest"] = dict(ok=stage0.selftest())
    return res


# ------------------------------------------------------------ alpha_L readout
def readout(blocks4, p1, p3):
    """blocks4: the four compressed corner blocks G(+p1), G(-p1), G(+p3), G(-p3) of one taste subspace."""
    E1, O1, fM = parity_split(blocks4[0], blocks4[1])
    E3, O3, _ = parity_split(blocks4[2], blocks4[3])
    return dict(
        alpha=odd_exponent(O1, O3, p1, p3),
        f_M=fM,
        normO1=float(np.linalg.norm(O1)),
        normO3=float(np.linalg.norm(O3)),
        normE1=float(np.linalg.norm(E1)),
    )


def alpha_file(f):
    """The alpha readout of one chain: ensemble average of the corner blocks first, then the light / heavy
    compression; alpha = ln(|O(p3)|/|O(p1)|)/ln(sin p3/sin p1); 10-block jackknife inflated by sqrt(2 tau_eff) of the
    per-trajectory sigma^2 series; free (sigma = 0) and Dirac-mass references; the floor (smallest free Dirac mass that
    moves alpha_L by twice the chain's error) and the label-swap distance |alpha_L - alpha_R| / sigma."""
    z = np.load(f, allow_pickle=False)
    ref = np.load(f.parents[1] / f"{z['ref']}.npz", allow_pickle=False)
    p1, p3 = float(z["p0s"][0]), float(z["p0s"][2])
    PL = np.kron(z["PLB"], np.eye(2))
    PR = np.kron(z["PRB"], np.eye(2))
    perr = max(
        np.abs(PL @ PL - PL).max(),
        np.abs(PR @ PR - PR).max(),
        abs(np.trace(PL).real - 16),
        abs(np.trace(PR).real - 16),
        np.abs(PL @ PR).max(),
    )
    out = dict(
        file=f.name,
        L=int(z["L"]),
        Lt=int(z["Lt"]),
        y=float(z["y"]),
        h=float(z["h"]),
        n=int(z["n"]),
        nb=int(z["block_means_L"].shape[0]),
        p1=p1,
        p3=p3,
        proj_err=float(perr),
        hook_identity=float(z["hook_identity"]),
        base=z["base"].tolist(),
    )
    s2 = z["sigma2_traj"]
    tauB = tau_int(s2, wmax=len(s2) // 10)
    D = float(np.median(np.diff(z["sel_traj"])))
    te = max(0.5, tauB / D)
    infl = float(np.sqrt(2 * te)) if te > 0.5 else 1.0
    out.update(tau_B=tauB, tau_eff=te, infl=infl)
    for side in ("L", "R"):
        bm = z[f"block_means_{side}"]
        nb = bm.shape[0]
        r = readout(bm.mean(0), p1, p3)
        jk = [readout((bm.sum(0) - bm[i]) / (nb - 1), p1, p3) for i in range(nb)]
        for k in ("alpha", "f_M", "normO1"):
            vals = np.array([j[k] for j in jk])
            r[k + "_err"] = float(np.sqrt((nb - 1) / nb * np.sum((vals - vals.mean()) ** 2))) * infl
        r["normO1_2sinp1"] = r["normO1"] * 2 * np.sin(p1)
        out[f"rd_{side}"] = r
    out["free"] = {f"rd_{s}": readout(ref[f"free_{s}"], p1, p3) for s in ("L", "R")}
    out["masses"] = ref["masses"].tolist()
    out["massed_L"] = [readout(m4, p1, p3)["alpha"] for m4 in ref["massed_L"]]
    thr = 2 * out["rd_L"]["alpha_err"]
    out["floor_m"] = next(
        (m for m, a in zip(out["masses"], out["massed_L"], strict=True) if a - out["free"]["rd_L"]["alpha"] > thr), None
    )
    out["floor_thr"] = thr
    dl = out["rd_L"]["alpha"] - out["rd_R"]["alpha"]
    out["swap_sigma"] = float(abs(dl) / np.hypot(out["rd_L"]["alpha_err"], out["rd_R"]["alpha_err"]))
    out["all_norms"] = z["all_norms"]
    out["sel_traj"] = z["sel_traj"]
    return out


def alpha_records(tag="S1"):
    """alpha readouts of every chain of data/derived/n1stage/alpha/<tag>/ (references in alpha/ref_*.npz)."""
    return [alpha_file(f) for f in sorted((N1DIR / "alpha" / tag).glob("*.npz"))]


def corner_blocks4(D, shape, bc):
    """The four corner blocks G_B(+-p1), G_B(+-p3) at the base momentum (p1 = the lowest time momentum, p3 the next)."""
    from ..corner import CornerBlocks, p0_loop

    cb = CornerBlocks(D, shape, 2)
    loop, base = p0_loop(shape, bc)
    Lt = shape[-1]
    return np.stack([cb.block(np.concatenate([base, [loop[i]]])) for i in (0, Lt - 1, 1, Lt - 2)])


# ------------------------------------------------------------ P_c series (K8)
def blocked10(x, nb=10):
    """(mean, error) over 10 blocks of a series (columns kept)."""
    x = np.asarray(x, float)
    x = x.reshape(len(x), -1)
    k = len(x) // nb
    B = x[: k * nb].reshape(nb, k, -1).mean(1)
    return B.mean(0), B.std(0, ddof=1) / np.sqrt(nb)


def sigma_channel_rows():
    """Per stage-1 aaaa chain (cut at trajectory 100, 10 blocks, no error inflation): S(pi), |Sigma_stag|, the
    light pair-channel midpoint cosh mass (delete-one-block jackknife), chi_L and chi_L(p_min)/chi_L(0)."""
    rows = {}
    for sub, Lt in (("L8", 8), ("L6x12", 12), ("L6", 6)):
        for f in sorted((N1DIR / "S1" / sub).glob("*.npz")):
            d, m = open_chain(f)
            tsel = np.arange(len(d["ts_dH"])) >= 100
            fsel = d["ts_cfg_traj"] >= 100
            Spi, Sab = blocked10(d["ts_S_pi"][tsel]), blocked10(d["ts_Sigma_stag_abs"][tsel])
            CL = d["ts_CL_t"][fsel]
            nb = 10
            k = len(CL) // nb
            Bm = CL[: k * nb].reshape(nb, k, -1).mean(1)
            t = Lt // 2 - 1
            mc = cosh_mass_point(Bm.mean(0), t, Lt, hi=6.0)
            jk = np.array([cosh_mass_point(np.delete(Bm, b, 0).mean(0), t, Lt, hi=6.0) for b in range(nb)])
            emc = np.sqrt((nb - 1) / nb * np.sum((jk - jk.mean()) ** 2))
            chi, pm = blocked10(d["ts_chi_L_sum"][fsel]), blocked10(d["ts_chi_L_sum_pmin"][fsel])
            rows[(sub, m["y"], m.get("h10", 0.0))] = dict(
                Spi=(Spi[0][0], Spi[1][0]),
                Sab=(Sab[0][0], Sab[1][0]),
                mc=(mc, emc),
                chi=(chi[0][0], chi[1][0]),
                peak=pm[0][0] / chi[0][0],
                V=m["L"] ** 3 * Lt,
            )
    return rows
