"""Stage 0 of the N1 test (4^4, 4^3x8 aaaa and one 4^4 pppa pair; 26 chains): the pre-registered reader, its
(a)/(b)/(c) verdict logic with the synthetic self-test, and the stage-0 gates G1-G4.

Reader (fixed before stage 0 was read): m_L, m_R = ln C(t*)/C(t*+1) of the P_L / P_R time-slice correlators at
t* = L_t/2 - 1 (a log-ratio proxy of the pair-channel gap), chi_L = ts_chi_L_sum, GL_p0, GR_p0, chi_R = ts_chi_f_sum,
phi_R = sum over flavours of phi_f, phi_T_R; the first min(100, n/4) measurements cut; errors from 20 blocks. Every
number is read next to the free value at the same L, L_t, bc, h.

Verdict per y (two volumes; the order's criteria):
  (a) keeps its gap:  r_L(h) >= 0.7 - 2 sigma for every h at the largest L, e + 2 sigma < 0.5 for the volume exponent
                      e = d ln chi_L / d ln V, and GL_p0/free <= 0.8 + 2 sigma;
  (b) re-Higgses:     e - 2 sigma > 0.5 at some h;
  (c) goes gapless:   r_L + 2 sigma < 0.3 with GL_p0/free >= 0.8 - 2 sigma at some h;
  M trigger:          r_L falling monotonically with r_L(2) + 2 sigma < 0.7 at both volumes, V-stable log-slope.
r_L(h) = [m_L(h)/m_L(0)] / [m_L^free(h)/m_L^free(0)].
"""

from __future__ import annotations

import numpy as np

from .n1stage import N1DIR, exponent, free_lookup, free_tables, open_chain

FREE_S0 = ("free_L4_aaaa.json", "free_L4x8_aaaa.json", "free_L4_pppa.json")
S0_DIRS = ("L4", "L4x8", "L4pppa")


def blocked(x, nb=20):
    """(mean, error) over nb blocks (naive error when fewer than nb samples)."""
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    if n < nb:
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
    b = x[:n].reshape(nb, -1).mean(1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb))


def eff_mass_series(C, t):
    """Per-measurement log-ratio ln C(t)/C(t+1) from an (n, Lt) series."""
    C = np.asarray(C, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(C[:, t] / C[:, t + 1])


def load_chain(path, cut=100):
    """The reader's row of one chain: parameters and (value, error) of every series."""
    d, meta = open_chain(path)
    L, Lt = meta["L"], meta.get("Lt") or meta["L"]
    bc = "".join("a" if b == -1 else "p" for b in meta["bc"])
    row = dict(y=meta["y"], L=L, Lt=Lt, bc=bc, h=meta["h10"], V=L**3 * Lt, file=str(path), n=int(len(d["ts_dH"])))
    t = Lt // 2 - 1
    nf = int(np.asarray(d["ts_ferm_flag"], bool).sum())
    c0 = max(0, min(cut, nf // 4))
    row["mL"] = blocked(eff_mass_series(d["ts_CL_t"], t)[c0:])
    row["mR"] = blocked(eff_mass_series(d["ts_CR_t"], t)[c0:])
    for key, name in (
        ("ts_chi_L_sum", "chi_L"),
        ("ts_GL_p0", "GL_p0"),
        ("ts_GR_p0", "GR_p0"),
        ("ts_chi_f_sum", "chi_R"),
        ("ts_phi_T_R", "phi_T_R"),
        ("ts_phi_T_L", "phi_T_L"),
    ):
        row[name] = blocked(np.asarray(d[key], float)[c0:])
    row["phi_R"] = blocked(np.asarray(d["ts_phi_f"], float).sum(1)[c0:])
    return row


def verdict(rows, y, Ls=(6, 8)):
    """(verdict, M trigger, notes) of the pre-registered reader on rows with y, L, h, V, mL, chi_L, GL_p0 (value,
    error), GL_p0_free and optionally mL_free."""
    R = [r for r in rows if abs(r["y"] - y) < 1e-9]
    Lmax = max(Ls)

    def get(L, h):
        return next((r for r in R if r["L"] == L and abs(r["h"] - h) < 1e-9), None)

    def free_ratio(r, b):
        fr = (r.get("mL_free", np.nan) / b.get("mL_free", np.nan)) if b.get("mL_free", 0) else 1.0
        return 1.0 if not np.isfinite(fr) or fr <= 0 else fr

    hs = sorted({r["h"] for r in R if r["h"] > 0})
    notes = []
    base = {L: get(L, 0.0) for L in Ls}
    if any(b is None for b in base.values()):
        return "UNDECIDED", False, ["missing h = 0 partner"]
    a_ok, b_flag, c_flag = True, False, False
    rL = {}
    for h in hs:
        r6, r8 = get(Ls[0], h), get(Lmax, h)
        if r8 is None:
            continue
        m0, e0 = base[Lmax]["mL"]
        m, e = r8["mL"]
        fr = free_ratio(r8, base[Lmax])
        ratio = m / m0 / fr if m0 > 0 else np.nan
        err = abs(ratio) * np.sqrt((e / m) ** 2 + (e0 / m0) ** 2) if m > 0 and m0 > 0 else np.nan
        rL[h] = (ratio, err)
        g, ge = r8["GL_p0"]
        gf = r8["GL_p0_free"]
        gr = g / gf
        if not (ratio >= 0.7 - 2 * err):
            a_ok = False
        if ratio + 2 * err < 0.3 and gr >= 0.8 - 2 * ge / gf:
            c_flag = True
        if not (gr <= 0.8 + 2 * ge / gf):
            a_ok = False
        if r6 is not None:
            ex, exe = exponent(r6["chi_L"][0], r6["chi_L"][1], r8["chi_L"][0], r8["chi_L"][1], r6["V"], r8["V"])
            notes.append(
                f"h={h}: r_L={ratio:.3f}±{err:.3f}, chi_L exponent = {ex:.2f}±{exe:.2f}, GL_p0/free = {gr:.2f}"
            )
            if not (ex + 2 * exe < 0.5):
                a_ok = False
            if ex - 2 * exe > 0.5:
                b_flag = True
    if b_flag:
        v = "(b) re-Higgs"
    elif c_flag:
        v = "(c) gapless"
    elif a_ok and rL:
        v = "(a) keeps the SMG gap"
    else:
        v = "UNDECIDED"
    M = False
    if len(hs) >= 3 and all(get(L, h) is not None for L in Ls for h in hs):

        def fall(L):
            vals = []
            for h in hs:
                m0, e0 = base[L]["mL"]
                m, e = get(L, h)["mL"]
                fr = free_ratio(get(L, h), base[L])
                v_ = m / m0 / fr
                vals.append((v_, v_ * np.sqrt((e / m) ** 2 + (e0 / m0) ** 2) if m > 0 else np.nan))
            mono = all(
                vals[i + 1][0] <= vals[i][0] + 2 * np.hypot(vals[i][1], vals[i + 1][1]) for i in range(len(vals) - 1)
            )
            last = vals[-1][0] + 2 * vals[-1][1] < 0.7
            lh = np.log(hs[-1] / hs[-2])
            er = np.log(vals[-1][0] / vals[-2][0]) / lh
            ere = np.sqrt((vals[-1][1] / vals[-1][0]) ** 2 + (vals[-2][1] / vals[-2][0]) ** 2) / lh
            return mono and last, er, ere

        f6, e6, ee6 = fall(Ls[0])
        f8, e8, ee8 = fall(Lmax)
        M = f6 and f8 and abs(e6 - e8) < 2 * np.hypot(ee6, ee8)
    return v, M, notes


def selftest(verbose=False):
    """Synthetic rows: a flat gap must read (a), an injected 1/(1 + 3h) fall (c) with the M trigger, an injected
    V-proportional chi_L (b). Returns True iff all three are flagged as such."""
    hs = [0.0, 0.2, 0.5, 1.0, 2.0]
    Ls = (6, 8)

    def rows_for(mL_fn, chi_fn, gl_fn):
        rows = []
        for L in Ls:
            V = L**4
            for h in hs:
                rows.append(
                    dict(
                        y=2.41,
                        L=L,
                        Lt=L,
                        bc="aaaa",
                        h=h,
                        V=V,
                        mL=(mL_fn(h, L), 0.01),
                        chi_L=(chi_fn(h, V), 0.02 * chi_fn(h, V)),
                        GL_p0=(gl_fn(h, L), 0.01),
                        GL_p0_free=1.0,
                    )
                )
        return rows

    base = rows_for(lambda h, L: 0.5, lambda h, V: 0.3, lambda h, L: 0.3)
    fall = rows_for(lambda h, L: 0.5 / (1 + 3 * h), lambda h, V: 0.3, lambda h, L: 0.3 + 0.9 * h / (1 + h))
    ssb = rows_for(lambda h, L: 0.5, lambda h, V: 0.3 * V / 1296.0 * (1 if h > 0.3 else 1296.0 / V), lambda h, L: 0.3)
    out = {}
    for name, rows in (("base (a)", base), ("injected 1/h fall", fall), ("injected SSB", ssb)):
        out[name] = verdict(rows, 2.41, Ls)[:2]
        if verbose:
            print(f"  {name:20s} -> {out[name][0]}, M trigger {out[name][1]}")
    return bool(
        out["base (a)"][0].startswith("(a)")
        and not out["base (a)"][1]
        and out["injected 1/h fall"][0].startswith("(c)")
        and out["injected 1/h fall"][1]
        and out["injected SSB"][0].startswith("(b)")
    )


# ------------------------------------------------------------ the stage-0 rows
def stage0_files():
    return sorted(p for sub in S0_DIRS for p in (N1DIR / "S0" / sub).glob("*.npz"))


def stage0_rows(cut=100):
    """The 26 stage-0 rows with their free values and the on-configuration asymmetry A(h) = sum_t (CL_t - CR_t) /
    sum_t CL_t (blocked directly; the first 75 measurements cut, as the reader's min(100, n/4))."""
    free = free_tables(FREE_S0)
    rows = []
    for f in stage0_files():
        r = load_chain(f, cut)
        fr = free_lookup(free, r["L"], r["Lt"], r["bc"], r["h"])
        assert fr is not None, f
        t = r["Lt"] // 2 - 1
        r.update(
            mL_free=fr["mL_eff"][t],
            mR_free=fr["mR_eff"][t],
            chi_L_free=fr["chi_L_sum"],
            GL_p0_free=fr["GL_p0"],
            chi_R_free=fr["chi_f_sum"],
            phi_R_free=float(np.sum(fr["phi_f"])),
            phi_T_R_free=fr["phi_T_R"],
            A_free=(np.sum(fr["CL_t"]) - np.sum(fr["CR_t"])) / np.sum(fr["CL_t"]),
        )
        d, _ = open_chain(f)
        c0 = 75
        CL = np.asarray(d["ts_CL_t"], float)[c0:]
        CR = np.asarray(d["ts_CR_t"], float)[c0:]
        r["A"] = blocked((CL - CR).sum(1) / CL.sum(1))
        r["acc"] = float(np.mean(d["ts_accepted"]))
        r["taus"] = [
            tau_series(s)
            for s in (
                eff_mass_series(d["ts_CL_t"], t)[c0:],
                np.asarray(d["ts_chi_L_sum"], float)[c0:],
                np.asarray(d["ts_GL_p0"], float)[c0:],
                np.asarray(d["ts_phi_f"], float).sum(1)[c0:],
            )
        ]
        rows.append(r)
    return rows


def tau_series(x, wmax=60):
    """tau_int with the window capped at min(wmax, n/4) (the stage-0 readout of the series' autocorrelation)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    v = (x * x).mean()
    if v == 0:
        return 0.5
    rho = [(x[:-t] * x[t:]).mean() / v for t in range(1, min(wmax, len(x) // 4))]
    tau = 0.5
    for W in range(1, len(rho) + 1):
        tau = 0.5 + sum(rho[:W])
        if W >= 6 * tau:
            break
    return tau


def get(rows, y, L, Lt, h, bc="aaaa"):
    return next(
        (
            r
            for r in rows
            if abs(r["y"] - y) < 1e-9 and r["L"] == L and r["Lt"] == Lt and r["bc"] == bc and abs(r["h"] - h) < 1e-9
        ),
        None,
    )


def r_L(r, b):
    """(m_L(h)/m_L(0), error, free ratio) against the h = 0 row b."""
    v = r["mL"][0] / b["mL"][0]
    e = abs(v) * np.hypot(r["mL"][1] / r["mL"][0], b["mL"][1] / b["mL"][0])
    return v, e, r["mL_free"] / b["mL_free"]


def gate1(R):
    """G1 readability on 4^3x8: relative errors of m_L, m_R < 15 %, of chi_L < 20 %."""
    out = []
    for r in [r for r in R if r["Lt"] == 8]:
        e = (r["mL"][1] / abs(r["mL"][0]), r["mR"][1] / abs(r["mR"][0]), r["chi_L"][1] / abs(r["chi_L"][0]))
        out.append((r["y"], r["h"], e, e[0] < 0.15 and e[1] < 0.15 and e[2] < 0.20))
    return out


def gate2(R):
    """G2 SYM control at y = 2.0 (4^4 aaaa): r_L within 2 sigma of the free ratio, chi_L/free in [0.5, 2],
    GL_p0/free in [0.7, 1.3]. Rows (h, r, err, free ratio, pull, chi_L/free, GL_p0/free, ok)."""
    sel = [r for r in R if abs(r["y"] - 2.0) < 1e-9 and r["L"] == 4 and r["Lt"] == 4 and r["bc"] == "aaaa"]
    b = next(r for r in sel if r["h"] == 0)
    out = []
    for r in sorted([r for r in sel if r["h"] > 0], key=lambda r: r["h"]):
        v, e, fr = r_L(r, b)
        pull = (v - fr) / e
        cr = r["chi_L"][0] / r["chi_L_free"]
        gr = r["GL_p0"][0] / r["GL_p0_free"]
        out.append((r["h"], v, e, fr, pull, cr, gr, abs(pull) <= 2 and 0.5 <= cr <= 2 and 0.7 <= gr <= 1.3))
    return out


def gate3(R):
    """G3 heavy half responds: m_R(h) - m_R(0) > 0 at > 3 sigma for h >= 0.5 and phi_R/h > 0 at > 5 sigma."""
    out = []
    for L, Lt in ((4, 4), (4, 8)):
        for y in (2.0, 2.41, 3.0):
            b = get(R, y, L, Lt, 0.0)
            if b is None:
                continue
            sel = [r for r in R if abs(r["y"] - y) < 1e-9 and r["L"] == L and r["Lt"] == Lt and r["bc"] == "aaaa"]
            for r in sorted([r for r in sel if r["h"] > 0], key=lambda r: r["h"]):
                dm = r["mR"][0] - b["mR"][0]
                edm = np.hypot(r["mR"][1], b["mR"][1])
                pr, epr = r["phi_R"][0] / r["h"], r["phi_R"][1] / r["h"]
                out.append(
                    dict(
                        L=L,
                        Lt=Lt,
                        y=y,
                        h=r["h"],
                        dm=dm,
                        edm=edm,
                        sig=dm / edm,
                        fdm=r["mR_free"] - b["mR_free"],
                        pr=pr,
                        epr=epr,
                        psig=pr / epr,
                        pratio=pr / (r["phi_R_free"] / r["h"]),
                        okm=(dm / edm > 3) if r["h"] >= 0.5 else True,
                        okp=pr / epr > 5,
                    )
                )
    return out


def gate4(R):
    """G4 (recorded): the composite/elementary ratio phi_T_R/phi_R on every aaaa row with h > 0."""
    out = []
    for r in sorted([r for r in R if r["h"] > 0 and r["bc"] == "aaaa"], key=lambda r: (r["y"], r["Lt"], r["h"])):
        ratio = r["phi_T_R"][0] / r["phi_R"][0]
        er = abs(ratio) * np.hypot(r["phi_T_R"][1] / r["phi_T_R"][0], r["phi_R"][1] / r["phi_R"][0])
        out.append(
            dict(
                y=r["y"],
                L=r["L"],
                Lt=r["Lt"],
                h=r["h"],
                ratio=ratio,
                err=er,
                pT=r["phi_T_R"][0] / r["h"],
                pTfree=r["phi_T_R_free"] / r["h"],
                psig=r["phi_R"][0] / r["phi_R"][1],
            )
        )
    return out
