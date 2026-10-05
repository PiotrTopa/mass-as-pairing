"""Direction of the critical-line shift under the explicit heavy mass h C_chi, and the light half against free (K8.4,
K8.5, K8.7).

Rows are read with the stage-1b reader (``stage1b.read_row``: cut ts_cfg_traj >= 100, error model C0) from
  * the stored stage-1 / new stage-1b / stage-1c chains at P_c = (2.41, -0.01) and at y = 3.0
    (data/derived/n1stage/S1*),
  * three 6^4 aaaa chains at y = 2.0 (h = 0, 3; "SYM") and y = 3.0 (h = 3) (data/derived/n1stage/TH),
  * the stage-0 4^4 chains at y = 2.0, 2.41, 3.0 (data/derived/n1stage/S0/L4).
Observables per row: g = GL_p0/free (light single-fermion readout at p_min), c = chi_L/free (light pair susceptibility),
S(pi) (staggered sigma structure factor), O4 (epsilon-vertex density), m (light pair-channel midpoint cosh mass).

Pre-registered tests (fixed before their chains ran):
  P1  on 6^4, with f_O = (O_Pc - O_SYM)/(O_SMG - O_SYM) for O = g, c: "toward SYM" iff f_g and f_c fall from h = 0 to 3
      beyond 2 sigma and dS = |S_Pc(3) - S_SMG(0)| - |S_Pc(3) - S_SYM(3)| > 2 sigma; "toward SMG" iff the opposite; else
      "mixed / undecided" (Gaussian Monte Carlo over independent chains, seed 7, 200 000 draws).
  P3  (y, h) = (3.0, 3) is "critical-like" iff g, c > 0.1, "deep SMG" iff g, c < 0.03.
"""

from __future__ import annotations

import copy

import numpy as np

from . import stage1b as B
from .n1stage import N1DIR, free_lookup, free_tables, jackknife_fit, open_chain

FREE_DIR = B.FREE_1B + ("free_L6_aaaa_h0.25_0.5_0.75.json",)
O4L_FILE = N1DIR / "o4l_series.npz"
HS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def hname(h):
    return "" if h == 0 else f"_h{h:g}"


FILES = {  # key: (derived path, selection)
    "Pc0": (N1DIR / "S1/L6/L6_y2.41_k-0.01_g0_g60_chiral_bcaaaa.npz", "stored"),
    "Pc3": (N1DIR / "S1b/L6/L6_y2.41_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "new"),
    "SMG0": (N1DIR / "S1/L6/L6_y3_k-0.01_g0_g60_chiral_bcaaaa.npz", "stored"),
    "SMG2": (N1DIR / "S1/L6/L6_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", "stored"),
    "SYM0": (N1DIR / "TH/sym6/L6_y2_k-0.01_g0_g60_chiral_bcaaaa.npz", "stored"),
    "SYM3": (N1DIR / "TH/sym6/L6_y2_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "stored"),
    "Y3H3": (N1DIR / "TH/y3h3/L6_y3_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "stored"),
}


def bl_tables():
    return free_tables(FREE_DIR)


# ------------------------------------------------------------ P1 / P3 (6^4)
def row(key, bl):
    p, sel = FILES[key]
    r = B.read_row(p, sel, bl, tag=key)
    return dict(
        key=key,
        y=r["y"],
        h=r["h"],
        n=r["n"],
        g=B.g_of(r),
        c=B.chi_over_free(r),
        S=r["Spi"],
        m=r["m"],
        O4=r["O4"],
        acc=r["acc_min50"],
        tau_B=r["tau_B"],
        m_free=r["free"]["m"],
    )


def mc(fun, rows, n=200000, seed=7):
    """Gaussian Monte Carlo over independent chains: fun(dict of sampled scalars) -> array; (mean, std)."""
    rng = np.random.default_rng(seed)
    samp = {k: rng.normal(v, e, n) for k, (v, e) in rows.items()}
    x = fun(samp)
    return float(np.mean(x)), float(np.std(x))


def p1_verdict(R):
    out = {}
    for O in ("g", "c"):
        vals = {k: R[k][O] for k in ("Pc0", "Pc3", "SMG0", "SYM0", "SYM3")}

        def f(s, h):
            if h == 3:
                return (s["Pc3"] - s["SYM3"]) / (s["SMG0"] - s["SYM3"])
            return (s["Pc0"] - s["SYM0"]) / (s["SMG0"] - s["SYM0"])

        out[f"f_{O}(0)"] = mc(lambda s, f=f: f(s, 0), vals)
        out[f"f_{O}(3)"] = mc(lambda s, f=f: f(s, 3), vals)
        out[f"df_{O}"] = mc(lambda s, f=f: f(s, 3) - f(s, 0), vals)
    vals = {k: R[k]["S"] for k in ("Pc3", "SMG0", "SYM3")}
    out["dS"] = mc(lambda s: np.abs(s["Pc3"] - s["SMG0"]) - np.abs(s["Pc3"] - s["SYM3"]), vals)  # > 0: closer to SYM
    sym = all(out[f"df_{O}"][0] + 2 * out[f"df_{O}"][1] < 0 for O in ("g", "c")) and out["dS"][0] - 2 * out["dS"][1] > 0
    smg = all(out[f"df_{O}"][0] - 2 * out[f"df_{O}"][1] > 0 for O in ("g", "c")) and out["dS"][0] + 2 * out["dS"][1] < 0
    out["verdict"] = "toward SYM" if sym else ("toward SMG" if smg else "mixed / undecided")
    return out


def p1_controls(R):
    """Synthetic rows: P_c(3) halfway to SMG with S at the SMG value -> 'toward SMG'; P_c(3) at SYM(3) -> 'toward
    SYM'."""
    res = {}
    fake = dict(R)
    mid = {O: (0.5 * (R["Pc0"][O][0] + R["SMG0"][O][0]), R["Pc3"][O][1]) for O in ("g", "c")}
    fake["Pc3"] = dict(R["Pc3"], g=mid["g"], c=mid["c"], S=(R["SMG0"]["S"][0], R["Pc3"]["S"][1]))
    res["synthetic_toward_SMG"] = p1_verdict(fake)["verdict"]
    fake2 = dict(R)
    fake2["Pc3"] = dict(
        R["Pc3"],
        g=(R["SYM3"]["g"][0], R["Pc3"]["g"][1]),
        c=(R["SYM3"]["c"][0], R["Pc3"]["c"][1]),
        S=(R["SYM3"]["S"][0], R["Pc3"]["S"][1]),
    )
    res["synthetic_at_SYM"] = p1_verdict(fake2)["verdict"]
    return res


def p3_verdict(R):
    r = R["Y3H3"]
    g, c = r["g"][0], r["c"][0]
    if g > 0.1 and c > 0.1:
        v = "critical-like"
    elif g < 0.03 and c < 0.03:
        v = "deep SMG"
    else:
        v = "intermediate"
    out = dict(verdict=v, g=r["g"], c=r["c"], S=r["S"], O4=r["O4"], m=r["m"])
    for O in ("g", "c"):
        out[f"d{O}_from_h2"] = (float(r[O][0] - R["SMG2"][O][0]), float(np.hypot(r[O][1], R["SMG2"][O][1])))
    return out


def analyse_direction(bl=None):
    bl = bl or bl_tables()
    R = {k: row(k, bl) for k in FILES}
    return dict(rows=R, P1=p1_verdict(R), P1_controls=p1_controls(R), P3=p3_verdict(R))


def stage0_positions(bl=None):
    """4^4 stage-0 chains (y = 2.0, 2.41, 3.0; h = 0, 2): the positions f_g, f_c of P_c between y = 2.0 and 3.0."""
    bl = bl or bl_tables()
    B.LAT.setdefault((4, 4), "L4")
    r4 = {}
    for y, hh in ((2, 0), (2, 2), (2.41, 0), (2.41, 2), (3, 0)):
        rr = B.read_row(N1DIR / f"S0/L4/L4_y{y}_k-0.01_g0_g60{hname(hh)}_chiral_bcaaaa.npz", "stored", bl)
        r4[(y, hh)] = dict(g=B.g_of(rr)[0], c=B.chi_over_free(rr)[0])

    def f(O, hh):
        return (r4[(2.41, hh)][O] - r4[(2, hh)][O]) / (r4[(3, 0)][O] - r4[(2, hh)][O])

    return {(O, hh): f(O, hh) for O in ("g", "c") for hh in (0, 2)}


# ------------------------------------------------------------ the light half against free (K8.4)
def chain_files():
    """(lattice, y, h, derived path, selection, source label) of every row of the free comparison."""
    out = []
    for lat in ("L8", "L6", "L6x12"):
        for h in (0, 0.5, 1, 2):
            out.append(
                (
                    lat,
                    2.41,
                    h,
                    N1DIR / f"S1/{lat}/{lat}_y2.41_k-0.01_g0_g60{hname(h)}_chiral_bcaaaa.npz",
                    "stored",
                    "stage1",
                )
            )
        for h in (1.5, 3):
            out.append(
                (lat, 2.41, h, N1DIR / f"S1b/{lat}/{lat}_y2.41_k-0.01_g0_g60{hname(h)}_chiral_bcaaaa.npz", "new", "1b")
            )
    for h in (0.25, 0.5, 0.75):
        out.append(("L6", 2.41, h, N1DIR / f"S1c/L6/L6_y2.41_k-0.01_g0_g60_h{h:g}_chiral_bcaaaa.npz", "stored", "1c"))
    for key in ("SYM0", "SYM3", "SMG0", "SMG2", "Y3H3"):
        p, sel = FILES[key]
        out.append(("L6", None, None, p, sel, "TH:" + key))
    return out


def ratio_meff(row, fr):
    """-ln[R(t+1)/R(t)] at t = L_t/2 - 1 with R = C/C_free; jackknife over the CL blocks x inflation."""
    Lt = row["Lt"]
    t = Lt // 2 - 1
    Cf = np.asarray(fr["CL_t"], float)

    def f(c):
        return -np.log((c[t + 1] / Cf[t + 1]) / (c[t] / Cf[t]))

    v, e = jackknife_fit({"c": dict(blocks=row["CL_blocks"], infl=row["infl_CL"], reduce=f)}, lambda v: {"x": v["c"]})
    return float(v["x"]), float(e["x"])


def free_compare(bl=None):
    """Rows of the free comparison: m, m_free, Rm = m/m_free (6^4, 8^4), dmeff, g, c, S(pi), O4."""
    bl = bl or bl_tables()
    rows = []
    for _lat, _y, _h, p, sel, src in chain_files():
        r = B.read_row(p, sel, bl)
        fr = free_lookup(bl, r["L"], r["Lt"], r["bc"], r["h"])
        m, mf = r["m"], r["free"]["m"]
        rows.append(
            dict(
                src=src,
                lat=r["lat"],
                y=r["y"],
                h=r["h"],
                n=r["n"],
                m=list(m),
                m_free=float(mf),
                Rm=[float(m[0] / mf), float(m[1] / mf)] if np.isfinite(mf) else None,
                dmeff=list(ratio_meff(r, fr)),
                g=list(B.g_of(r)),
                c=list(B.chi_over_free(r)),
                S=list(r["Spi"]),
                O4=list(r["O4"]),
            )
        )
    return rows


def pc_rows(rows):
    """{(lat, h): row} of the P_c reference series (stage 1 and 1b) and {key: row} of the 6^4 TH rows."""
    pc = {(r["lat"], r["h"]): r for r in rows if r["src"] in ("stage1", "1b") and abs(r["y"] - 2.41) < 1e-9}
    ref = {r["src"][3:]: r for r in rows if r["src"].startswith("TH:")}
    return pc, ref


def crit_no_overshoot(pc):
    """6^4/8^4: m + 2 sigma < m_free at every h and m/m_free(3) - m/m_free(0) > 2 sigma. g, c on all three lattices:
    rise h = 0 -> 3 beyond 2 sigma, stay <= 1 + 2 sigma, and > 0.8 at h = 3."""
    res = {}
    for lat in ("L6", "L8"):
        R = [pc[(lat, h)]["Rm"] for h in HS]
        below = all(r[0] + 2 * r[1] < 1.0 for r in R)
        rise = R[-1][0] - R[0][0] > 2 * np.hypot(R[-1][1], R[0][1])
        res[lat + "_m"] = (below and rise, R)
    for lat in ("L6", "L8", "L6x12"):
        for O in ("g", "c"):
            a, b = pc[(lat, 0.0)][O], pc[(lat, 3.0)][O]
            good = (
                (b[0] - a[0] > 2 * np.hypot(a[1], b[1]))
                and all(pc[(lat, h)][O][0] - 2 * pc[(lat, h)][O][1] <= 1.0 for h in HS)
                and b[0] > 0.8
            )
            res[f"{lat}_{O}"] = (good, (a, b))
    return res


def sabotage_free(pc):
    """(flagged?, flagged?): a 6^4 h = 3 row 10 % above free; an 8^4 h = 3 row with the SMG value of g."""
    s1 = copy.deepcopy(pc)
    s1[("L6", 3.0)]["Rm"] = [1.10, s1[("L6", 3.0)]["Rm"][1]]
    s2 = copy.deepcopy(pc)
    s2[("L8", 3.0)]["g"] = [0.003, 0.001]
    return not crit_no_overshoot(s1)["L6_m"][0], not crit_no_overshoot(s2)["L8_g"][0]


# ------------------------------------------------------------ the epsilon vertex, h-matched (K8.7)
def o4_series(path, sel, o4l):
    """O4 and its light-doublet restriction O4L of one chain (C0 errors, the reader's selection and tau_B)."""
    d, meta = open_chain(path)
    keep, tmask, _, _ = B.selection(d, sel)
    n = len(keep)
    nb = 20 if n >= 150 else 10
    ct = np.asarray(d["ts_cfg_traj"])[keep]
    D = float(np.median(np.diff(ct)))
    ntr = int(tmask.sum())
    tauB = max(
        B.tau_int(np.asarray(d["ts_Sigma_stag_abs"], float)[tmask], wmax=ntr // 10),
        B.tau_int(np.asarray(d["ts_sigma2"], float)[tmask], wmax=ntr // 10),
    )
    out = dict(L=meta["L"], Lt=meta.get("Lt") or meta["L"], y=float(meta["y"]), h=float(meta["h10"]), n=n)
    for k, x in (("O4", np.asarray(d["ts_O4"], float)), ("O4L", o4l)):
        m, s, _ = B.err_model(np.asarray(x, float)[keep], nb, tauB / D)
        out[k] = (float(m), float(s))
    return out


def o4_rows():
    """{(L, Lt, h): row} for the P_c rows (stage 1, 1b, 1c; a later 6^4 row at the same h replaces an earlier one, as
    in the notebook) and {"TH:<key>": row} for the 6^4 TH rows."""
    z = np.load(O4L_FILE)
    R = {}
    for _lat, _y, _h, p, sel, src in chain_files():
        key = p.relative_to(N1DIR.parent).as_posix().replace("/", "__")
        o4l = z[key] if key in z.files else open_chain(p)[0]["ts_O4L"]
        r = o4_series(p, sel, o4l)
        r["src"] = src
        R[src if src.startswith("TH:") else (r["L"], r["Lt"], r["h"])] = r
    return R


def f_o4(pc, sym, smg):
    """Position (O4_Pc - O4_SYM)/(O4_SMG - O4_SYM) and its Gaussian-MC error (seed 3, 100 000 draws)."""
    v = (pc["O4"][0] - sym["O4"][0]) / (smg["O4"][0] - sym["O4"][0])
    rng = np.random.default_rng(3)
    n = 100000

    def s(r):
        return rng.normal(r["O4"][0], r["O4"][1], n)

    x = (s(pc) - s(sym)) / (s(smg) - s(sym))
    return float(v), float(x.std())


def toward_sym_o4(R, pc0, pc3):
    f0, e0 = f_o4(pc0, R["TH:SYM0"], R["TH:SMG0"])
    f3, e3 = f_o4(pc3, R["TH:SYM3"], R["TH:Y3H3"])
    return (f3 - f0) < -2 * np.hypot(e0, e3), (f0, e0, f3, e3)


# ------------------------------------------------------------ the SMG side (K8.5)
def smg_rows(bl=None):
    """y = 3.0: the stored stage-1 rows (h = 0, 0.5, 1, 2 on 8^4, 6^4, 6^3x12), the 6^4 (3.0, 3) row, and the four
    independent stage-1c 8^4 replicas."""
    bl = bl or bl_tables()
    rows = {}
    for lat in ("L8", "L6", "L6x12"):
        for h in (0, 0.5, 1, 2):
            r = B.read_row(N1DIR / f"S1/{lat}/{lat}_y3_k-0.01_g0_g60{hname(h)}_chiral_bcaaaa.npz", "stored", bl)
            rows[(lat, float(h))] = dict(g=B.g_of(r), c=B.chi_over_free(r), m=r["m"])
    r = B.read_row(FILES["Y3H3"][0], "stored", bl)
    rows[("L6", 3.0)] = dict(g=B.g_of(r), c=B.chi_over_free(r), m=r["m"])
    rep = {}
    for d in ("L8_h1_rA", "L8_h1_rB", "L8_h2_rA", "L8_h2_rB"):
        for p in sorted((N1DIR / "S1c" / d).glob("*.npz")):
            r = B.read_row(p, "stored", bl)
            rep[d] = dict(h=r["h"], g=B.g_of(r), c=B.chi_over_free(r), m=r["m"])
    return rows, rep


def monotone_toward_free(rows, lat, O, hs):
    """Every step non-decreasing within 2 sigma and an overall rise beyond 3 sigma."""
    a, b = rows[(lat, hs[0])][O], rows[(lat, hs[-1])][O]
    steps_ok = all(
        rows[(lat, hs[i + 1])][O][0]
        >= rows[(lat, hs[i])][O][0] - 2 * np.hypot(rows[(lat, hs[i])][O][1], rows[(lat, hs[i + 1])][O][1])
        for i in range(len(hs) - 1)
    )
    return steps_ok and (b[0] - a[0] > 3 * np.hypot(a[1], b[1])), (a, b)


def ref_series(rows, lat, key):
    """The P_c reference series (stored h in {0, 0.5, 1, 2}, new h in {1.5, 3}) of one stage-1b row quantity."""
    out = []
    for h in HS:
        r = B.ref(rows, lat, h)
        out.append(dict(O4=r["O4"][0], g=B.g_of(r)[0], c=B.chi_over_free(r)[0])[key])
    return out


# ------------------------------------------------------------ the direction test on 8^4 (K8.9)
FILES8 = {  # key: (derived path, selection)
    "Pc0": (N1DIR / "S1/L8/L8_y2.41_k-0.01_g0_g60_chiral_bcaaaa.npz", "stored"),
    "Pc3": (N1DIR / "S1b/L8/L8_y2.41_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "new"),
    "SMG0": (N1DIR / "S1/L8/L8_y3_k-0.01_g0_g60_chiral_bcaaaa.npz", "stored"),
    "SYM0": (N1DIR / "TH/sym8/L8_y2_k-0.01_g0_g60_chiral_bcaaaa.npz", "stored"),
    "SYM3": (N1DIR / "TH/sym8/L8_y2_k-0.01_g0_g60_h3_chiral_bcaaaa.npz", "stored"),
}


def p1prime(bl=None):
    """P1 applied unchanged to the 8^4 rows; data sufficiency (FULL >= 300 production trajectories per new chain,
    PARTIAL >= 200); amendment 1: v_X = O_X(8^4, h = 3) / O_X(6^4, h = 3) for X = P_c, SYM and O = g, c, "gap beyond
    SYM dressing" iff v_Pc < v_SYM by more than 2 sigma, else "free-like volume trend"."""
    bl = bl or bl_tables()
    R = {}
    for k, (p, sel) in FILES8.items():
        r = B.read_row(p, sel, bl, tag=k)
        d, _ = open_chain(p)
        R[k] = dict(
            key=k,
            y=r["y"],
            h=r["h"],
            n=r["n"],
            nprod=int(len(d["ts_dH"])),
            g=B.g_of(r),
            c=B.chi_over_free(r),
            S=r["Spi"],
            m=r["m"],
            O4=r["O4"],
            acc=r["acc_min50"],
            m_free=r["free"]["m"],
        )
    nps = [R[k]["nprod"] for k in ("SYM0", "SYM3")]
    suff = "FULL" if min(nps) >= 300 else ("PARTIAL" if min(nps) >= 200 else "NO RESULT")
    v = p1_verdict(R)
    dS = v["dS"]
    s_clause = "S toward SYM" if dS[0] - 2 * dS[1] > 0 else ("S toward SMG" if dS[0] + 2 * dS[1] < 0 else "S undecided")
    R6 = {k: row(k, bl) for k in ("Pc3", "SYM3")}
    am = {}
    for O in ("g", "c"):
        vP = mc(lambda s: s["a"] / s["b"], {"a": R["Pc3"][O], "b": R6["Pc3"][O]})
        vS = mc(lambda s: s["a"] / s["b"], {"a": R["SYM3"][O], "b": R6["SYM3"][O]})
        diff, err = vP[0] - vS[0], float(np.hypot(vP[1], vS[1]))
        verdict = "gap beyond SYM dressing" if diff < -2 * err else "free-like volume trend"
        am[O] = dict(v_Pc=vP, v_SYM=vS, diff=(diff, err), verdict=verdict)
    return dict(rows=R, sufficiency=suff, P1prime=v, S_clause=s_clause, controls=p1_controls(R), amendment1=am)
