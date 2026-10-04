"""Stage 1c of the N1 test: two independent 8^4 replicas per h at y = 3.0 (h = 1, 2) and the small-h onset of the light
pair-channel gap at P_c on 6^4 (h = 0.25, 0.5, 0.75, with independent replicas at h = 0.25, 0.5).

Rows are the stage-1c chains after the cut at trajectory 100 (90 measurements per 8^4 chain, 225 per 6^4 chain). The
estimators, the error model and the y = 3.0 clauses are those of the stage-1b analysis (masspairing.analysis.stage1b:
read_row, verdict_y3, fit_m2, the injections S2'/S3'/S6'), unchanged.

Fixed before any stage-1c observable was printed:
  pooling   union of the members' measurements; each member is split into 10 contiguous blocks (np.array_split, no
            row dropped; blocks never straddle members); every estimator from the pooled means (sum of sums / sum of
            counts); sigma by a delete-one-block jackknife over all blocks, var = (N - 1)/N sum_b infl_m(b)^2
            (theta_b - mean)^2, infl_m the member's own error-model inflation of that series (the CL midpoint for
            masses, chi_L for the peak ratio, the series' own for means). The bare jackknife (infl = 1) is kept beside
            it (``*_bare``). A one-member pool reproduces read_row.
  (A)       the stage-1b clauses on (i) each replica, (ii) the pool rA u rB (the verdict line), (iii) the pool with the
            stage-1b rows of the extended chain (labelled); the 6^4 side is the stage-1b new row (h = 1: L6, h = 2: L5).
            h = 2: the partial outcome "growth relative to free" is CONFIRMED iff e - e_free - 2 sigma > 0 on (ii),
            else NOT REPRODUCED; h = 1: "resolved" / "not resolved" by the same rule.
  consistency  O in {m (midpoint cosh mass), <chi_L_sum>, <GL_p0>}, each member with its own error-model sigma:
            chi^2 against the inverse-variance mean, p = chi2.sf; CONSISTENT iff p > 0.01 for every O in the 3-member
            test {rA, rB, stage-1b}; rA vs rB failing makes the pooled verdict UNDECIDED. Control: O_rB shifted by
            3 sigma_c (sigma_c = sqrt(sigma_rA^2 + sigma_rB^2)) away from the inverse-variance mean of {rA, stage-1b}
            must give p <= 0.01 in the 3-member and the rA-vs-rB test, every O.
  half-split   m on the two halves (5 blocks each, x the chain's CL inflation) and the error-model half-pulls of
            chi_L, GL_p0; |pull| > 3 flags.
  (B) F1    m(h)^2 - m0^2 = (c h^p)^2 (fit_m2) on h in {0.25, 0.5, 0.75, 1} with m = cosh mass at t = 2 on CL_t,
            m0 = the stored h = 0 chain; sigma_p by the weighted delete-one-block jackknife over every chain's blocks
            (h = 0 included; each block x its chain's CL inflation). I = [p - 2 sigma_p, p + 2 sigma_p]; bands
            quadratic [1.6, 2.4], linear [0.7, 1.3] ("meets" = closed-interval overlap); UNDECIDED if I meets both
            bands, chi^2/dof > 3 or sigma_p > 0.3. Labelled: F1' (+ the stored h = 0.5 chain), F2 (h <= 0.75),
            F1f (free-normalised). The replicas at h = 0.25, 0.5 are pooled iff they pass a chi^2 test (dof 1,
            p > 0.01 for every O). Control: each entry's jackknife values shifted rigidly to a synthetic centre (real
            per-block scatter, real weights): quadratic m^2 = m0^2 + (0.3 h^2)^2, linear m^2 = m0^2 + (0.3 h)^2,
            saturating m = m0 + 0.5 (1 - e^-h).
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import chi2 as CHI2

from . import stage1b as B
from .n1stage import N1DIR, free_tables, jackknife, open_chain

S1C = N1DIR / "S1c"
CH_A = {  # tag: (file under data/derived/n1stage/S1c, h); 8^4, y = 3.0
    "rA_h2": ("L8_h2_rA/L8_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", 2.0),
    "rB_h2": ("L8_h2_rB/L8_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", 2.0),
    "rA_h1": ("L8_h1_rA/L8_y3_k-0.01_g0_g60_h1_chiral_bcaaaa.npz", 1.0),
    "rB_h1": ("L8_h1_rB/L8_y3_k-0.01_g0_g60_h1_chiral_bcaaaa.npz", 1.0),
}
CH_B = {  # tag: (file, h); 6^4, P_c
    "B2": ("L6/L6_y2.41_k-0.01_g0_g60_h0.25_chiral_bcaaaa.npz", 0.25),
    "B1": ("L6/L6_y2.41_k-0.01_g0_g60_h0.5_chiral_bcaaaa.npz", 0.5),
    "B3": ("L6/L6_y2.41_k-0.01_g0_g60_h0.75_chiral_bcaaaa.npz", 0.75),
    "B2r": ("L6_r2/L6_y2.41_k-0.01_g0_g60_h0.25_chiral_bcaaaa.npz", 0.25),
    "B1r": ("L6_r2/L6_y2.41_k-0.01_g0_g60_h0.5_chiral_bcaaaa.npz", 0.5),
}
MD5_DELIVERED = {  # md5 of the chain files as delivered by the stage-1c runs
    "rA_h2": "692e96aa9a116b0c013dd0acdb829460",
    "rB_h2": "e1888471a93bd3b367407d81c1dec623",
    "rA_h1": "3e1b5e0be072fc850893aa9abf951505",
    "rB_h1": "706c2416910d70737700db95c85a7301",
    "B1": "d4b130953d3a821eb9a9de7c1c312ee7",
    "B2": "08675156b011742ee0e3d080a5dc3e73",
    "B3": "394c44350e67648926fd42aea5bbec0d",
    "B1r": "0e4fdc4b1ed10f6ccea595adb229e32a",
    "B2r": "4da6af63679c76ea3c6c769cbd059eea",
}
SEEDS = {"rA_h2": 1101, "rB_h2": 1202, "rA_h1": 1303, "rB_h1": 1404}
SEEDS.update({"B1": 2505, "B2": 2606, "B3": 2707, "B2r": 2808, "B1r": 2909})
ONE_B = {2.0: "R1", 1.0: "R3"}  # the stage-1b extensions at y = 3.0 on 8^4 (rows at trajectory >= 1000)
SIX_B = {2.0: "L5", 1.0: "L6"}  # the 6^4 side of e(h), b2, b3 (stage-1b new rows)
FREE_1C = "free_L6_aaaa_h0.25_0.5_0.75.json"
BANDS = dict(quad=(1.6, 2.4), lin=(0.7, 1.3))
NB_POOL = 10
OBS = ("m", "chi_L", "GL_p0")
fmt = B.fmt
to_json = B.to_json


def baselines():
    """The free baselines: the 6^4 file at h in {0.25, 0.5, 0.75} first, then the stage-1b files."""
    return free_tables([FREE_1C]) + free_tables(B.FREE_1B)


# ------------------------------------------------------------ members and pooling
def member(path, tag=None):
    """Per-member arrays of the selection 'new' (cut 100; 1000 for the stage-1b extensions)."""
    d, meta = open_chain(path)
    keep, tmask, lo, hi = B.selection(d, "new")
    L, Lt = meta["L"], meta.get("Lt") or meta["L"]
    ser = dict(
        CL=np.asarray(d["ts_CL_t"], float)[keep],
        chi_L=np.asarray(d["ts_chi_L_sum"], float)[keep],
        chi_L_pmin=np.asarray(d["ts_chi_L_sum_pmin"], float)[keep],
        GL_p0=np.asarray(d["ts_GL_p0"], float)[keep],
        phi_R=np.asarray(d["ts_phi_f"], float)[keep].sum(1),
    )
    return dict(tag=tag, path=str(path), L=L, Lt=Lt, n=len(keep), ser=ser)


def pool_blocks(members, key, infl_key, rows_by_tag):
    """(sum, count, inflation) per block over all members (10 contiguous blocks each, np.array_split)."""
    out = []
    for mb in members:
        x = mb["ser"][key]
        r = rows_by_tag[mb["tag"]]
        infl = r["infl_CL"] if infl_key == "CL_mid" else r["stats"][infl_key]["infl"]
        for part in np.array_split(np.arange(len(x)), NB_POOL):
            out.append((x[part].sum(0), len(part), infl))
    return out


def jk_pooled(blocks_by_name, func, target=None):
    """Weighted delete-one-block jackknife over the blocks of pooled series (every name shares the blocking).
    func(means dict) -> float. Returns (value, sigma inflated, sigma bare); target shifts the jackknife values
    rigidly to a synthetic centre."""
    names = list(blocks_by_name)
    nbk = len(blocks_by_name[names[0]])
    S = {n: sum(b[0] for b in blocks_by_name[n]) for n in names}
    N = {n: sum(b[1] for b in blocks_by_name[n]) for n in names}
    full = func({n: S[n] / N[n] for n in names})
    vals, infl = [], []
    for i in range(nbk):
        vals.append(func({n: (S[n] - blocks_by_name[n][i][0]) / (N[n] - blocks_by_name[n][i][1]) for n in names}))
        infl.append(blocks_by_name[names[0]][i][2])
    vals, infl = np.array(vals, float), np.array(infl, float)
    if target is not None:
        vals = vals - full + target
        full = target
    dev2 = (vals - vals.mean()) ** 2 * (nbk - 1) / nbk
    return float(full), float(np.sqrt(np.sum(dev2 * infl**2))), float(np.sqrt(np.sum(dev2)))


def pooled_row(members, rows_by_tag, label):
    """A read_row-shaped dict of the pooled members (central values and sigma) for verdict_y3 and the tests."""
    r0 = rows_by_tag[members[0]["tag"]]
    L, Lt = members[0]["L"], members[0]["Lt"]
    t = Lt // 2 - 1
    cl = pool_blocks(members, "CL", "CL_mid", rows_by_tag)
    row = dict(
        tag=label,
        sel="new",
        lat=r0["lat"],
        y=r0["y"],
        h=r0["h"],
        L=L,
        Lt=Lt,
        V=r0["V"],
        bc=r0["bc"],
        n=sum(m["n"] for m in members),
        members=[m["tag"] for m in members],
        pooled=True,
    )
    v, s, sb = jk_pooled({"c": cl}, lambda M: B.coshmass(M["c"], t, Lt))
    row["m"], row["m_bare"] = (v, s), (v, sb)
    mc = {}
    for tt in range(Lt // 2):
        vv, ss, _ = jk_pooled({"c": cl}, lambda M, tt=tt: B.coshmass(M["c"], tt, Lt))
        mc[str(tt)] = (vv, ss)
    row["mcosh_t"] = mc
    for key in ("chi_L", "chi_L_pmin", "GL_p0", "phi_R"):
        v, s, sb = jk_pooled({"x": pool_blocks(members, key, key, rows_by_tag)}, lambda M: float(M["x"]))
        row[key], row[key + "_bare"] = (v, s), (v, sb)
    v, s, sb = jk_pooled(
        {
            "a": pool_blocks(members, "chi_L", "chi_L", rows_by_tag),
            "b": pool_blocks(members, "chi_L_pmin", "chi_L", rows_by_tag),
        },
        lambda M: float(M["a"] / M["b"]),
    )
    row["peak"], row["peak_bare"] = (v, s), (v, sb)
    CLm = sum(b[0] for b in cl) / sum(b[1] for b in cl)
    CLe = (
        np.sqrt(
            sum((rows_by_tag[m["tag"]]["n"] ** 2) * np.asarray(rows_by_tag[m["tag"]]["CL_t_err"]) ** 2 for m in members)
        )
        / row["n"]
    )
    mf, Af, c2 = B.cosh_fit3(CLm, CLe, Lt)
    row["fit3"] = dict(
        m=mf,
        err=float("nan"),
        A=Af,
        chi2=c2,
        dof=1,
        accepted=None,
        note="pooled: central only, errors from the members' blocked CL_t errors",
    )
    row["CL_t"] = CLm.tolist()
    row["free"] = r0["free"]
    for key in ("Spi", "Sab"):  # bosonic, recorded only: trajectory-weighted mean, sigma in quadrature
        w = np.array([rows_by_tag[m["tag"]]["ntraj_sel"] for m in members], float)
        x = np.array([rows_by_tag[m["tag"]][key] for m in members], float)
        row[key] = (float(np.sum(w * x[:, 0]) / w.sum()), float(np.sqrt(np.sum((w * x[:, 1]) ** 2)) / w.sum()))
    row["_cl_blocks"] = cl
    return row


# ------------------------------------------------------------ reading
def read_all():
    """(free baselines, stage-1/1b rows, stage-1c rows by tag (+ the stage-1b rows used), pooling members)."""
    bl = baselines()
    rows1b = B.read_all(bl)
    tagrow = {}
    for tag, (f, _) in {**CH_A, **CH_B}.items():
        tagrow[tag] = B.read_row(S1C / f, "new", bl, tag=tag)
    for tag in ("R1", "R3", "L5", "L6"):
        tagrow[tag] = next(r for r in rows1b.values() if r.get("tag") == tag and r["sel"] == "new")
        r = tagrow[tag]
        tagrow[tag + "_stored"] = B.R(rows1b, r["lat"], r["y"], r["h"], "stored")
    mem = {tag: member(S1C / f, tag) for tag, (f, _) in {**CH_A, **CH_B}.items()}
    for tag in ("R1", "R3"):
        mem[tag] = member(N1DIR / "S1b" / B.CHAINS_1B[tag][0], tag)
    return bl, rows1b, tagrow, mem


# ------------------------------------------------------------ (A): the clauses
def rows_with(rows1b, rep):
    """Copy of rows1b with ('L8', 3.0, h, 'new') replaced by rep[h]; the stage-1b rows kept under 'new1b'."""
    Rc = dict(rows1b)
    for h, r in rep.items():
        key = ("L8", 3.0, h, "new")
        Rc[("L8", 3.0, h, "new1b")] = rows1b[key]
        Rc[key] = dict(r)
    return Rc


def clauses(rows1b, rep):
    v = B.verdict_y3(rows_with(rows1b, rep), "new")
    v.pop("pooled_usable", None)
    return v


def score_A(rows1b, tagrow, mem):
    out = {}
    out["per_replica"] = {
        rep: clauses(rows1b, {2.0: tagrow[f"{rep}_h2"], 1.0: tagrow[f"{rep}_h1"]}) for rep in ("rA", "rB")
    }
    pooled = {h: pooled_row([mem[f"rA_h{h:g}"], mem[f"rB_h{h:g}"]], tagrow, f"rA∪rB_h{h:g}") for h in (1.0, 2.0)}
    out["pooled_rows"] = pooled
    out["pooled"] = clauses(rows1b, pooled)
    p1b = {
        h: pooled_row([mem[f"rA_h{h:g}"], mem[f"rB_h{h:g}"], mem[ONE_B[h]]], tagrow, f"rA∪rB∪{ONE_B[h]}new_h{h:g}")
        for h in (1.0, 2.0)
    }
    out["pooled_with_1b_rows"] = p1b
    out["pooled_with_1b"] = clauses(rows1b, p1b)
    one, ref = pooled_row([mem["rA_h2"]], tagrow, "check"), tagrow["rA_h2"]
    out["pool_identity"] = dict(
        m=(one["m"], ref["m"]),
        chi_L=(one["chi_L"], ref["chi_L"]),
        peak=(one["peak"], ref["peak"]),
        ok=bool(
            np.allclose(
                [one["m"][0], one["m"][1], one["chi_L"][0], one["peak"][0], one["peak"][1]],
                [ref["m"][0], ref["m"][1], ref["chi_L"][0], ref["peak"][0], ref["peak"][1]],
                rtol=1e-9,
            )
            and abs(one["chi_L"][1] / ref["chi_L"][1] - 1) < 1e-9
        ),
    )
    return out


# ------------------------------------------------------------ (A): consistency
def chi2_test(vals):
    """vals: list of (O, sigma). chi^2 against the inverse-variance mean, dof = n - 1."""
    x = np.array([v[0] for v in vals], float)
    s = np.array([v[1] for v in vals], float)
    w = 1 / s**2
    mw = float(np.sum(w * x) / np.sum(w))
    c2 = float(np.sum(w * (x - mw) ** 2))
    dof = len(x) - 1
    return dict(chi2=c2, dof=dof, p=float(CHI2.sf(c2, dof)), mean_w=mw, ok=bool(CHI2.sf(c2, dof) > 0.01))


def consistency_A(tagrow, pooled_rows, shift=None):
    """The replica-consistency tests per h; shift: optional {h: {O: delta}} added to rB's O (the control)."""
    out = {}
    for h in (2.0, 1.0):
        rA, rB, r1 = tagrow[f"rA_h{h:g}"], tagrow[f"rB_h{h:g}"], tagrow[ONE_B[h]]
        rs, pp = tagrow[ONE_B[h] + "_stored"], pooled_rows[h]
        res = {}
        for O in OBS:
            b = (rB[O][0] + (shift[h][O] if shift else 0.0), rB[O][1])
            three, ab = chi2_test([rA[O], b, r1[O]]), chi2_test([rA[O], b])
            pv = (pp[O][0] + (shift[h][O] / 2 if shift else 0.0), pp[O][1])  # the pooled mean moves by delta/2
            res[O] = dict(
                values=dict(rA=rA[O], rB=b, one_b_new=r1[O], stored=rs[O], pooled=pv),
                three=three,
                rA_vs_rB=ab,
                pooled_vs_1b=chi2_test([pv, r1[O]]),
                four_with_stored=chi2_test([rA[O], b, r1[O], rs[O]]),
            )
        three_ok = all(res[O]["three"]["ok"] for O in OBS)
        ab_ok = all(res[O]["rA_vs_rB"]["ok"] for O in OBS)
        if not ab_ok:
            reading = "replicas disagree: only the per-replica clauses are read; the pooled row carries no verdict"
        elif three_ok:
            reading = f"CONSISTENT: the stage-1b extension agrees with the independent replicas at h = {h:g}"
        else:
            reading = "the stage-1b extension is inconsistent with independent chains; pooled-with-1b not used"
        out[str(h)] = dict(obs=res, consistent=bool(three_ok), rA_vs_rB_ok=bool(ab_ok), reading=reading)
        out[str(h)]["member_1b"] = ONE_B[h]
    return out


def c2_control(tagrow, pooled_rows):
    """A 3 sigma_c shift into rB's O (away from the inverse-variance mean of rA and the stage-1b row) must fail."""
    shift = {}
    for h in (2.0, 1.0):
        rA, rB, r1 = tagrow[f"rA_h{h:g}"], tagrow[f"rB_h{h:g}"], tagrow[ONE_B[h]]
        shift[h] = {}
        for O in OBS:
            w = np.array([1 / rA[O][1] ** 2, 1 / r1[O][1] ** 2])
            ob = float((w[0] * rA[O][0] + w[1] * r1[O][0]) / w.sum())
            s = np.sign(rB[O][0] - ob) or 1.0
            sc = float(np.hypot(rA[O][1], rB[O][1]))
            shift[h][O] = float(s * 3 * sc)
    c = consistency_A(tagrow, pooled_rows, shift)
    hs = (2.0, 1.0)
    ok = all(
        c[str(h)]["obs"][O]["three"]["p"] <= 0.01 and c[str(h)]["obs"][O]["rA_vs_rB"]["p"] <= 0.01
        for h in hs
        for O in OBS
    )
    return dict(
        shift=shift,
        p_three={str(h): {O: c[str(h)]["obs"][O]["three"]["p"] for O in OBS} for h in hs},
        p_ab={str(h): {O: c[str(h)]["obs"][O]["rA_vs_rB"]["p"] for O in OBS} for h in hs},
        ok=bool(ok),
    )


def half_split(row, mb):
    """m on the two halves (5 blocks each, x the chain's CL inflation) and the half-pulls of chi_L, GL_p0."""
    CL = mb["ser"]["CL"]
    n, Lt = len(CL), mb["Lt"]
    t = Lt // 2 - 1
    hv = [jackknife(lambda c: B.coshmass(c, t, Lt), [part], 5, row["infl_CL"]) for part in (CL[: n // 2], CL[n // 2 :])]
    pm = (hv[0][0] - hv[1][0]) / np.hypot(hv[0][1], hv[1][1])
    pulls = dict(m=float(pm), chi_L=row["stats"]["chi_L"]["half_pull"], GL_p0=row["stats"]["GL_p0"]["half_pull"])
    return dict(m_halves=hv, pulls=pulls, flag=bool(any(abs(p) > 3 for p in pulls.values())))


def verdict_A(score, cons):
    out = {}
    for h in (2.0, 1.0):
        d = score["pooled"]["rows"][str(h)]
        e, s = d["e_minus_free"]
        lower = e - 2 * s
        if not cons[str(h)]["rA_vs_rB_ok"]:
            line = "UNDECIDED (replicas disagree)"
        elif h == 2.0:
            line = "CONFIRMED" if lower > 0 else "NOT REPRODUCED"
        else:
            line = "growth at h = 1 resolved" if lower > 0 else "not resolved"
        cls = "(b)" if d["b"] else ("(c)" if d["c"] else ("(a)" if d["a"] else "none"))
        out[str(h)] = dict(
            e_minus_free=(e, s),
            lower_2sigma=lower,
            line=line,
            b_fires=bool(d["b"]),
            cls_pooled_h=cls,
            pooled_with_1b_used=bool(cons[str(h)]["consistent"]),
        )
    out["class_pooled"] = score["pooled"]["class"]
    return out


# ------------------------------------------------------------ (B): onset
def b_entries(rows1b, tagrow, mem, use_pooled):
    """name -> dict(h, blocks [(sum, count, inflation)], sigma, origin, m) of the onset fit."""
    ent = {}

    def blocks_of(r=None, members=None):
        if members is not None:
            return pool_blocks(members, "CL", "CL_mid", tagrow)
        Bm = r["CL_blocks"]  # stored rows: the read_row blocking (20 blocks at n >= 150)
        k = r["n"] // len(Bm)
        return [(Bm[i] * k, k, r["infl_CL"]) for i in range(len(Bm))]

    r0 = B.R(rows1b, "L6", 2.41, 0.0, "stored")
    ent["m0"] = dict(h=0.0, blocks=blocks_of(r0), sigma=r0["m"][1], origin="stage 1 (h = 0)", m=r0["m"])
    for h, a, b in ((0.25, "B2", "B2r"), (0.5, "B1", "B1r")):
        if use_pooled.get(h, True):
            pr = pooled_row([mem[a], mem[b]], tagrow, f"{a}∪{b}")
            ent[f"m{h:g}"] = dict(
                h=h, blocks=blocks_of(members=[mem[a], mem[b]]), sigma=pr["m"][1], origin="stage 1c, pooled", m=pr["m"]
            )
        else:
            r = tagrow[a]
            ent[f"m{h:g}"] = dict(
                h=h, blocks=blocks_of(members=[mem[a]]), sigma=r["m"][1], origin="stage 1c, one chain", m=r["m"]
            )
    r = tagrow["B3"]
    ent["m0.75"] = dict(h=0.75, blocks=blocks_of(members=[mem["B3"]]), sigma=r["m"][1], origin="stage 1c", m=r["m"])
    r1 = B.R(rows1b, "L6", 2.41, 1.0, "stored")
    ent["m1"] = dict(h=1.0, blocks=blocks_of(r1), sigma=r1["m"][1], origin="stage 1", m=r1["m"])
    rs = B.R(rows1b, "L6", 2.41, 0.5, "stored")
    ent["m0.5s"] = dict(h=0.5, blocks=blocks_of(rs), sigma=rs["m"][1], origin="stage 1", m=rs["m"])
    return ent


def fit_onset(ent, names, free_norm=None, targets=None):
    """m^2 - m0^2 = (c h^p)^2 on the entries ``names`` (m0 always in); sigma_p by the weighted jackknife over every
    block of every entry (m0 included). free_norm: name -> factor; targets: name -> synthetic central m."""
    Lt, t = 6, 2
    fac = {n: (free_norm[n] if free_norm else 1.0) for n in ["m0"] + names}

    def mval(n, c):
        return B.coshmass(c, t, Lt) * fac[n]

    full = {
        n: mval(n, sum(b[0] for b in ent[n]["blocks"]) / sum(b[1] for b in ent[n]["blocks"])) for n in ["m0"] + names
    }
    sh = {n: ((targets[n] - full[n]) if targets else 0.0) for n in full}
    hs = [ent[n]["h"] for n in names]
    sig = [ent[n]["sigma"] * fac[n] for n in names]

    def fit(v):
        return B.fit_m2(hs, [v[n] for n in names], sig, v["m0"])

    v0 = {n: full[n] + sh[n] for n in full}
    p, c2, chi2 = fit(v0)
    var = 0.0
    for n in full:
        bl = ent[n]["blocks"]
        S, N, nb = sum(b[0] for b in bl), sum(b[1] for b in bl), len(bl)
        jv = []
        for i in range(nb):
            v = dict(v0)
            v[n] = mval(n, (S - bl[i][0]) / (N - bl[i][1])) + sh[n]
            jv.append(fit(v)[0])
        jv, infl = np.array(jv), np.array([b[2] for b in bl])
        var += float(np.sum((jv - jv.mean()) ** 2 * infl**2) * (nb - 1) / nb)
    sp = float(np.sqrt(var))
    dof = len(names) - 2
    red = chi2 / dof if dof > 0 else float("nan")
    I = (p - 2 * sp, p + 2 * sp)
    meets = lambda band: bool(I[0] <= band[1] and I[1] >= band[0])
    mq, ml = meets(BANDS["quad"]), meets(BANDS["lin"])
    if (mq and ml) or (dof > 0 and red > 3) or sp > 0.3 or not np.isfinite(sp):
        verdict = "UNDECIDED"
    elif mq:
        verdict = "quadratic-like onset"
    elif ml:
        verdict = "linear"
    else:
        verdict = "other"
    return dict(
        names=names,
        hs=hs,
        m=dict(v0),
        sig=dict(zip(names, sig, strict=True)),
        p=(p, sp),
        c=float(np.sqrt(max(c2, 0))),
        chi2=chi2,
        dof=dof,
        chi2_dof=red,
        I=I,
        meets_quad=mq,
        meets_lin=ml,
        verdict=verdict,
    )


def consistency_B(tagrow):
    """The 6^4 replica test (dof 1) at h = 0.25, 0.5."""
    out = {}
    for h, a, b in ((0.25, "B2", "B2r"), (0.5, "B1", "B1r")):
        res = {}
        for O in OBS:
            x, y = tagrow[a][O], tagrow[b][O]
            c2 = (x[0] - y[0]) ** 2 / (x[1] ** 2 + y[1] ** 2)
            res[O] = dict(orig=x, r2=y, chi2=float(c2), p=float(CHI2.sf(c2, 1)), ok=bool(CHI2.sf(c2, 1) > 0.01))
        out[str(h)] = dict(obs=res, consistent=bool(all(v["ok"] for v in res.values())), pair=[a, b])
    return out


def score_B(rows1b, tagrow, mem, bl):
    cons = consistency_B(tagrow)
    use = {0.25: cons["0.25"]["consistent"], 0.5: cons["0.5"]["consistent"]}
    ent = b_entries(rows1b, tagrow, mem, use)
    ent_pooled = b_entries(rows1b, tagrow, mem, {0.25: True, 0.5: True})
    F1n = ["m0.25", "m0.5", "m0.75", "m1"]
    out = dict(consistency=cons, used_pooled=use)
    out["F1"] = fit_onset(ent, F1n)
    if not all(use.values()):
        out["F1_pooled_labelled"] = fit_onset(ent_pooled, F1n)
    out["F1prime"] = fit_onset(ent, F1n + ["m0.5s"])
    out["F2"] = fit_onset(ent, ["m0.25", "m0.5", "m0.75"])

    def mfree(h):
        return B.coshmass(B.free_lookup(bl, 6, 6, "aaaa", h)["CL_t"], 2, 6)

    fn = {n: mfree(0.0) / mfree(ent[n]["h"]) for n in ent}
    out["free_factors"] = fn
    out["F1f"] = fit_onset(ent, F1n, free_norm=fn)
    old = [
        r
        for r in free_tables(B.FREE_1B)
        if r["L"] == 6 and r["Lt"] == 6 and r["bc"] == "aaaa" and abs(r["h"] - 0.5) < 1e-12
    ]
    out["free_h05_agreement"] = (
        float(abs(old[0]["chi_L_sum"] - B.free_lookup(bl, 6, 6, "aaaa", 0.5)["chi_L_sum"])) if old else None
    )
    a, s = ent["m0.5"]["m"], ent["m0.5s"]["m"]
    pull = float((a[0] - s[0]) / np.hypot(a[1], s[1]))
    out["pull_new_vs_stored_h05"] = dict(new=a, stored=s, pull=pull, flag=bool(abs(pull) > 3))
    curve = [dict(h=0.0, m=ent["m0"]["m"], origin=ent["m0"]["origin"])]
    for n in ("m0.25", "m0.5", "m0.75"):
        curve.append(dict(h=ent[n]["h"], m=ent[n]["m"], origin=ent[n]["origin"]))
    curve.append(dict(h=0.5, m=s, origin="stage 1 [beside]"))
    for h in (1.0, 2.0):
        curve.append(dict(h=h, m=B.R(rows1b, "L6", 2.41, h, "stored")["m"], origin="stage 1"))
    for h in (1.5, 3.0):
        curve.append(dict(h=h, m=B.R(rows1b, "L6", 2.41, h, "new")["m"], origin="stage 1b"))
    for c in curve:
        c["m_free_norm"] = c["m"][0] * mfree(0.0) / mfree(c["h"])
    out["curve"] = curve
    main = sorted([c for c in curve if "[beside]" not in c["origin"]], key=lambda c: c["h"])
    m0 = main[0]["m"][0]
    loc = []
    for a_, b_ in zip(main[1:-1], main[2:], strict=True):
        qa, qb = a_["m"][0] ** 2 - m0**2, b_["m"][0] ** 2 - m0**2
        k = float(0.5 * np.log(qb / qa) / np.log(b_["h"] / a_["h"])) if qa > 0 and qb > 0 else None
        loc.append(dict(step=f"{a_['h']:g}->{b_['h']:g}", kappa_local=k))
    out["local_exponents"] = loc
    mono = []
    for a_, b_ in zip(main[:-1], main[1:], strict=True):
        if b_["h"] > 1.0:
            break
        p_ = float((b_["m"][0] - a_["m"][0]) / np.hypot(a_["m"][1], b_["m"][1]))
        mono.append(dict(step=f"{a_['h']:g}->{b_['h']:g}", pull=p_, rises=bool(b_["m"][0] > a_["m"][0])))
    out["monotonicity"] = mono
    m0v = ent["m0"]["m"][0]
    ctl = {}
    for name, fn_ in (
        ("quadratic", lambda h: np.sqrt(m0v**2 + (0.3 * h**2) ** 2)),
        ("linear", lambda h: np.sqrt(m0v**2 + (0.3 * h) ** 2)),
        ("saturating", lambda h: m0v + 0.5 * (1 - np.exp(-h))),
    ):
        tg = {n: (m0v if n == "m0" else float(fn_(ent[n]["h"]))) for n in ["m0"] + F1n}
        ctl[name] = fit_onset(ent, F1n, targets=tg)
    ctl["ok"] = bool(
        ctl["quadratic"]["verdict"] == "quadratic-like onset"
        and ctl["linear"]["verdict"] == "linear"
        and ctl["saturating"]["verdict"] != "quadratic-like onset"
    )
    out["F1_controls"] = ctl
    return out


# ------------------------------------------------------------ integrity
def integrity(tagrow):
    """Per stage-1c chain: delivered md5 (recorded in the derived file's meta), seed, h, trajectories, duplicates,
    rows after the cut, acceptance, and the kill A(h) < 0 beyond 3 sigma."""
    out = {}
    for tag, (f, h) in {**CH_A, **CH_B}.items():
        d, meta = open_chain(S1C / f)
        ct = np.asarray(d["ts_cfg_traj"])
        r = tagrow[tag]
        rec = dict(
            file=meta["source"],
            md5=meta["source_md5"],
            ntraj=len(d["ts_dH"]),
            n_rows=len(ct),
            n_cut=r["n"],
            dup=int(len(ct) - len(np.unique(ct))),
            seed=meta["seed"],
            start=meta["start"],
            h=float(meta["h10"]),
            y=float(meta["y"]),
            ferm_every=meta["ferm_every"],
            mts=meta["mts"],
            acc_min50=r["acc_min50"],
            A=r["A"],
            tau_B_over_D=r["tau_B_over_D"],
            infl_CL=r["infl_CL"],
            git=str(meta.get("git_hash"))[:7],
        )
        rec["ok"] = bool(
            rec["md5"] == MD5_DELIVERED[tag]
            and rec["dup"] == 0
            and rec["seed"] == SEEDS[tag]
            and rec["ntraj"] == 1000
            and abs(rec["h"] - h) < 1e-12
            and rec["n_cut"] == (90 if tag.startswith("r") else 225)
            and rec["acc_min50"] >= 0.5
            and not (r["A"][0] < -3 * r["A"][1])
        )
        out[tag] = rec
    out["all_ok"] = bool(all(v["ok"] for v in out.values()))
    return out


# ------------------------------------------------------------ controls on the pooled (A) rows
def controls_A(rows1b, score):
    Rc = rows_with(rows1b, score["pooled_rows"])
    s2, s3, s6 = B.inject_s2(Rc), B.inject_s3(Rc), B.inject_free_null(Rc)
    d = s2["rows"]["2.0"]
    return {
        "S2'": dict(cls=s2["class"], b1=d["b1"], b2=d["b2"], b3=d["b3"], ok=bool(s2["class"] == "(b)" and d["b"])),
        "S3'": dict(cls=s3["class"], ok=bool(s3["class"] == "(c)")),
        "S6'": dict(
            cls=s6["verdict"]["class"],
            ok=bool(s6["verdict"]["class"] not in ("(a)", "(b)") and not s6["verdict"]["rows"]["2.0"]["b"]),
        ),
    }


def sabotage_free_scaling(rows1b, score):
    """The pooled 8^4 chi_L at h = 2 replaced by its free-scaling value chi_L(6^4) (V8/V6)^e_free: the growth line
    must read NOT REPRODUCED."""
    pr = copy.deepcopy(score["pooled_rows"])
    r6, r8 = B.R(rows1b, "L6", 3.0, 2.0, "new"), pr[2.0]
    _, ef = B.exponent(r6, r8)
    null_chi = r6["chi_L"][0] * (r8["V"] / r6["V"]) ** ef
    r8["chi_L"] = (null_chi, r8["chi_L"][1] * null_chi / r8["chi_L"][0])
    d = clauses(rows1b, pr)["rows"]["2.0"]
    line = "CONFIRMED" if d["e_minus_free"][0] - 2 * d["e_minus_free"][1] > 0 else "NOT REPRODUCED"
    return dict(row=d, line=line)


# ------------------------------------------------------------ recorded (post hoc, no verdict)
def sensitivity(rows1b, score, res_B, ent):
    """(1) e - e_free of the pooled 8^4 row with the 6^4 side swapped from the stage-1b new rows to the stored and the
    pooled stored + new rows, and the 6^4 chi_L factor that would bring the h = 2 lower edge to 0; (2) a Gaussian
    Monte Carlo of sigma_p of F1 (points and m0 drawn with their sigma; 4000 draws, fixed seed)."""
    out = {"six_side_swap": {}}
    for sel in ("new", "stored", "pooled"):
        Rc = rows_with(rows1b, score["pooled_rows"])
        for h in (1.0, 2.0):
            Rc[("L6", 3.0, h, "new6keep")] = rows1b[("L6", 3.0, h, "new")]
            Rc[("L6", 3.0, h, "new")] = rows1b[("L6", 3.0, h, sel)]
        v = B.verdict_y3(Rc, "new")
        out["six_side_swap"][sel] = {
            h: dict(
                e_minus_free=d["e_minus_free"],
                lower2=d["e_minus_free"][0] - 2 * d["e_minus_free"][1],
                b3=d["b3"],
                cls=v["class"],
            )
            for h, d in v["rows"].items()
        }
    d2 = out["six_side_swap"]["new"]["2.0"]
    out["chi6_factor_to_flip_h2"] = float(np.exp(d2["lower2"] * np.log(4096 / 1296)))
    r6, r6s = rows1b[("L6", 3.0, 2.0, "new")], rows1b[("L6", 3.0, 2.0, "stored")]
    out["L5_chi_L_new_vs_stored"] = dict(new=r6["chi_L"], stored=r6s["chi_L"], ratio=r6["chi_L"][0] / r6s["chi_L"][0])
    F1 = res_B["F1"]
    names, hs = F1["names"], F1["hs"]
    sig = [F1["sig"][n] for n in names]
    m0, s0 = ent["m0"]["m"]
    rng = np.random.default_rng(20261004)
    ps = []
    for _ in range(4000):
        mm = [F1["m"][n] + rng.normal() * s for n, s in zip(names, sig, strict=True)]
        mz = m0 + rng.normal() * s0
        ps.append(B.fit_m2(hs, mm, sig, mz)[0])
    ps = np.array(ps)
    out["F1_mc"] = dict(
        p_jk=F1["p"],
        mc_std=float(ps.std()),
        mc_percentiles_2p5_16_50_84_97p5=np.percentile(ps, [2.5, 16, 50, 84, 97.5]).tolist(),
        frac_p_ge_1p6=float((ps >= 1.6).mean()),
        frac_p_in_lin=float(((ps >= 0.7) & (ps <= 1.3)).mean()),
    )
    return out


# ------------------------------------------------------------ driver
def strip(o):
    """Drop block arrays and private keys (JSON output)."""
    if isinstance(o, dict):
        return {
            k: strip(v)
            for k, v in o.items()
            if not (isinstance(k, str) and (k.startswith("_") or k.endswith("_blocks") or k == "blocks"))
        }
    if isinstance(o, list | tuple):
        return [strip(v) for v in o]
    return o


def analyse(with_sensitivity=False):
    """(rows1b, tagrow, results) of the stage-1c analysis; results['sensitivity'] only if asked (about 20 s)."""
    bl, rows1b, tagrow, mem = read_all()
    res = dict(integrity=integrity(tagrow))
    sA = score_A(rows1b, tagrow, mem)
    res["A_scores"] = sA
    cons = consistency_A(tagrow, sA["pooled_rows"])
    res["A_consistency"] = cons
    res["A_verdict"] = verdict_A(sA, cons)
    res["A_half_split"] = {tag: half_split(tagrow[tag], mem[tag]) for tag in CH_A}
    res["A_controls"] = controls_A(rows1b, sA)
    res["A_controls"]["C2_control"] = c2_control(tagrow, sA["pooled_rows"])
    res["A_controls"]["all_ok"] = bool(all(v["ok"] for v in res["A_controls"].values() if isinstance(v, dict)))
    res["B"] = score_B(rows1b, tagrow, mem, bl)
    res["B_half_split"] = {tag: half_split(tagrow[tag], mem[tag]) for tag in CH_B}
    if with_sensitivity:
        ent = b_entries(rows1b, tagrow, mem, res["B"]["used_pooled"])
        res["sensitivity"] = sensitivity(rows1b, sA, res["B"], ent)
    return rows1b, tagrow, res


def rows_json(tagrow):
    """The stage-1c rows (and the stage-1b rows used) without blocks and error-model details."""
    return to_json(
        strip(
            {
                k: {kk: vv for kk, vv in v.items() if not kk.endswith("_blocks") and kk != "stats"}
                for k, v in tagrow.items()
            }
        )
    )
