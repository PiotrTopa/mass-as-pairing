#!/usr/bin/env python3
"""K8.1 -- anti-seesaw, tier 1: the partner-mass lemma (a Majorana mass on a Dirac partner can only lower the light
gap), and the stored stage-1 facts at P_c: the epsilon/sigma channel collapses onto its SMG value, the light
pair-channel midpoint cosh mass rises monotonically with h, no SSB of the light doublet, and part of the rise is
present at fixed sigma (quench of stored configurations). Reads data/derived/n1stage/S1, the free baselines and
data/configs/n1stage_quench_L6_y2.41_h0.npz (dense re-measurement of 4 configurations at two h; about 6-8 min)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis.n1stage import cosh_mass_point, free_tables, remeasure, stored_configs
from masspairing.claimcheck import Check

c = Check("K8.1")

# (1) lemma
rng = np.random.default_rng(199)
viol = 0
for _ in range(2000):
    D = rng.uniform(0.05, 3.0)
    M = np.sort(rng.uniform(0.0, 10.0, 2))
    lam = lambda m: np.sqrt(D * D + m * m / 4) - m / 2
    ev = lambda m: np.min(np.abs(np.linalg.eigvalsh(np.array([[0.0, D], [D, m]]))))
    viol += not (abs(lam(M[0]) - ev(M[0])) < 1e-12 and abs(lam(M[1]) - ev(M[1])) < 1e-12)
    viol += not (lam(M[1]) < lam(M[0]) <= D + 1e-12)
c.item(
    "lemma: lambda = sqrt(D^2 + M^2/4) - M/2 = min|eig [[0, D], [D, M]]| <= D and strictly decreasing in M (2000 "
    "random pairs; violations)",
    viol,
    viol == 0,
)
c.item(
    "lemma: d lambda/dM = M/(4 sqrt(D^2 + M^2/4)) - 1/2 < 0 (D = 0.1, 1, 5; M = 0, 0.5, 2, 50)",
    True,
    all((m / (4 * np.sqrt(d * d + m * m / 4)) - 0.5) < 0 for d in (0.1, 1, 5) for m in (0, 0.5, 2, 50)),
)

# (3) stored stage-1 data
rows = stage1.sigma_channel_rows()
for (sub, y, h), r in sorted(rows.items()):
    c.record(
        f"{sub} y={y} h={h}",
        f"S(pi) {r['Spi'][0]:.2f}({r['Spi'][1]:.2f}) |Sigma_stag| {r['Sab'][0]:.4f}({r['Sab'][1]:.4f}) "
        f"light pair-channel midpoint cosh mass {r['mc'][0]:.3f}({r['mc'][1]:.3f}) chi_L {r['chi'][0]:.2e} "
        f"chi_L(p_min)/chi_L(0) {r['peak']:.3f}",
    )
hs = (0.0, 0.5, 1.0, 2.0)
for sub in ("L8", "L6", "L6x12"):
    S = [rows[(sub, 2.41, h)]["Spi"] for h in hs]
    Ss = [rows[(sub, 3.0, h)]["Spi"] for h in hs]
    steps = [(S[i][0] - S[i + 1][0]) / np.hypot(S[i][1], S[i + 1][1]) for i in range(3)]
    tot = (S[0][0] - S[3][0]) / np.hypot(S[0][1], S[3][1])
    c.item(
        f"(i) {sub}: S(pi) at P_c falls (step pulls; no rise beyond 1 sigma, h = 1 -> 2 > 3 sigma); total fall h = 0 "
        "-> 2",
        (np.round(steps, 1).tolist(), tot),
        all(s > -1 for s in steps) and steps[2] > 3 and tot > 5,
        "{0[0]}; {0[1]:.0f} sigma",
    )
    c.item(
        f"(i) {sub}: S(pi) at P_c h = 0 -> 2 vs the SMG (y = 3.0, h = 0) value, within 15 %",
        (S[0][0], S[3][0], Ss[0][0]),
        abs(S[3][0] / Ss[0][0] - 1) < 0.15,
        "{0[0]:.1f} -> {0[1]:.2f} vs {0[2]:.2f}",
    )
    A = [rows[(sub, 2.41, h)]["Sab"][0] for h in hs]
    As = [rows[(sub, 3.0, h)]["Sab"][0] for h in hs]
    c.item(
        f"(i) {sub}: |Sigma_stag| at P_c h = 0 -> 2 vs SMG, within 15 %",
        (A[0], A[3], As[0]),
        abs(A[3] / As[0] - 1) < 0.15,
        "{0[0]:.4f} -> {0[1]:.4f} vs {0[2]:.4f}",
    )
    c.item(
        f"(i) {sub}: at y = 3.0 S(pi) and |Sigma_stag| h-independent within 15 %",
        True,
        max(abs(s[0] / Ss[0][0] - 1) for s in Ss) < 0.15 and max(abs(a / As[0] - 1) for a in As) < 0.15,
    )
for sub in ("L8", "L6", "L6x12"):
    mcs = [rows[(sub, 2.41, h)]["mc"] for h in hs]
    c.item(
        f"(ii) {sub}: light pair-channel midpoint cosh mass at P_c, h = 0, 0.5, 1, 2 (monotone rise)",
        [f"{m[0]:.3f}({m[1]:.3f})" for m in mcs],
        all(mcs[i + 1][0] > mcs[i][0] for i in range(3)),
    )
    ms = [rows[(sub, 3.0, h)]["mc"] for h in hs]
    c.item(
        f"(ii) {sub}: at y = 3.0 h-stable within 10 % (h = 0 -> 2)",
        ms[3][0] / ms[0][0] - 1,
        abs(ms[3][0] / ms[0][0] - 1) < 0.10,
        "{:+.1%}",
    )
m8, m82 = rows[("L8", 2.41, 1.0)]["mc"], rows[("L8", 2.41, 2.0)]["mc"]
c.item(
    "(ii) 8^4 P_c: the h = 1 -> 2 step",
    (m82[0] - m8[0]) / np.hypot(m8[1], m82[1]),
    (m82[0] - m8[0]) / np.hypot(m8[1], m82[1]) > 10,
    "{:.0f} sigma",
)
free = {}
for r in free_tables(stage1.FREE_S1):
    if r.get("bc") == "aaaa" and r["L"] >= 6:
        free[(r["L"], r.get("Lt", r["L"]), float(r["h"]))] = r
lnV = np.log(rows[("L8", 2.41, 0.0)]["V"] / rows[("L6", 2.41, 0.0)]["V"])
de, pk = [], []
for h in hs:
    e = np.log(rows[("L8", 2.41, h)]["chi"][0] / rows[("L6", 2.41, h)]["chi"][0]) / lnV
    ef = np.log(free[(8, 8, h)]["chi_L_sum"] / free[(6, 6, h)]["chi_L_sum"]) / lnV
    de.append(e - ef)
    for sub, L, Lt in (("L8", 8, 8), ("L6", 6, 6), ("L6x12", 6, 12)):
        pr = rows[(sub, 2.41, h)]["peak"]
        prf = free[(L, Lt, h)]["chi_L_sum_pmin"] / free[(L, Lt, h)]["chi_L_sum"]
        pk.append(pr / prf - 1)
c.item(
    "(iii) no SSB at P_c: e - e_free on (6^4, 8^4) at h = 0, 0.5, 1, 2 (< 0.1)",
    np.round(de, 3).tolist(),
    all(v < 0.1 for v in de),
)
c.item(
    "(iii) chi_L(p_min)/chi_L(0) relative to free at every h and lattice, max deviation (< 20 %)",
    max(abs(v) for v in pk),
    all(abs(v) < 0.2 for v in pk),
    "{:.1%}",
)

# (iv) quench at fixed sigma
z, ch = stored_configs("n1stage_quench_L6_y2.41_h0")
Lt = ch["Lt"]
b0 = remeasure(ch, z["fields"], 0.0)
ident = max(np.max(np.abs(b["CL_t"] - z["CL_t"][j])) for j, b in enumerate(b0))
c.item(
    "(iv) re-measurement of the stored 6^4 P_c h = 0 configurations reproduces the stored CL_t",
    ident,
    ident < 1e-10,
    "{:.1e}",
)
b2 = remeasure(ch, z["fields"], 2.0)
m0 = np.array([cosh_mass_point(b["CL_t"], Lt // 2 - 1, Lt, hi=6.0) for b in b0])
m2 = np.array([cosh_mass_point(b["CL_t"], Lt // 2 - 1, Lt, hi=6.0) for b in b2])
diff = m2 - m0
se = diff.std(ddof=1) / np.sqrt(len(diff))
c.record(
    "(iv) per-configuration midpoint cosh mass h = 0 / h = 2",
    f"{np.round(m0, 3).tolist()} / {np.round(m2, 3).tolist()}",
)
c.item(
    "(iv) h = 2 in the operator at fixed sigma raises the light midpoint cosh mass on every configuration: paired "
    "difference",
    (diff.mean(), se, diff.mean() / se),
    diff.mean() / se > 2 and np.all(diff > 0),
    "{0[0]:+.3f} +- {0[1]:.3f} ({0[2]:.1f} sigma)",
)

# sabotage
mm = rows[("L8", 2.41, 0.0)]["mc"][0]
inj = [mm / (1 + 3 * h) for h in hs]
c.item(
    "sabotage: injected seesaw rows m(h) = m(0)/(1 + 3h) fail the monotone-rise clause",
    True,
    not all(inj[i + 1] > inj[i] for i in range(3)),
)
c149 = [np.sqrt(0.4**2 + (0.6 * h) ** 2 / 4) - 0.6 * h / 2 for h in hs]
c.item(
    "sabotage: the partner-mass series (D = 0.4, M = 0.6h) falls and is rejected by the rise clause",
    np.round(c149, 3).tolist(),
    all(c149[i + 1] < c149[i] for i in range(3)),
)
Ssw = [rows[("L8", 3.0, h)]["Spi"] for h in hs]
steps = [(Ssw[i][0] - Ssw[i + 1][0]) / np.hypot(Ssw[i][1], Ssw[i + 1][1]) for i in range(3)]
c.item(
    "sabotage: y labels swapped -- the 'P_c' S(pi) series has no > 3 sigma fall at every step (pulls)",
    np.round(steps, 1).tolist(),
    not all(s > 3 for s in steps),
)
c.done()
