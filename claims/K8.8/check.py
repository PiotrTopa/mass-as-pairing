#!/usr/bin/env python3
"""K8.8 -- the discriminating weight of the large-N fit of K8.6 (post hoc by construction): generic saturating
crossovers with one scale h0 ~ 2 fit the P_c structure factor as well as delta_L by AIC; the both-doublets shape fits
better on two lattices; delta_L's cross-lattice hold-out beats generic curves but fails within errors; the Gaussian
sigma-mode form of the light pair-channel mass holds on 6^4 only. Same data and weighted least squares as K8.6
(dense inverses up to V = 4096; about 5-10 min on one thread)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from masspairing.analysis import largen as G
from masspairing.claimcheck import Check

c = Check("K8.8")
drows = G.data_rows()
F = G.pentest(drows)
A, Bh, Cm = F["A"], F["B"], F["C"]
lats = G.LATS

# 1. shape competitors for 1/S(pi), AIC = chi^2 + 2k
for lat in lats:
    a = A[lat]
    c.record(
        f"{lat}: chi^2 (dof) [AIC]",
        f"delta_L {a['RPA_deltaL']['chi2']:.2f} (4) [{a['RPA_deltaL']['AIC']:.1f}]; both doublets "
        f"{a['both_doublets']['chi2']:.2f} (4); "
        + "; ".join(f"{g} {a[g]['chi2']:.2f} (3) [{a[g]['AIC']:.1f}] h0 {abs(a[g]['p'][2]):.2f}" for g in G.GENERIC)
        + f"; single power {a['POW']['chi2']:.1f} (3)",
    )
dA = [min(A[lat][g]["AIC"] for g in G.GENERIC) - A[lat]["RPA_deltaL"]["AIC"] for lat in lats]
c.item(
    "a generic saturating curve is as good as delta_L by AIC on every lattice: best generic AIC - delta_L AIC (8^4, "
    "6^4, 6^3x12)",
    dA,
    all(x <= 0.5 for x in dA),
    "{0[0]:+.1f}, {0[1]:+.1f}, {0[2]:+.1f}",
)
h0 = [abs(A[lat]["Pade"]["p"][2]) for lat in lats]
c.item(
    "the generic crossover scale is lattice-independent: Pade h0",
    h0,
    max(h0) - min(h0) < 0.3,
    "{0[0]:.2f}, {0[1]:.2f}, {0[2]:.2f}",
)
c.item(
    "a single power of h fails on every lattice (chi^2 for 3 dof)",
    [A[lat]["POW"]["chi2"] for lat in lats],
    all(A[lat]["POW"]["chi2"] > 9 for lat in lats),
    "{0[0]:.1f}, {0[1]:.1f}, {0[2]:.1f}",
)
c.item(
    "the both-doublets shape fits better than delta_L on 6^4 and 6^3x12 (chi^2 vs chi^2); not on 8^4",
    [(A[lat]["both_doublets"]["chi2"], A[lat]["RPA_deltaL"]["chi2"]) for lat in lats],
    all(A[lat]["both_doublets"]["chi2"] <= A[lat]["RPA_deltaL"]["chi2"] for lat in ("L6", "L6x12")),
    "8^4 {0[0][0]:.2f} vs {0[0][1]:.2f}; 6^4 {0[1][0]:.2f} vs {0[1][1]:.2f}; 6^3x12 {0[2][0]:.2f} vs {0[2][1]:.2f}",
)

# 2. hold-out
for k, v in Bh.items():
    c.record(f"hold-out {k}: chi^2 over 5 points", " | ".join(f"{nm} {v[nm]['chi2']:.1f}" for nm in v))
wins = sum(Bh[k]["RPA_deltaL"]["chi2"] < min(Bh[k]["Pade"]["chi2"], Bh[k]["tanh2"]["chi2"]) for k in Bh)
c.item("delta_L transfers between lattices better than the generic curves (directions out of 6)", wins, wins >= 5)
hc = [Bh[k]["RPA_deltaL"]["chi2"] for k in Bh]
c.item(
    "but not within errors: delta_L hold-out chi^2 for 5 points (min, max)",
    (min(hc), max(hc)),
    max(hc) > 20,
    "{0[0]:.1f} ... {0[1]:.1f}",
)

# 3. the light pair-channel mass in the Gaussian sigma-mode form, in sample
chi = [Cm[lat]["Gauss_sigma_mode"]["chi2"] / Cm[lat]["Gauss_sigma_mode"]["dof"] for lat in lats]
c.item(
    "m^2 = m0^2 + a delta_L: chi^2/dof on 8^4, 6^4, 6^3x12 -- holds on 6^4 only",
    chi,
    chi[1] < 2 and chi[0] > 10,
    "{0[0]:.1f}, {0[1]:.1f}, {0[2]:.1f}",
)

# 4. sabotage
bad = G.pade_permuted(drows)
c.item(
    "sabotage: the Pade crossover on permuted h labels fails (chi^2/dof)",
    bad,
    min(bad) > 5,
    "{0[0]:.0f}, {0[1]:.0f}, {0[2]:.0f}",
)
c.done()
