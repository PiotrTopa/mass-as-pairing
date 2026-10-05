# K8.8 — the weight of the large-N fit: any saturating crossover with one scale h₀ ≈ 2 describes the P_c σ channel as well as the large-N shape δ_L, the both-doublets shape fits better on two lattices, δ_L does not transfer across lattices within errors; what the data fix is "a saturating crossover, not a power of h"

**Statement.** On the same data and weighted least squares as K8.6 (1/S(π) at P_c, h ∈ {0, 0.5, 1, 1.5, 2, 3}, three
lattices):
1. Generic saturating crossovers with a free scale h₀ — Padé s₀ + b h²/(1 + h²/h₀²), s₀ + b tanh²(h/h₀),
   s₀ + b(1 − e^{−h²/h₀²}) — fit as well as δ_L by AIC on every lattice (best generic AIC − δ_L AIC = +0.0 / −1.4 / −2.8 on
   8⁴ / 6⁴ / 6³×12), with a lattice-independent scale (Padé h₀ = 2.10 / 2.21 / 2.05). A single power of h fails on every
   lattice (χ² 19.2 / 15.6 / 9.7 for 3 dof).
2. The shape of a mass on BOTH doublets fits better than δ_L on 6⁴ and 6³×12 (χ² 4.08 vs 4.45, 2.14 vs 5.68), worse on
   8⁴ (12.26 vs 2.21): the data do not identify the heavy half as the carrier of the subtraction.
3. Hold-out (shape and amplitude from one lattice, the target's h = 0 point as the only anchor, χ² over its 5 points):
   δ_L transfers better than the generic curves in 5 of 6 directions, but not within errors (χ² 8.4 … 28.0).
4. The light pair-channel mass in the Gaussian σ-mode form m² = m₀² + a δ_L has χ²/dof 24.8 (8⁴), 1.3 (6⁴), 3.0 (6³×12):
   it holds on 6⁴ only.

What the σ-channel data fix is a saturating crossover with scale h₀ ≈ 2, not a power of h. The large-N calculation
contributes the sign of the shift (K8.6), not a selected shape. Because the action is even in h on a finite gapped box,
an h² onset (p = 1 in K8.3's quadrature form) follows for any analytic shape and does not discriminate δ_L.

**Numbers.** (K8.8 check output)

| lattice | δ_L χ² (4) [AIC] | both doublets χ² (4) | Padé χ² (3) [AIC], h₀ | tanh² χ² (3) | Gauss χ² (3) | single power χ² (3) |
|---|---|---|---|---|---|---|
| 8⁴ | 2.21 [6.2] | 12.26 | 0.34 [6.3], 2.10 | 0.25 | 0.56 | 19.2 |
| 6⁴ | 4.45 [8.4] | 4.08 | 1.86 [7.9], 2.21 | 1.37 | 1.04 | 15.6 |
| 6³×12 | 5.68 [9.7] | 2.14 | 0.92 [6.9], 2.05 | 0.96 | 1.22 | 9.7 |

Hold-out χ² (5 points), δ_L / Padé: 8⁴→6⁴ 28.0 / 69.2, 8⁴→6³×12 13.1 / 23.5, 6⁴→8⁴ 26.4 / 53.3, 6⁴→6³×12 12.4 / 10.0,
6³×12→8⁴ 12.0 / 20.3, 6³×12→6⁴ 8.4 / 18.3.

**Method.** `masspairing.analysis.largen.pentest` (AIC = χ² + 2k; δ_L and the both-doublets δ from the closed-form bubble
of K8.6, whose curvature the chain runner's operator reproduces).

**Controls.** The Padé crossover on permuted h labels fails (χ²/dof 242 / 154 / 166).

**Caveats.** Post hoc by construction (a test of a post-hoc fit). Six h values per lattice, L ≤ 8.

**Data.** As K8.6: the stage-1/1b P_c chains in `data/derived/n1stage/{S1,S1b}/` and the free baselines.

**Check.** `python claims/K8.8/check.py` (≈ 2.5 min on one thread).

**Provenance.** Notebook claim C213 (unhiggsed_notebook @ ea0ce47).
