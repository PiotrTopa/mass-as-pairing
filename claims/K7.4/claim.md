# K7.4 — ξ(y) at L ≤ 12 cannot separate walking from a power law

**Statement.** For the two continuous scenarios ξ_W(t) = a·exp(c/√t) (walking, merged fixed point) and
ξ_P(t) = a·t^(−ν) (power law, ν = 0.61), t = |y − y_c|, matched to the same dynamic range on the measured grid and using
only points with ξ ≤ L_max/3 (an infinite-volume ξ needs a box ≥ 3ξ), the expected Δχ² of the wrong model fitted with a
free y_c is undefined on the three-point grid t = 0.042, 0.127, 0.38 (too few usable points), undefined or at most 1.2
for L_max ≤ 12 even with 7 log-spaced points in [0.01, 0.4] and 1 % errors on ξ, and becomes decisive only at L_max = 16 with
1 % errors or at L_max = 24. With y_c fixed externally, L_max = 12 at 3 % would give Δχ² of 4.7 / 14.1, but y_c is not
known to better than the smallest usable t. The bound is optimistic (Gaussian errors, no corrections to scaling, no growth
of τ_int). The order of the transition (K7.2, K7.3) is decidable at L ≤ 8; the walking question is not decidable at
L ≤ 12.

**Numbers.** (Δχ² of the wrong model: truth walking / fit power law | truth power law / fit walking)

| grid | y_c | L_max | ε | usable points | Δχ² |
|---|---|---|---|---|---|
| t = 0.042, 0.127, 0.38 | fixed | 8, 12, 16 | 3 % | 2 | no fit |
| t = 0.042, 0.127, 0.38 | free | 8, 12, 16 / 24 | 1 % | 2 / 3 | no fit |
| 7 log-spaced in [0.01, 0.4] | free | 8 | 1, 3, 5 % | 3 | no fit |
| 7 log-spaced in [0.01, 0.4] | free | 12 | 1 / 3 / 5 % | 4 | 1.19 / 0.13 / 0.05 (largest of the two) |
| 7 log-spaced in [0.01, 0.4] | free | 16 | 1 % | 5 | 12.7 / 7.8 |
| 7 log-spaced in [0.01, 0.4] | free | 16 | 3 % | 5 | 1.42 / 0.87 |
| 7 log-spaced in [0.01, 0.4] | free | 24 | 1 % | 6 | 76.0 / 32.0 |
| 7 log-spaced in [0.01, 0.4] | fixed | 12 | 3 % | 4 | 4.7 / 14.1 |

The complete table (five grids, L_max = 8, 12, 16, 24, ε = 1, 3, 5 %, y_c fixed and free) is `results/K7_discrimination.csv`.

**Method.** `masspairing.analysis.fss.discrimination_table` (deterministic, ≈ 3 s): the true curve is sampled exactly at
the usable grid points with relative errors ε·ξ; the wrong model (parameters log a and its exponent, plus a shift of y_c
when free) is fitted by least squares from four starting points; its minimum χ² is the expected Δχ². Both curves have
ξ(t_max) = 1.5 and the dynamic range (t_max/t_min)^ν of the power law.

**Caveats.** Pseudo-data without noise realisations: Δχ² is the expected value. Corrections to scaling and the growth of
τ_int near y_c would lower the discrimination further. A 1 % error on ξ₂ at L = 16 near P_c is an estimated
10⁴ independent configurations per point (not computed here).

**Data.** None (deterministic computation). Table: `results/K7_discrimination.csv` (`scripts/table_k7_discrimination.py`).

**Check.** `python claims/K7.4/check.py` (≈ 3 s): recomputes the table and asserts the inequalities (a)–(c) of its
docstring.

**Provenance.** Notebook claim C082 (unhiggsed_notebook @ 3c4a02c).
