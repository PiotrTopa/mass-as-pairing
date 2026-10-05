# K8.3 — small-h onset at P_c on 6⁴: p = 1.14(37), consistent with the h² onset required by the h → −h symmetry; the pre-registered quadratic band tested δm ∝ h⁴ and is recorded as mis-specified

**Statement.** The pre-registered onset fit F1, m² − m₀² = (c h^p)² on 6⁴ aaaa at P_c = (2.41, −0.01), with
h ∈ {0.25, 0.5 (pooled replicas), 0.75, 1} and m₀ the stored h = 0 chain (m the light pair-channel midpoint cosh mass),
gives p = 1.14(37), χ²/dof 0.17, I = p ± 2σ = [0.41, 1.87]. By the letter of the order, I meets both the "quadratic-like
onset" band [1.6, 2.4] and the linear band [0.7, 1.3] and σ_p > 0.3: **UNDECIDED**.

Reading. Every observable is even in h (the action is invariant under h → −h with σ → −σ, so det D(h) = det D(−h)), and on
a finite gapped box it is analytic in h: the onset is δm ∝ h². In this regime δm ≪ m₀ (m(1) − m₀ = 0.061 against
m₀ = 0.392), so m² − m₀² ≈ 2m₀δm and the fit's p is the exponent of h² ↔ p = 1. The measured p = 1.14(37) is consistent with
p = 1 at 0.4σ. The order's quadratic band [1.6, 2.4] corresponds to δm ∝ h⁴, which nothing predicts; it was carried
over from the 8⁴ quadrature exponent κ ≈ 1.8 of K8.2, read in the opposite regime m ≫ m₀ where κ is the exponent of m
itself. The bands are recorded as mis-specified; the numbers and the verdict of the letter stand as computed. The test
has no power beyond "consistent with the h² onset".

Power-limit control (pre-registered): a synthetic quadratic onset with the real errors also returns UNDECIDED
(p = 2.00(48)); the linear synthetic returns "linear" (1.00(20)); the saturating one is not called quadratic (0.44(2)).

**Numbers.** (K8.3 check output)

| h | m(t = 2) on CL_t | origin |
|---|---|---|
| 0 | 0.3919(84) | stage 1 |
| 0.25 | 0.3921(72) | stage 1c, pooled |
| 0.5 | 0.4041(52) | stage 1c, pooled (stored stage 1: 0.4066(87), −0.26σ) |
| 0.75 | 0.4270(57) | stage 1c |
| 1 | 0.4527(43) | stage 1 |

- F1: p = 1.140(366), χ²/dof 0.17, I = [0.41, 1.87] → UNDECIDED by the letter; (p − 1)/σ_p = 0.38.
- Labelled fits: F1′ (with the stored h = 0.5) 1.128(362), F2 (h ≤ 0.75) 1.38(85), F1f (free-normalised) ≡ F1; all UNDECIDED.
- Monotonicity pulls 0 → 0.25 → 0.5 → 0.75 → 1: 0.0σ, 1.4σ, 3.0σ, 3.6σ.
- Local quadrature exponents 0.5 → 3: 1.34, 1.01, 0.91, 0.80, 0.53 (a crossover to saturation, K8.2).
- Recorded (post hoc): Gaussian-MC σ_p = 0.44; P(p ≥ 1.6) = 0.16; P(0.7 ≤ p ≤ 1.3) = 0.60.

**Method.** `masspairing.analysis.stage1c`: m by the stage-1b estimator (delete-one-block jackknife × error-model
inflation); replicas pooled after the K5.9 replica test (each member's jackknife × its own inflation); F1 by weighted
least squares with m₀ fixed, σ_p from a chain-block jackknife including the h = 0 chain.

**Pre-registration and deviations.** Fit, bands, controls and the reading rule were fixed before the stage-1c chains
started (notebook order K-N1-S1c, commit 56a8e32). The quadratic control failing is the order's own power limit,
reported as written. The bands are re-read here (h → −h evenness, δm ≪ m₀); no assertion of the check was changed.

**Controls.** The three synthetic onsets above, with the real errors.

**Caveats.** One volume (6⁴), five h values; the h = 0 chain carries 0.31 of σ_p. The p = 1 onset is a property of any
analytic, even response on a gapped finite box; it does not discriminate between mechanisms (K8.8).

**Data.** `data/derived/n1stage/S1c/{L6,L6_r2}/*.npz`, `data/derived/n1stage/{S1,S1b}/L6/`,
`data/derived/free/free_L6_aaaa_h0.25_0.5_0.75.json`. Raw archive: `results/xi_scan/K_N1_S1c/{L6,L6_r2}/`,
`results/laneK1/free_L6_aaaa_h0.25_0.5_0.75.json` (bundle `mass-as-pairing-data_n1-stage1c.tar.gz`).

**Check.** `python claims/K8.3/check.py` (≈ 30 s).

**Provenance.** Notebook claim C205 (unhiggsed_notebook @ fd0c934); reading per the notebook's audit (ea0ce47).
