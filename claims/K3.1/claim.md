# K3.1 — the complete 10-channel correlator does not grow with the volume at five link-field points, L ≤ 8

**Statement.** On the sign-free link-field (wedge) model at κ = −0.01, λ = 1, at the five points (y, g₁₀, g₆) =
(2.41, 0.05, +0.05), (2.41, 0.1, +0.1), (3.0, 0.1, +0.1), (2.41, 0.05, 0), (2.41, 0.02, 0), the complete SU(4)-invariant
10-channel correlator χ₁₀ = Σ_ab⟨Φ̄^{ab}Φ^{ab}⟩ at zero momentum (complete estimator with the fermionic boundary sign,
I.6) is largest at p = 0 among the 8 independent corner momenta at every point and volume, and it falls slightly with
the volume: d ln χ₁₀/d ln V on (6, 8) is between −0.12 and −0.03, with exponent + 2σ ≤ 0.10 at every point, and
ξ₁₀/L decreases from 6⁴ to 8⁴. By the criterion fixed before the data were read ("exponent + 2σ < 0.5 on (6,8) and
ξ₁₀/L not rising"), there is no (10,3,1) condensate at these points up to L = 8. The columnar dimer channel grows
instead (positive control; K3.6).

**Numbers.** (blocked jackknife; exponents d ln X/d ln V with linear error propagation)

| (y, g₁₀, g₆) | χ₁₀(p=0) L = 4 / 6 / 8 | exponent (4,6) / (6,8) | ξ₁₀/L 6 → 8 |
|---|---|---|---|
| (2.41, 0.05, +0.05) | 0.3795(47) / 0.3664(16) / 0.350(26) | −0.02(1) / −0.04(7) | 0.0924(2) → 0.069(18) |
| (2.41, 0.1, +0.1) | 0.2684(17) / 0.2285(6) / 0.2035(22) | −0.10(0) / −0.10(1) | 0.0829(1) → 0.055(3) |
| (3.0, 0.1, +0.1) | 0.1382(20) / 0.1077(10) / 0.1027(71) | −0.15(1) / −0.04(6) | 0.0753(3) → 0.061(19) |
| (2.41, 0.05, 0) | 0.4465(37) / 0.3983(14) / 0.383(17) | −0.07(1) / −0.03(4) | 0.0895(2) → 0.076(11) |
| (2.41, 0.02, 0) | 0.537(7) / 0.4960(22) / 0.436(34) | −0.05(1) / −0.12(6) | 0.0881(2) → 0.011(55) |

- Positive control: V⟨D_∥²⟩ at (2.41, 0.1, +0.1) is 4.97 / 15.72 / 39.28 at L = 4 / 6 / 8, exponents 0.71(1) and
  0.80(1): the same pipeline resolves a growing channel.
- Local plaquette energy E₁₀ = ⟨T_Q⟩ at 8⁴: 0.479, 0.337, 0.177, 0.526, 0.672 (points in table order).
- Also printed, not asserted: the two-site 6 channel (largest corner) is small and flat or falling (exponents
  −0.32 … +0.12); the ε channel V⟨|φ_stag|²⟩ grows sub-linearly (0.16 … 0.46 on (6,8)).

**Method.** 15 chains (L = 4: exact propagators, 400 trajectories after 60; L = 6: 1000 after 100, exact propagators
every 4th trajectory; L = 8: 1500 after 100, 8 Z2 noise vectors every 2nd trajectory), all with trajectory-length
jitter 0.3 (I.5), Hasenbusch + nested integrator at L = 6, 8. Per ensemble: Madras–Sokal τ_int and a blocked
jackknife with block ⌈2τ_int⌉ of the slowest summarised series; corner structure factors and ξ₁₀ (second moment
from S(p_c) and S(p_c + p_min)) with that block. Estimators and the kill criterion:
`masspairing.analysis.k3.ensemble`, `exponent`, `scaling_verdict`.

**Pre-registration and deviations.** The kill criterion ("largest-corner exponent < 0.5 on (6,8) at every point with
ξ₁₀/L not rising → null; exponent ≥ 0.8 with ξ₁₀/L rising → alive"; operationally ±2σ) was fixed before the data were read
(notebook commit 270b3ab). The 8⁴ chain at (2.41, 0.02, 0) ran its first 600 trajectories with a poorly
accepting integrator (acceptance 0.04–0.32); they are dropped, and the chain was split at trajectory 1300 into the
original stream (to 1459) and two re-seeded replicas (to 1420), kept from 1400: 859 + 20 + 20 trajectories. No other cut.

**Controls.** (1) Link-field equilibration ⟨s_j²⟩/2g_j over the first 50 kept trajectories 1.02–1.18 on every chain
(the freezing of I.5 is absent). (2) Positive control above. (3) Robustness: with 100 more trajectories cut from every
stream the largest-corner exponents move by ≤ 0.03 and every verdict is unchanged.

**Caveats.** The 8⁴ errors are noise-dominated (stochastic estimator with 8 noise vectors; per-measurement noise/value
0.5–0.6 on soft configurations), 7–10× the 6⁴ ones; the verdict holds because every (6,8) exponent + 2σ is ≤ 0.10, far
below 0.5. Two volume pairs; the exponents are finite-size exponents of a UV-dominated, free-like correlator (K3.2),
not scaling dimensions.

**Data.** `data/derived/k3/F8_P1_wedge/`, `data/derived/k3/F8_P1_wedge_rA/`, `_rB/` (series of the archive chains
`results/xi_scan/F8_P1_wedge*/`, `scripts/derive_k3.py series`). Table `results/K3_chi10_scaling.csv`
(`scripts/table_k3.py`), figure `figures/k3_chi10_response.pdf` (`scripts/fig_k3.py`).

**Check.** `python claims/K3.1/check.py` (≈ 5 s): completeness, equilibration, the table, p = 0 the largest corner,
the verdict at every point, the positive control, the robustness cut.

**Provenance.** Notebook claim C104 (unhiggsed_notebook @ 3c4a02c).
