# I.3 — calibration: y_ours = √2 · y_ref, P_c = (2.41, −0.01)

**Statement.** In the conventions of this package (|σ|² = Σ_a (σ^a)², Yukawa term i y σ·τ), the phase diagram of the
reference SU(2) staggered-fermion study arXiv:2608.18239 (L = 8 window of the antiferromagnetic phase at κ = 0.2 between y₁ = 1.706 and
y₂ = 2.69, merged point (1.706, −0.01)) is reproduced by one rescaling, **y_ours = √2 · y_ref**, with κ and λ unchanged.
The merged point is therefore **P_c = (√2 · 1.706, −0.01) = (2.41, −0.01)**. The factor is fixed empirically: the other
two candidate conventions, y_ours = y_ref and y_ours = y_ref/√2, misclassify six and eight of the fifteen L = 8 points.

**Numbers.** (L = 8, κ = 0.2: 1000 trajectories per point; blocked jackknife; points with τ_int > N/50 flagged)

1. Lower transition: χ_Σ,stag = V(⟨|Σ_stag|²⟩ − ⟨|Σ_stag|⟩²) = 0.461(25) at y = 1.9, 0.935(57) at 2.2, 1.90(13) at 2.3,
   **4.09(53)** at 2.4 (τ_int 8.4), 2.26(21) at 2.5, 1.36(15) at 2.6. Resampled parabolas through the maximum and its
   neighbours give **y₁(L = 8) = 2.405 (+0.004 −0.003)**, against √2 · 1.706 = 2.413. Every point with y ≤ 1.9 is
   disordered (|Σ_stag| = 0.0213–0.0256); at y = 1.7 |Σ_stag| falls from L = 6 to L = 8 by 1.725 (1/√V: 1.778).
2. Upper transition: ordered at y = 2.6, 3.0, 3.6 (|Σ_stag| = 0.276, 0.392, 0.270), disordered at 3.9 (0.097, τ_int 37:
   flagged) and 4.2 (0.051): **y₂(L = 8) ∈ (3.6, 3.9)**, against √2 · 2.69 = 3.80.
3. Rescaling y_ours = s · y_ref (points within 0.15 s of a predicted edge not classified; ordered ⇔ |Σ_stag| ≥ 0.15):

   | s | predicted L = 8 window | misclassified L = 8 points |
   |---|---|---|
   | √2 | (2.41, 3.80) | none |
   | 1 | (1.71, 2.69) | 6 (y = 1.9, 2.2, 2.3, 2.4, 3.0, 3.6) |
   | 1/√2 | (1.21, 1.90) | 8 (y = 1.6, 1.65, 1.7, 1.75, 2.5, 2.6, 3.0, 3.6) |

4. |Σ_stag| near y = 3 does not decay with L: 0.383 (L = 4, y = 3.0), 0.397 (L = 6, 3.0), 0.392 (L = 8, 3.0), 0.380
   (L = 12, y = 2.9): y = 3.0 lies inside the rescaled window, not a finite-size effect.
5. κ scan (y = 2.6, 3.0 at κ ∈ {0.1, 0.05, 0, −0.01, −0.03}; 800 trajectories at L = 6, 400 at L = 8) against the rescaled
   window (√2 · 1.706, √2 · y₂(κ)) with y₂ linear between (−0.01, 1.706) and (0.2, 2.69): all 10 L = 6 points and all 7
   scored L = 8 points are classified correctly; ordered only at (κ, y) = (0.1, 2.6), (0.1, 3.0), (0.05, 2.6). Not scored
   (acceptance < 0.2): L = 8 at (−0.01, 2.6), (0, 2.6), (0.05, 2.6).

**Method.** Per stored chain (all trajectories): ⟨|Σ_stag|⟩ with its blocked-jackknife error, χ_Σ,stag and ξ₂,stag with
block length ⌈2 max τ_int⌉ over the chain's scalar and fermion series (`masspairing.analysis.calibration.point_summary`);
y₁ from 4000 parabolas through the χ maximum and its neighbours with the three values resampled within their errors
(`y1_parabola`, seed 0; median and 16–84 %). κ is not rescaled: y₁ does not depend on κ in the reference, y₂ does
(dy₂/dκ ≈ 6.6 in these units), and y₂(L = 8) matches √2 · 2.69 within the bracket, i.e. κ_ours = κ_ref to about ±10 %.

**Caveats.** Near P_c (y ≈ 2.3–2.6, κ ≲ 0.05) the spectrum of D†D reaches down to 10⁻⁴–10⁻³; with 10 Omelyan steps
⟨|ΔH|⟩ is then O(10) and the acceptance falls to 0.02–0.3 (the unscored κ-scan points); production chains near P_c use 40–50
steps. The κ scan brackets κ_c between −0.03 and 0.05, consistent with −0.01; the published κ_c = −0.01 is used. The
upper-edge point y = 3.9 has τ_int = 37 against N = 1000 (flagged), as long autocorrelations are expected at that boundary.

**Data.** `data/derived/k7/calib/*.npz` (55 chains: L = 8, 6, 12 at κ = 0.2; the κ scan at L = 6, 8; the 4⁴ and 6⁴
κ = 0.2 runs), from the archive chains `results/calib/{L8_k0.2, L6_k0.2, L12_k0.2, kscan2/*}/` and
`results/hmc_validation/{L4_k0.2, L6_k0.2}/` (where a point exists in both sets, the hmc_validation chain is used:
L = 6, κ = 0.2, y = 2.2). Tables: `results/I3_calibration.csv`, `results/I3_calibration.json`
(`scripts/table_i3_calibration.py`); figure `figures/i3_calibration.pdf` (`scripts/fig_i3_calibration.py`).

**Check.** `python claims/I.3/check.py` (≈ 2 s): prints every point with its τ_int and asserts (1)–(5), including the
convention control (only s = √2 has no misclassified point, the other two at least 3 each).

**Provenance.** Notebook claim C050 (unhiggsed_notebook @ 3c4a02c).
