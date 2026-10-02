# K7.3 — 8⁴ three-start hysteresis, inner pair and pooled P_c: "continuous (consistent with, L ≤ 8), sharpened"

**Statement.** Adding to the pilot set (K7.2) three 8⁴ chains at P_c = (2.41, −0.01) from afm, cold and hot starts
(2000 trajectories each, distinct seeds) and the inner pair y = 2.368 / 2.452 at L = 8 on the κ = −0.01 line: all three
starts are scored, there is no start dependence at P_c on 8⁴, the four 8⁴ P_c chains agree across the two GPU code paths
that produced them, no new chain has a double peak, and with the pooled P_c replicas the staggered susceptibility exponent
is γ/ν_eff = 1.58(16), 16σ below the first-order value. The pre-registered rule returns **CONTINUOUS (consistent with,
L ≤ 8), sharpened**, and three sabotaged copies of the data passed through the same unchanged pipeline never return it.
It remains a statement on two volumes — no first-order signature at L ≤ 8 — and not a positive proof of continuity.

**Numbers.** (first 100 trajectories cut; blocked jackknife, block = 2τ_int(m²))

1. Scoring of the 8⁴ starts (n = 1900 each): τ_int(m²) = 16.8 / 10.3 / 16.7 (afm / cold / hot; N/50 = 38, windows
   converged), half-split pulls on m² / σ² = 2.6/0.6, 1.5/0.5, 1.2/1.0σ; acceptance 0.83 / 0.84 / 0.85.
2. Hysteresis on 8⁴: 15 pulls on {m², σ², O₄, S(π), S(π + p_min)}, largest **2.42σ** (cold–hot on S(π + p_min)); m² pulls
   0.6 / 0.8 / 0.3σ; energy-like (σ², O₄) largest 1.51σ. Per start: |Σ_stag| = 0.0782(33) / 0.0817(25) / 0.0818(35),
   σ² = 1.3026(5) / 1.3040(8) / 1.3034(8), O₄ = 0.0411(11) / 0.0426(12) / 0.0419(19), S(π) = 29.2(2.3) / 31.0(1.8) /
   31.9(2.5), ξ₂/L = 0.386(19) / 0.393(14) / 0.411(20). Combined with 6⁴ (largest 2.66σ, K7.2): H1 negative.
3. Code paths: the pilot set's 8⁴ P_c chain (GPU path without the fused kernels) and the three starts (fused path):
   30 pulls, largest 2.42σ, pilot chain vs the starts ≤ 1.31σ — no code-path discrepancy.
4. Pooled P_c (inverse variance): L = 8, 4 replicas, n = 6600, largest pairwise m² pull 1.2σ: ξ₂/L = **0.399(9)**,
   S(π) = 31.2(1.1), R₄ = 1.509(28), |Σ_stag| = 0.0815(16). L = 6, 5 replicas (two seeds + the three 6⁴ starts),
   n = 5500, largest pull 2.7σ: ξ₂/L = **0.373(7)**, S(π) = 19.80(54), R₄ = 1.457(21).
5. Signals. No bimodal series in the five new chains; energy-cumulant ratio 0.28(2) (1/V law 0.32). **γ/ν_eff = 1.58(16),
   η_eff = +0.42(16)** pooled, 1.79(30) unpooled, 1.81(58) from the grid maxima (y = 2.452 on both volumes). ΔR₄ at 2.41:
   +0.052(35) pooled, −0.052(72) unpooled. Crossing: Δ(ξ₂/L) at 2.41 = **+2.4σ** pooled (+1.8σ unpooled), −0.047 at
   y = 2.537; the growth of ξ₂/L at P_c from 6⁴ to 8⁴, 1.07(3), is far from the first-order peak growth ∝ L (8/6 = 1.33):
   on these volumes the (6,8) crossing sits slightly above y = 2.41.
6. Inner pair at L = 8: y = 2.368: ξ₂/L = 0.344(14), S(π) = 22.5(1.3), scored, Δ(ξ₂/L) = +0.016(15) (+1.0σ),
   γ/ν_eff = 1.52(24); y = 2.452: ξ₂/L = 0.393(41), S(π) = 34.6(5.7), τ_int = 29.3 > N/50 (acceptance 0.74) — not
   scored. The ξ₂/L maximum is at y = 2.41 on both volumes. θ_inner = 1.34(1.46) (SYM side), 2.3(8.1) (SMG side, unscored
   input); (6,8) drift θ_outer − θ_inner = −0.10(1.55) (SYM), −0.55(8.1) (SMG). The first-order mock gives θ = 1.01(10) at
   the inner pair, classified continuous: θ has no power against first order on this grid either and is reported only.
7. Verdict by the rule: CONTINUOUS (consistent with, L ≤ 8), sharpened; every P_c chain scored at both volumes.
8. Phase labels of the five chains: y = 2.368 SYM, the three P_c starts critical, y = 2.452 SMG (all near-critical,
   |y − 2.41| ≤ 0.05); `results/K7_phase_labels.csv`.

**Method.** `masspairing.analysis.t3a.analyse_sharpened`: the statistics, scoring, signals and decision rule of K7.2, with
H1 at 8⁴ (positive only if, in addition, the last-half means differ by > 3σ), combined H1 over both volumes, H2 over all 44
chains, H3 weighted only if its first-order mock at the inner pair is not called continuous, H4 / H5 / the crossing on
both the pooled and the unpooled P_c (the two must agree), and all P_c chains scored at both volumes.

**Pre-registration and deviations.** The criteria (8⁴ H1, replica test, pooling, inner-pair θ and its weight, combined
decision rule, controls) were fixed before any number of the new chains was read (notebook commit 9e0c97e). The rule was
first applied with 1000 trajectories per start; the afm chain then failed the stationarity condition (half-split 3.5σ, a
low-m² excursion) and the rule returned UNDECIDED, with every signal on the continuous side. The three start chains were
then extended to 2000 trajectories and the same rule, with no change, gives the result above; the inner-pair chains were
not extended.

**Controls.** Injected jump at 8⁴ (afm m² shifted by 6× the combined error away from cold): 6.6σ, flagged. Bimodality at
τ = 16.8: 5σ mixture 40/40, 4σ mixture 32/40, AR(1) 0/40 false positives. Mocks with the pooled errors: the first-order
mock gives γ/ν = 4.0(16) and is flagged first order by H4; the smooth mock (ν = 0.61, γ/ν = 3.08) is continuous-consistent
and never first order under H4 or H3 (θ_inner = 2.94(1.73)). Sabotage through the complete unchanged pipeline:
(a) ordered-start memory (afm m² + 8× the afm–cold combined error) → H1(8) positive at 7.4σ, verdict UNDECIDED;
(b) a two-state telegraph signal in the hot chain's m and σ² (levels 5 standard deviations apart, dwell ≈ 60 trajectories)
→ bimodal in m and σ², H1(8) positive at 7.9σ, H2 indicative, verdict UNDECIDED; (c) a late drift in the afm chain (second
half shifted, realised half-split 3.1σ) → non-stationary and τ_int > N/50, unscored, verdict UNDECIDED.

**Caveats.** Two volumes. The afm chain has a slow mode of several hundred trajectories: m² = 5.69(77) × 10⁻³ over
trajectories 100–1000 and 8.38(73) × 10⁻³ over 1000–2000 (2.5σ; 200-trajectory windows 8.4, 6.8, 4.0, 4.4, 5.7, 4.7,
9.2, 9.2, 10.4, 9.0 × 10⁻³) — a downward excursion that ended, not an ordered-start memory; it is stationary by the rule at
2.6σ. 2000 trajectories are ≈ 110–180 independent samples per start. The pooled η_eff is dominated by the three start
chains (the pilot chain is the high replica). The crossing statistic at 2.41 is +2.4σ pooled, below the 3σ at which the
crossing would be lost; a third volume would locate it. The SMG-side inner point is unscored. Pooling treats the 6⁴ hot /
afm pair (shared seed, different starts) as independent.

**Data.** `data/derived/k7/t3a/` (pilot set of K7.2, and `F12_L8_k-0.01_inner/`, `F12_hyst_L8_{afm,cold,hot}/`), from the
archive chains `results/xi_scan/F12_L8_k-0.01_inner/`, `results/xi_scan/F12_hyst_L8_{afm,cold,hot}/`
(`scripts/derive_k7.py archive`). Tables: `results/K7_hysteresis.csv`, `results/K7_fss.csv`, `results/K7_verdict.json`,
`results/K7_phase_labels.csv` (`scripts/table_k7_fss.py`, `scripts/table_k7_order.py`); figure `figures/k7_xi_over_L.pdf` (`scripts/fig_k7_xi_over_L.py`).

**Check.** `python claims/K7.3/check.py` (≈ 25 s): runs the analysis of the 44 chains and the three sabotaged copies from
the derived series and asserts items 1–8 and the controls.

**Provenance.** Notebook claims C171, C192 (unhiggsed_notebook @ 3c4a02c).
