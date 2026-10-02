# K7.1 — finite-size-scaling crossing of ξ₂,stag/L at P_c from L = 6 and 8

**Statement.** On the κ = −0.01 line, the RG-invariant ratio ξ₂,stag/L of the staggered scalar channel is the same on
6⁴ and 8⁴ at the calibrated merged point y = 2.41 (a crossing at P_c), and it decreases with L on the SMG side at
|y − y_c| = 0.127 and is small (< 0.15) on both volumes at |y − y_c| ≈ 0.38. The Binder-type ratio R₄ at y = 2.41 lies
between the Gaussian value 5/3 and the ordered value 1 on both volumes. The two-volume slope ratio on the SMG side gives a
finite effective exponent; it shows that two volumes with about 10³ trajectories resolve the slope ratio at the 2σ level,
and it is not a measurement of ν (two volumes, one side, no corrections to scaling).

**Numbers.** (blocked jackknife, block = ⌈2τ_int(m²)⌉, m² = |Σ_stag|²; analysed prefix of each chain in brackets)

| y | L = 6: ξ₂/L | L = 8: ξ₂/L | Δ(8 − 6) |
|---|---|---|---|
| 2.028 | 0.120(5) [1500] | 0.091(6) [1000] | |
| 2.41 | **0.391(14)** [1300] | **0.401(17)** [900] | +0.010 (+0.5σ) |
| 2.537 | 0.243(8) [1500] | 0.194(14) [900] | −3.1σ |
| 2.792 | 0.104(6) [1500] | 0.073(9) [1000] | |

- R₄ = ⟨m⁴⟩/⟨m²⟩² at y = 2.41: 1.454(38) (L = 6), 1.408(56) (L = 8).
- Slopes d(ξ₂/L)/dy between y = 2.41 and 2.537: 1.16(13) (L = 6), 1.63(17) (L = 8); 1/ν_eff = ln(s₈/s₆)/ln(8/6) =
  1.18 ± 0.53, ν_eff ≈ 0.84 (+0.69 −0.26).
- τ_int(m²) at y = 2.41: 7.8 (L = 6), 6.4 (L = 8) trajectories.
- The other grid points (y = 2.283, 2.368, 2.452 at L = 6; 2.283 at L = 8 with 100 trajectories, τ_int > N/50) are
  printed by the check and listed in `results/K7_fss.csv`.

**Method.** Chains of the ε model at κ = −0.01, λ = 1 (L = 6: y = 2.028 … 2.792; L = 8: y = 2.028, 2.283, 2.41, 2.537,
2.792). Each chain is analysed over a fixed prefix (`masspairing.analysis.fss.K71_PREFIX`), all trajectories of the
prefix, no thermalisation cut. ξ₂,stag from S(π) and S(π + p_min): ξ₂² = (S(π)/S(π + p_min) − 1)/(4 sin²(π/L)).
Estimators: `masspairing.analysis.fss.ratio_row`, `slope_ratio`.

**Caveats.** Two volumes, one side of P_c for the slope ratio, no corrections to scaling; the prefixes are shorter than
the complete chains. The complete chains with the thermalisation cut and second seeds (K7.2) give ξ₂/L at 2.41 of
0.377(10) / 0.415(19) (+1.8σ), and the pooled P_c replicas (K7.3) 0.373(7) / 0.399(9) (+2.4σ): the crossing stays within
3σ, and at these volumes it sits slightly above y = 2.41 (K7.3).

**Data.** `data/derived/k7/t3a/F_L6_k-0.01/*.npz`, `data/derived/k7/t3a/F_L8_k-0.01/*.npz` (series `ts_Sigma_stag`,
`ts_S_pi`, `ts_S_pi_pmin`), from the archive chains `results/xi_scan/F_L6_k-0.01/`, `results/xi_scan/F_L8_k-0.01/`
(`scripts/derive_k7.py archive`). Table: `results/K7_fss.csv` (`scripts/table_k7_fss.py`); figure `figures/k7_xi_over_L.pdf`
(`scripts/fig_k7_xi_over_L.py`).

**Check.** `python claims/K7.1/check.py` (< 5 s): recomputes the table from the derived series and asserts (1) ξ₂/L at
2.41 equal on both volumes within 1σ and in [0.35, 0.45], (2) the drop at 2.537 by more than 2.5σ and ξ₂/L < 0.15 at
2.028 and 2.792, (3) a finite slope ratio, and R₄ at 2.41 more than 3σ from 5/3 and from 1.

**Provenance.** Notebook claim C083 (unhiggsed_notebook @ 3c4a02c).
