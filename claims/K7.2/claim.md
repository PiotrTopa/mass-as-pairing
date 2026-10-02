# K7.2 — no first-order signature at P_c at L ≤ 8 (pilot set, 39 chains)

**Statement.** On the 39 stored chains of the ε model at L = 6 and 8 around P_c = (2.41, −0.01) — the κ = −0.01 line at
seven y (second seeds at y = 2.368, 2.41, 2.452 on 6⁴), the κ = −0.04 and κ = +0.02 lines, and an afm / cold / hot start
set at P_c on 6⁴ — every first-order signature these volumes can carry is absent: no start dependence at P_c, no
two-peak histogram in any chain, an energy cumulant whose deficit falls like 1/V, a staggered susceptibility that grows
far more slowly than the volume, an L-independent Binder ratio and a persisting ξ₂/L crossing. The pre-registered
decision rule returns **CONTINUOUS (consistent with, L ≤ 8)**. "Consistent with" is the whole content: two volumes cannot
establish continuity, and a weak first-order transition with a correlation length much larger than 8 lattice spacings is
not excluded. Nothing here concerns walking (K7.4).

**Numbers.** (first 100 trajectories of every chain cut; blocked jackknife, block = 2τ_int(m²), m = |Σ_stag|; replicas
combined by inverse variance)

1. Hysteresis at P_c on 6⁴ (afm / cold / hot, 1000 trajectories each): 15 pulls on {m², σ², O₄, S(π), S(π + p_min)}, largest
   **2.66σ** (afm–cold on m²); σ² and O₄ pulls ≤ 1.65σ; τ_int(m²) = 7.5 / 5.7 / 5.2; all three chains scored.
2. Double peaks: none of the 117 series (m, σ², O₄ of 39 chains) is bimodal. Energy cumulant
   max_y(2/3 − V_e(σ²)) = 8.68(25) × 10⁻⁴ at L = 6 (y = 2.452), 2.44(16) × 10⁻⁴ at L = 8 (y = 2.41): ratio **0.28(2)**
   (1/V law: 0.32; a latent heat keeps it constant).
3. Susceptibility exponent at y = 2.41: S(π) = V⟨m²⟩ = 20.0(8) → 33.5(2.5), **γ/ν_eff = 1.79(30)**, η_eff = +0.21(30),
   7.4σ below the first-order value 4; from the grid maxima of S(π): 1.69(28). R₄ = 1.470(32) → 1.419(65),
   ΔR₄ = −0.052(72).
4. Crossing: ξ₂/L at y = 2.41 = **0.377(10)** (L = 6), **0.415(19)** (L = 8), +1.8σ; it falls with L at every other y of
   the κ = −0.01 line: 2.028 −3.3σ, 2.283 −1.7σ, 2.537 −2.9σ, 2.792 −2.3σ. Finite-difference slope exponent
   θ = ln(s₈/s₆)/ln(8/6) between 2.41 and 2.41 ∓ 0.127: 1.23(50) (SYM side), 1.77(53) (SMG side) (1/ν_eff for a cusp,
   2/ν_eff for a smooth top). The pilot set has no L = 8 chain at y = 2.368 / 2.452 on this line, so the inner-pair θ
   and its (6,8) drift are not formed here (K7.3 adds them).
5. Verdict: CONTINUOUS (consistent with, L ≤ 8); the same by the letter of the rule with H3 kept.
6. The other κ lines. κ = +0.02: at y = 2.452 |Σ_stag| = 0.152(11) (L = 6) and 0.152(2) (L = 8), ratio 1.01 (ordered),
   γ/ν_eff = 3.89(51), R₄(8) = 1.09; at y = 2.368, 2.41 the ratio |Σ_stag|₆/|Σ_stag|₈ is 1.37, 1.30 with ξ₂/L rising
   with L (+2.0σ, +3.6σ). κ = −0.04: ξ₂/L falls with L at every y (−1.2 to −3.5σ), |Σ_stag|₆/|Σ_stag|₈ = 1.75–1.78
   (pure fluctuation: 1.78), γ/ν_eff compatible with 0 (S(π) L-independent), while O₄ (L = 6) rises from 0.0138 at
   y = 2.028 to 0.0887 at y = 2.452.
7. Phase labels of all 39 chains (AFM ⇔ ⟨|Σ_stag|⟩ ≥ 0.15; else SMG ⇔ ⟨O₄⟩ ≥ 0.04; else SYM; the six P_c chains
   "critical"): `results/K7_phase_labels.csv` (the stored configuration of each archive chain is its array
   `sigma_final`). Not scored (τ_int > N/50 or non-stationary): L = 6, κ = −0.04, y = 2.283, 2.368, 2.41; L = 6,
   κ = +0.02, y = 2.452.

**Method.** `masspairing.analysis.t3a.analyse_pilot`. Per chain after the cut: m², S(π), S(π + p_min), σ², O₄ (every second
trajectory), R₄, ξ₂/L, V_e = 1 − ⟨e⁴⟩/(3⟨e²⟩²) with e = σ². A chain is scored iff its τ_int window converged,
τ_int ≤ N/50 and its two halves agree within 3σ on m² and σ². Signals: H1 hysteresis (positive if a pull > 4σ with both
chains scored; negative if all < 3σ); H2 bimodality (Gaussian kernel density at the Scott bandwidth and 0.75 × Scott,
two maxima, the lower ≥ 20 % of the higher, valley ≤ 50 % of the lower; a first-order sign needs the same (κ, y) bimodal
on both volumes) with the V_e ratio as support; H3 slope exponent θ; H4 γ/ν_eff (first-order sign if γ/ν − 2σ ≥ 3.5,
continuous-consistent if γ/ν + 2σ < 4, the grid-maximum value must agree); H5 ΔR₄. CONTINUOUS ⇔ H1 negative, H2 negative,
H4 continuous-consistent, H5 not first-order, H3 continuous-consistent or without power, the P_c chains scored, the
exclusion controls passed.

**Pre-registration and deviations.** The data set, cuts, scoring, signals, thresholds, decision rule, controls and phase
rule were fixed before the data were read (notebook commits 35fc472, a92f7a6; the second added, also before any number,
the grid-maximum robustness of H4, the cusp/smooth-top reading of θ and O₄ as a second energy-like density). After the
controls were run: (i) the injected-jump control was coded with the wrong sign (it moved the afm series towards the cold
one); fixed to inject away from it, no criterion changed. (ii) The pre-registered bandwidth pair (Scott, 1.5 × Scott)
over-smoothed the 4σ control mixture (2/40 detected); replaced by (Scott, 0.75 × Scott) with the 5σ mixture as the
control and the 4σ rate reported; the data are unimodal under both rules. (iii) H3's first-order mock is classified
continuous-consistent (θ ≈ 1), so H3 has no power on this grid and is reported only; with H3 kept the letter of the rule
gives the same verdict. (iv) H4's first-order mock (γ/ν = 4.0 ± 0.3 with the real errors) lands "undecided", not
"first-order sign": at this precision H4 excludes first order but cannot establish it. (v) Phase labels: the P_c chains
are labelled "critical"; off the κ = −0.01 line "near-critical" means adjacent to a label change along the same (κ, L)
line.

**Controls.** Injected jump (afm m² shifted by 6× the combined error away from cold): 8.7σ, flagged. Bimodality: 5σ
two-component AR(1) mixture detected 40/40, 4σ mixture 34/40, unimodal AR(1) 0/40 false positives (τ of the 6⁴ P_c
point). Finite-size-scaling mocks on the real grid with the real errors: first order (pseudo-transition of width 1/V,
S(π) ∝ V, ξ₂/L peak ∝ L) gives γ/ν = 4.0(3), never continuous-consistent under H4; smooth (ν = 0.61, γ/ν = 3.08) gives
γ/ν = 3.08(30), continuous-consistent, never first order; the first-order mock's θ = 1.0(2) on both sides (no power).

**Caveats.** Two volumes. The hysteresis test here is at 6⁴ only (K7.3 adds 8⁴); the 6⁴ hot and afm chains share a seed
(different starts, same noise stream). κ_c = −0.01 is the published value transferred by I.3 (its κ scan brackets κ_c in
(−0.03, 0.05)). η_eff is the scalar staggered channel, not a fermion-bilinear exponent. The three unscored κ = −0.04
points carry long low-acceptance stretches.

**Data.** `data/derived/k7/t3a/<dir>/*.npz` for the 11 directories of the pilot set (series of
`masspairing.analysis.t3a.SERIES`), from the archive chains `results/xi_scan/{F_L6_k-0.01, F5_L6_k-0.01_seed2,
L6_k-0.04, L6_k0.02, F_L8_k-0.01, F2_L8_k-0.04, F2_L8_k0.02, L8_k-0.04, F_hyst_L6_afm, F_hyst_L6_cold, F_hyst_L6_hot}/`
(`scripts/derive_k7.py archive`). Tables: `results/K7_fss.csv`, `results/K7_hysteresis.csv`, `results/K7_verdict.json`,
`results/K7_phase_labels.csv` (`scripts/table_k7_fss.py`, `scripts/table_k7_order.py`); figure `figures/k7_xi_over_L.pdf` (`scripts/fig_k7_xi_over_L.py`).

**Check.** `python claims/K7.2/check.py` (≈ 6 s): runs the whole analysis from the derived series and asserts items 1–7
and the controls.

**Provenance.** Notebook claim C051 (unhiggsed_notebook @ 3c4a02c).
