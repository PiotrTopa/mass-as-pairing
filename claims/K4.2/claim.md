# K4.2 — stage 0 (4⁴, 4³×8): the SMG phase screens the elementary taste-chiral Majorana mass and the ε-vertex composite carries the response

**Statement.** In model N1 (ε model at κ = −0.01, λ = 1, g₁₀ = 0, with the flavour-blind taste-chiral Majorana mass
h C_χ on the heavy taste doublet "R" of every flavour, all-antiperiodic boxes) on 4⁴ and 4³×8:
(i) in the SMG phase (y = 3.0) the elementary Majorana channel does not respond to h at first order — the heavy/light
on-configuration asymmetry is zero and the elementary amplitude φ_R/h is ≤ 3 % of free — while the composite
Ψ = Π_{b≠a} χ^b in the same direction responds linearly in h, h-independent within 5.3 % per unit h and equal on 4⁴ and
4³×8 within 3.3 %, ≥ 50 × the free value; at y = 2.0 (SYM) and at P_c (y = 2.41) the response sits on the elementary
side (|φ_T_R/φ_R| ≤ 0.0015 and 0.0045–0.0066);
(ii) the light sector is ordered SYM → P_c → SMG relative to free (light pair susceptibility and momentum readout fall),
and at y = 3.0 both fall from L_t = 4 to 8 (zero-like, recorded on two L_t only);
(iii) the light sector is h-stable: the t* log-ratio of the light pair-channel correlator moves by < 2.5σ for h ≤ 0.5
and stays within [0.84, 1.09] of its h = 0 value (free-normalised) at h = 2 on every row;
(iv) the screening is a property of the configurations: re-measured on the same stored σ, switching h = 2 → 0 moves the
heavy channel by 13–23 % and the light by < 3.5 % at y = 2.0 (doublet-selective, as in the free theory), but moves light
and heavy together at y = 3.0 on 4⁴ and 4³×8.

**Numbers.** (K4.2 check output; 26 chains, 600 trajectories, measurements every 2, first 75 of 300 cut, 20-block errors)

Composite / elementary ratio ρ = φ_T_R/φ_R (aaaa rows with h > 0):

| y | ρ | composite φ_T_R/h |
|---|---|---|
| 2.0 (SYM) | \|ρ\| ≤ 0.0015 at every h | (−1.5 … −3.8)·10⁻⁵ |
| 2.41 (P_c) | 0.0045–0.0066 | (−1.0 … −1.6)·10⁻⁴ |
| 3.0 (SMG), rows with φ_R resolved (> 2σ) | 2.6–5.8 (4 rows) | −0.001736 … −0.001830 (4⁴), −0.001828 / −0.001819 (4³×8) |

- y = 3.0: φ_R/h ≤ 0.025 × free at every h (h ∈ {0.1, 0.2, 0.5, 1, 2} on 4⁴, {0.5, 2} on 4³×8); φ_T_R/h ≥ 70 × free;
  h-spread of φ_T_R/h 5.3 % on 4⁴, 4⁴ vs 4³×8 3.3 %.
- On-configuration asymmetry A(h) = Σ_t (CL_t − CR_t)/Σ_t CL_t: y = 2.0/2.41, h ≥ 0.5: A > 0 at ≥ 4.6σ on all 8 rows;
  A(2) = 0.139–0.153 (free 0.111–0.119); y = 3.0: max |A| = 0.0076, max |pull| 2.3 at every h; 4³×8: |A| < 0.001 at
  h = 0.5 and 2 (free 0.009 / 0.119).
- Light sector at h = 0 relative to free (χ_L/free, G_L(p_min)/free, χ_R/free):

| | y = 2.0 | y = 2.41 | y = 3.0 |
|---|---|---|---|
| 4⁴ | 0.9031, 0.9420, 0.9033 | 0.7067, 0.8315, 0.7092 | 0.0145, 0.0945, 0.0143 |
| 4³×8 | — | 0.6858, 0.7591, 0.6893 | 0.0077, 0.0446, 0.0084 |

- Light t* log-ratio vs h: raw m_L(h)/m_L(0) within 1.3σ of 1 for h ≤ 0.5 on every row; free-normalised r_L within 2.5σ
  of 1 for h ≤ 0.5 except (y = 3.0, 4³×8, h = 0.5): 0.9828(57), −3.0σ (the free O(h p²) shift of +1.2 % is absent in the
  SMG phase); r_L(2) = 0.836–1.093 over (y, lattice); at y = 3.0 χ_L/free ≤ 0.037 and G_L(p_min)/free ≤ 0.115 at every h.
- Quench at fixed σ (10 stored configurations per chain, dense re-measurement at the chain's h and at h = 0): stored
  CL_t, CR_t, φ_f reproduced to 2.8e-17; y = 2.0 4⁴: heavy channel moves 13–23 %, light ≤ 3.5 %; y = 3.0: max
  |Δ_R − Δ_L| = 0.008 (4⁴) / 0.011 (4³×8) over t, both moving 12–20 % (4⁴) and 7–19 % (4³×8).

**Method.** Reader fixed before stage 0 was read (`masspairing.analysis.stage0.load_chain`): log-ratio
m = ln C(t*)/C(t*+1) at t* = L_t/2 − 1 of the P_L / P_R time-slice correlators CL_t, CR_t (positive zero-momentum pair
channels), χ_L = V⟨φ_L²⟩, G_L(p_min) the momentum readout (1 for a free massless doublet), φ_R = Σ_f φ_f (elementary,
C_χ direction), φ_T_R (composite); every number next to the free value at the same L, L_t, bc, h. Quench: dense N1
measure of the stored configurations (`masspairing.measure.N1Measure`, exact) with h = 2 and h = 0 in the operator.

**Pre-registration and deviations.** The reader and the stage-0 order were fixed before the data were read (notebook
commit b60c1d1); ρ was recorded, not a gate. Stage 0 reads differences only; it is not a verdict.

**Controls.** The ρ ordering fails with the y labels 2.0 ↔ 3.0 swapped. The SMG signature (|A| < 0.012 with φ_R < 0.03
free) never occurs at y ≤ 2.41 (h ≥ 0.5), the SYM signature (A > 0.1) never at y = 3.0 (both from one dense propagator
per configuration). With L ↔ R swapped the y = 2.0 selectivity statement fails.

**Caveats.** L = 4 only (large finite-size term), two L_t, a log-ratio gap proxy at t* (cosh-contaminated). ρ at y = 3.0
is resolved on four rows only (φ_R unresolved below h = 1). The quench uses 10 configurations per chain.

**Data.** `data/derived/n1stage/S0/{L4,L4x8,L4pppa}/*.npz` (26 chains, time series only); free baselines
`data/derived/free/free_L4_aaaa.json`, `free_L4x8_aaaa.json`, `free_L4_pppa.json`; stored configurations
`data/configs/n1stage_quench_L4_y2_h2.npz`, `n1stage_quench_L4_y3_h2.npz`, `n1stage_quench_L4x8_y3_h2.npz`. Raw archive:
`results/xi_scan/K_N1_S0/{L4,L4x8,L4pppa}/*.npz`, `results/laneK1/free_L4*.json` (md5 in `data/MANIFEST.tsv`).

**Check.** `python claims/K4.2/check.py` (≈ 4 min single-threaded: 60 dense re-measurements): reads the derived chains,
prints every number above, asserts the ranges, re-measures the stored configurations.

**Provenance.** Notebook claims C190 (gate G4), C191 (unhiggsed_notebook @ 3c4a02c).
