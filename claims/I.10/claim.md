# I.10 — N1 RHMC exact vs exact-determinant Metropolis; the phase-flipped pattern is rejected

**Statement.** The N1 instrument — the flavour-blind taste-chiral Majorana mass h C_χ (K2.5, K5.1) in the RHMC — is
exact: the operator D_c(h) = D_c(0) − h C_χ ⊗ 1₂ is applied correctly by the merged sparse stencil, the action and the
forces are correct, and the RHMC reproduces an exact-determinant Metropolis on 20 observables of the N1 measure on the
4⁴ all-antiperiodic box (full spatial hopping, no zone-boundary mode, so every taste observable is non-trivial). The
same RHMC with the phase-flipped pattern — sign-free (det > 0) but not a chiral mass (K5.1) — is rejected: a wrong
pattern that the positivity of the weight cannot see is caught by this comparison and by the mass test of K5.1.

**Numbers.**
1. Operator, 4⁴ with production and all-antiperiodic bc, ε model and wedge g₁₀ = 0.1 (12 hot configurations): C_χ has 8
   entries per row, merged with K 16 neighbours; D_c(h) = D_c(0) − h C_χ ⊗ 1₂ and D† = −D exactly (0.0); the merged
   operator applies D and D† to single vectors and batches as the dense matrix (≤ 4e-15); the flavour-selective path with
   the all-flavour mask is the same operator (0.0); det D_c > 0 on all configurations and on the stored 4⁴ P_c
   configuration at h = 1 (log det 504.5 … 598.9).
2. 4⁴ wedge (g₁₀, g₆) = (0.25, 0.05), h = 0.3 chiral: Lanczos S_pf vs dense φ†(D†D)^{−1/2}φ 8.5e-14; σ- and s-forces vs
   4-point finite differences 7.7e-8; reversibility 2.7e-14 / 1.4e-13.
3. Exact-determinant Metropolis vs RHMC, 4⁴ all-antiperiodic, ε model (y, κ, λ) = (2, −0.01, 1), h = 0.3 C_χ, 20
   observables (ε channel; heavy and light pair channels with the flavour-summed susceptibilities; the light (6,1,1)
   channel; the light/heavy time-slice correlators and momentum readouts; the composites): largest pull 2.08σ (O4);
   e.g. φ_R 0.00840(2) vs 0.00841(2), chi_L_sum 0.02821(3) vs 0.02815(2), GL_p0 0.9420(8) vs 0.9411(6). The Woodbury
   determinant ratios are real to 4.8e-14.
4. Phase-flipped pattern in the same RHMC (500 trajectories): rejected at 107σ — φ_R halves (0.00420(3)), chi_R_sum falls
   by 30 % (0.03256(10) vs 0.04609(11)), and the light amplitude φ_L becomes −0.00417(4) (the non-chiral mass gaps the
   light half; the taste split makes the wrong pattern visible in the light channel).

**Method.** Metropolis: single-site Gaussian proposals of σ (width 0.6); determinant ratio det(1₂ + Δ D⁻¹_xx) of the
rank-2 site update; D⁻¹ kept by Woodbury updates and recomputed every 50 sweeps (agreement asserted to 1e-8); two chains
of 5000 sweeps, one measurement every 8 sweeps after sweep 300 (587 per chain). RHMC: τ = 1 with ±30 % jitter, 8
Omelyan steps, 200 thermalisation trajectories, 2000 measured trajectories (flipped pattern: 500). Errors: 20 equal
blocks of the concatenated Metropolis series (≈ 470 sweeps each; the local Metropolis has an autocorrelation time of
O(100) sweeps on the σ zero mode) and of the RHMC series. Exact (dense) N1 measurements.

**Pre-registration and deviations.** Criteria: every pull ≤ 3σ, the flipped pattern rejected at > 5σ. A first run with
2 × 2500 sweeps in blocks of 110 sweeps gave 3.3–3.5σ pulls on the slowest quantities (chi_L_sum, GL_p0, GR_p0); the
certified run uses 2 × 5000 sweeps and blocks of ≈ 470 sweeps, with the criteria unchanged.

**Controls.** The phase-flipped pattern (item 4). Every observable has non-zero variance in both chains (non-trivial on
this box).

**Caveats.** CPU reference path only. One box (4⁴ aaaa) and one coupling point; the comparison certifies the algorithm
and the measure, not the physics of a phase.

**Data.** `data/derived/n1inst/I10_met0.npz`, `I10_met1.npz`, `I10_rhmc.npz`, `I10_flipped.npz` (series, acceptance, dH of
every trajectory, the boson-action trace of the Metropolis sweeps, the last measured configuration;
`scripts/derive_n1inst.py i10`, a few hours on one core); `data/configs/n1inst/L4_y2.41_eps_pppa.npz` (archive
`results/xi_scan/F8_P3_eps/L4_y2.41_k-0.01_g0_g60.npz`).

**Check.** `python claims/I.10/check.py` (≈ 30 s): items 1–2 live; items 3–4 from the frozen chains, with the first 20
Metropolis sweeps and the first 10 trajectories of both RHMC chains regenerated live and required bit-identical, and the
last measured configuration of every chain re-measured.

**Provenance.** Notebook claim C161 (CPU part) (unhiggsed_notebook @ 3c4a02c).
