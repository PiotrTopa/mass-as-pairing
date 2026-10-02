# I.7 — the same-parity pairing source: sign-free, exact HMC, linear response

**Statement.** The weight e^{hO_h}, O_h = Σ_planes[Σ_{x even} w Σ_a χ^a(x)χ^a(x+δ) + Σ_{y odd} w Σ_a χ^a(y)χ^a(y+δ′)]
(δ = μ̂ ± ν̂, δ′ = −μ̂ ± ν̂, w the fermionic boundary sign) is K → K − hC₀ in D_c with C₀ real antisymmetric,
same-parity and flavour-blind: the nodal same-parity pairing field (a pairing-channel source, not a Majorana mass).
D_c stays anti-hermitian and τ₂K-symmetric, so det D_c > 0; the RHMC with the source is exact, and ⟨Φ_src⟩ responds
linearly in h with the slope given by the exact h-derivative estimator.

**Numbers.**
1. 4⁴: C₀ antisymmetric, same-parity, 24 entries ±1 per row; D_c(h) = D_c(0) − hC₀ ⊗ 1₂ and D_c† = −D_c (0.0); sparse
   application vs dense 3.7 × 10⁻¹⁵; det D_c > 0 on 10 hot configurations at h = 0.3 (log det 588.1–602.1); log det
   invariant under a two-site time translation of σ (1.1 × 10⁻¹³).
2. 4⁴, h = 0.1: forces vs 4-point finite differences 7.8 × 10⁻⁸; reversibility 4.0 × 10⁻¹⁴ / 7.0 × 10⁻¹³.
3. 2³ × 4 (all antiperiodic), (y, κ, λ) = (2, 0, 1), g₁₀ = 0.4, g₆ = 0.1, h = 0.1 (72 quads, 144 link fields):
   exact-determinant Metropolis (8000 sweeps, det > 0 asserted on every proposal) vs RHMC (2500 trajectories,
   τ-jitter 0.3, acceptance 1.00), 13 observables incl. ⟨Φ_src⟩ = 0.02874(89) vs 0.02783(58): largest pull 1.61σ.
4. Linear response (Metropolis): ⟨Φ_src⟩ = −0.00004(5), 0.01398(32), 0.02874(89), 0.05766(158) at h = 0, 0.05, 0.1, 0.2;
   ⟨Φ_src⟩/h = 0.2795(65), 0.2874(89), 0.2883(79); h = 0 susceptibility ⟨∂φ_src/∂h⟩ + V var(Φ_src) = 0.2660(73);
   pulls −0.72 (two slopes), +1.39, +1.86σ (slopes vs susceptibility).

**Method.** Metropolis and RHMC runs as in I.4 (`masspairing.analysis.k3_exact`), dense channel measure on every
measured configuration; errors from 20 equal blocks.

**Controls.** The odd-sublattice half of C₀ sign-flipped in the operator (the measured direction unchanged) is rejected:
⟨Φ_src⟩ = 0.00007(6), 32σ; even part −0.01376(39), 47σ.

**Pre-registration and deviations.** The lattice is 2³ × 4, not 2⁴: on 2⁴ the source vanishes identically (x + δ and
x − δ are the same site and the two terms cancel).

**Caveats.** Small lattice for the sampler comparison; the 4⁴ tests are per configuration.

**Data.** `data/derived/k3/chains/source_metropolis_h{0,0.05,0.1,0.2}.npz`, `source_rhmc_h0.1.npz`,
`source_rhmc_h0.1_flip.npz` (`scripts/derive_k3.py chains`, about 30 min on one core).

**Check.** `python claims/I.7/check.py` (≈ 50 s): items 1–4; a prefix of every frozen chain used in 3 is regenerated and
must be bit-identical.

**Provenance.** Notebook claim C112 (unhiggsed_notebook @ 3c4a02c).
