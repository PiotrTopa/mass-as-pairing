# I.5 — link-field HMC resonance at ω_j τ ≈ nπ; trajectory-length jitter removes it

**Statement.** In the link-field HMC each pair field s_{Q,j} has the Gaussian prior s²/(4g_j) and unit mass, so it is a
harmonic oscillator of frequency ω_j = (2g_j)^{−1/2} on which the fermion force is a small perturbation (the exact
conditional distribution of a single field on equilibrium configurations has variance 2g_j within 3 %). A trajectory of
length τ maps x → x cos ω_jτ + (p/ω_j) sin ω_jτ; with fresh momenta ⟨x²⟩_n/2g_j = 1 − cos^{2n}(ω_jτ) from x = 0. At
ω_jτ = nπ the amplitude never changes (an exact non-ergodicity of the free part), and near it the mixing per trajectory
is sin²(ω_jτ). With τ = 1 the coupling g_j = 0.05 sits at ω_jτ = 3.162 (sin² = 0.0004): fixed-τ chains there do not
thermalise their link fields. The resonances are at g_j = 1/(2n²π²) = 0.0507, 0.0127, 0.0056 (n = 1, 2, 3). Drawing
τ = τ₀(1 + j u), u ~ U(−1, 1), from a state-independent stream before every trajectory is exact and removes the
freezing; with j = 0.3 no g_j in [0.005, 1] keeps sin²(ω_jτ) < 0.1 on average. The chains of K3.1–K3.5 carry j = 0.3;
the fixed-τ chains of K3.6 have g_j ∈ {0.1, 0.2}, off the resonances.

**Numbers.**

| stored 4⁴ chain (y, g₁₀, g₆) | ω τ | ⟨s²⟩/2g first → last quarter | exact conditional variance / 2g |
|---|---|---|---|
| (1, 0.05, 0) | 3.162 | 0.417 → 0.526 | 0.997 |
| (1.7, 0.05, 0) | 3.162 | 0.434 → 0.529 | 1.009 |
| (2.41, 0.05, 0) | 3.162 | 0.406 → 0.496 | 1.014 |
| (3.0, 0.05, 0) | 3.162 | 0.400 → 0.472 | 0.999 |
| (2.41, 0.02, 0) control | 5.000 | 1.018 → 1.021 | 1.003 |
| (2.41, 0.05, +0.05) control | 2.236 | 1.091 → 1.112 | 0.991 |

- Free-oscillator ramp ⟨x²⟩_n/2g at n = 1, 60, 150, 460: 0.000, 0.025, 0.062, 0.179 (ωτ = 3.162); with the Omelyan
  phase error (ωτ_eff ≈ 3.175) 0.001, 0.066, 0.157, 0.407 — the observed ramp of the stored chains.
- Per-trajectory mixing sin²(ωτ) at g = 0.05: 0.0004 (τ = 1.0), 0.641 (τ = 0.7).
- Live 4⁴ chains (y = 1, g₁₀ = 0.05, g₆ = 0, 150 trajectories from s = 0, 10 steps): τ = 1.0 gives ⟨s²⟩/2g = 0.375 /
  0.421 (first / second half, acceptance 0.99); τ = 0.7 gives 1.044 / 1.049 (acceptance 1.00).

**Method.** Stored chains: 400 trajectories each, τ = 1, the stored final configuration; conditional variance of 4
random link fields from a 33-point dense log-determinant scan over ±4√(2g_j) at fixed other fields. Live chains:
`scripts/run_chain.py` (seed 71), compared bit by bit with the stored runs.

**Controls.** The controls at ωτ = 5.0 and 2.24 are equilibrated (0.9–1.2) with the same exact conditional variance;
the live τ = 1.0 and τ = 0.7 chains separate (< 0.6 vs > 0.9) with identical code, start and seed.

**Caveats.** The 4⁴ chains at g₁₀ = 0.05, g₆ = 0 with τ = 1 sampled their fermionic observables with link-field
fluctuations at about half the correct variance; no claim of this repository uses such chains.

**Data.** `data/derived/k3/c072/`, `c072_n20/` (series ⟨s²⟩ of the stored chains), `data/configs/k3/c072*.npz` (their
final configurations), `data/derived/k3/c102_tau1.0/`, `c102_tau0.7/` (the stored live runs); archive
`results/laneW2/c072/`, `results/laneR/c102/`.

**Check.** `python claims/I.5/check.py` (≈ 3 min, two 150-trajectory 4⁴ chains).

**Provenance.** Notebook claim C102 (unhiggsed_notebook @ 3c4a02c).
