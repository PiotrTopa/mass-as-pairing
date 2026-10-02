# K1.1 — 16 ⊗ 16 bilinears of Spin(10): Sym² = 10 ⊕ 126, Λ² = 120; ranks of the 126 and 10 mass matrices

**Statement.** On one Weyl 16 of Spin(10) the Lorentz-scalar bilinears ψᵀCΓ^{[k]}ψ span a 10-dimensional symmetric
space for k = 1, a 120-dimensional antisymmetric space for k = 3 and a 126-dimensional symmetric space for k = 5, and
vanish for k = 0, 2, 4. Hence Sym²(16) = 10 ⊕ 126 and Λ²(16) = 120: there is no bare Majorana mass, the Majorana mass
channels are the 10 and the 126. A generic 126 vacuum expectation value gives a mass matrix of full rank 16; the
SU(5)-singlet direction of the 126 has rank 1 (one Majorana mass, the right-handed neutrino, 15 massless); a real 10
vacuum expectation value gives all 16 Weyl fermions the same mass (M†M ∝ |v|² 1₁₆).

**Numbers.**

| quantity | value |
|---|---|
| span dimension, k = 0 … 5 | 0, 10 (sym), 0, 120 (antisym), 0, 126 (sym) |
| generic 126 vev: rank over 500 random complex five-form coefficient vectors | 16 (all samples) |
| smallest normalised singular value s₁₆/‖M‖ over the 500 samples | 9.83 × 10⁻⁴ |
| SU(5)-singlet 126 vev Ω = ∧ₖ(e₂ₖ + i e₂ₖ₊₁): rank | 1 |
| real 10 vev, 20 random v ∈ ℝ¹⁰: relative spread of the 16 singular values | ≤ 1.0 × 10⁻¹⁵ (rank 16) |

**Method.** 32-dimensional Dirac representation from Pauli tensors (Γ₂ₖ real symmetric, Γ₂ₖ₊₁ imaginary
antisymmetric), C = Γ₀Γ₂Γ₄Γ₆Γ₈ (CΓᵢ = Γᵢᵀ C), the Weyl 16 as the +1 eigenspace of the chirality; the 16 × 16 blocks
PᵀCΓ^{[k]}P of all ordered k-index products (`masspairing.algebra.spin10`); ranks by SVD with tolerance 10⁻⁸.

**Caveats.** The generic-rank statement is sampled (500 Gaussian draws, fixed seed), not proved; the minimum
normalised singular value is a property of the sample.

**Data.** None (exact linear algebra).

**Check.** `python claims/K1.1/check.py` (≈ 1 s): spans and symmetry for k = 0 … 5, the 500-sample 126 rank and its
smallest normalised singular value, the rank-1 SU(5) singlet, the degenerate 10 spectrum.

**Provenance.** Notebook claims C001, C002, C003, C004 (unhiggsed_notebook @ 3c4a02c).
