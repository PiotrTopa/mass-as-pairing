# K1.2 — the unique holomorphic quartic Φ₁₀²; Φ₁₂₆² ≡ 0; 126 ⊗ 126; the unique charge-8 operator

**Statement.** (1) For one Weyl ψ ∈ 16 ⊗ (½,0), the Spin(10) × SL(2,ℂ)-invariant Grassmann quartics of U(1)_ψ charge 4
form a one-dimensional space spanned by Φ₁₀·Φ₁₀ = Σᵢ(ψᵀCΓᵢψ)² (ε^{αβ} Lorentz contraction), a nonzero polynomial. The
metric contraction of the 126 channel, Σ_I Φ_IΦ_I over the 252 five-form bilinears, vanishes identically (self-duality).
The only U(1)_ψ-breaking quartic is therefore in the 10 channel; the 126 channel's only quartic is |Φ₁₂₆|² (K1.3).
(2) Sym²(16) = 10 ⊕ 126, Λ²(16) = 120; 126 ⊗ 126 = 54 ⊕ 945 ⊕ 1050 ⊕ 2772 ⊕ 4125 ⊕ 6930 with
Sym²(126) = 54 ⊕ 1050 ⊕ 2772 ⊕ 4125 and Λ²(126) = 945 ⊕ 6930; 126 ⊗ 126 ∌ 1 and 126 ⊗ 126̄ ∋ 1 (the 126 is complex);
the 54 is real, so 126^{⊗4} contains singlets (3; Sym⁴(126) one) and a bosonic (Δ·Δ)₅₄·(Δ·Δ)₅₄ + h.c. term exists for a
126 Hubbard–Stratonovich field Δ. (3) There is no holomorphic Lorentz-scalar Spin(10) invariant of charge 6, and the
charge-8 space is one-dimensional, spanned by O₈ = (Φ₁₀·Φ₁₀)² ≠ 0. The 126-channel candidate Σ(Φ₁₂₆Φ₁₂₆)₅₄(Φ₁₂₆Φ₁₂₆)₅₄ is
a nonzero multiple of O₈; already at quartic level the 54-plets are Fierz-identical: (Φ₁₂₆Φ₁₂₆)₅₄ = −10 (Φ₁₀Φ₁₀)₅₄. An
8-fermion term reducing U(1)_ψ → ℤ₈ exists, but there is no 126-channel 8-fermion operator distinct from (Φ₁₀·Φ₁₀)²; the
ℤ₈ question is an anomaly question (K1.4), not an operator-existence one.

**Numbers.**

| quantity | value |
|---|---|
| S_{(2,2)}(16) (Lorentz scalars in Λ⁴(16 ⊗ 2)) | 1 ⊕ 54 ⊕ 210 ⊕ 1050 ⊕ 4125 (dim 5440) |
| singlets in S_{(2,2)}(16); in Sym⁴(16), S_{(2,1,1)}(16), Λ⁴(16) | 1; 0, 0, 0 |
| zero-weight monomials in Λ⁴(32); kernel of the 40 + 2 root vectors (Gram spectrum head 0, 20, 20) | 240; dimension 1 |
| Φ₁₀·Φ₁₀: monomials, norm; \|Gram v\|/\|v\| | 240, 350.542437; 0 |
| Σ_I Φ_IΦ_I over 252 five-forms: norm (reference scale) | 0.0 (22 084.174) |
| singlets in 126⊗126, 126⊗126̄, 54⊗54, 126^{⊗4}, Sym⁴(126) | 0, 1, 1, 3, 1 |
| holomorphic Lorentz-scalar invariants of charge 6, 8 | 0, 1 |
| rank of the 55 (Φ₁₀Φ₁₀)₅₄ and (Φ₁₂₆Φ₁₂₆)₅₄ quartics: separately, together | 54, 54, 54 |
| Fierz ratio (Φ₁₂₆Φ₁₂₆)₅₄ / (Φ₁₀Φ₁₀)₅₄ (identical polynomials) | −10 |
| octics (Φ₁₀·Φ₁₀)², Σ(Φ₁₀Φ₁₀)₅₄², Σ(Φ₁₂₆Φ₁₂₆)₅₄²: norms; rank | 170 489.081, 153 440.173, 15 344 017.285; 1 |

**Method.** (a) Cauchy: the Lorentz scalars in Λ^{2k}(16 ⊗ ℂ²) are S_{(k,k)}(16); characters by Jacobi–Trudi from exact
weight multisets, decomposed with the Weyl-group multiplicity formula (`masspairing.algebra.characters`). (b)
Independently: so(10) ⊕ sl(2) invariants of Λ⁴(32) as the joint kernel of all root vectors on the zero-weight subspace
(Gram matrix of the derivation matrices). (c) Sparse Grassmann polynomials in the weight basis of the 16
(`masspairing.algebra.grassmann`): Φ₁₀·Φ₁₀, Σ_IΦ_IΦ_I, the 54-plets (traceless symmetric parts) of both channels and the
three octics; ranks by SVD.

**Caveats.** None (exact algebra; the polynomial identities hold to machine precision with the stated reference scales).

**Data.** None.

**Check.** `python claims/K1.2/check.py` (≈ 1 min, of which ≈ 40 s for the charge-8 character).

**Provenance.** Notebook claims C005, C006 (unhiggsed_notebook @ 3c4a02c).
