# K1.3 — Fierz basis of the ψψψ̄ψ̄ quartics of one 16: {|Φ₁₀|², |Φ₁₂₆|²}

**Statement.** The Lorentz-scalar, Spin(10)- and U(1)_ψ-invariant quartics of type ψψψ̄ψ̄ of one Weyl ψ ∈ 16 ⊗ (½,0)
form a two-dimensional space, spanned by |Φ₁₀|² and |Φ₁₂₆|². Λ²(16 ⊗ (½,0)) = (10,(0,0)) ⊕ (126,(0,0)) ⊕ (120,(1,0));
the 120 piece pairs only with a (0,1) and gives no Lorentz scalar. The three current–current terms J_A·J_A
(16 ⊗ 16̄ = 1 ⊕ 45 ⊕ 210) are linear combinations of the two. Together with K1.2 (the unique holomorphic quartic
Φ₁₀² + h.c.), the two-channel Lagrangian −g₁₀|Φ₁₀|² − g₁₂₆|Φ₁₂₆|² − g′(Φ₁₀² + h.c.) is the most general Lorentz-scalar
Spin(10)-invariant four-fermion Lagrangian of one 16.

**Numbers.**

| quantity | value |
|---|---|
| invariants in Λ²V ⊗ Λ²V̄ (D5 × A1 × A1 characters) | 2 |
| rank of {\|Φ₁₀\|², \|Φ₁₂₆\|², J₁·J₁, J₄₅·J₄₅, J₂₁₀·J₂₁₀} as Grassmann polynomials | 2 (singular values 1437.14, 206.20, then 0) |
| max relative violation of so(10) ⊕ sl(2,ℂ) invariance | ≤ 1 × 10⁻¹⁴ |
| J₁·J₁ | −(1/256)\|Φ₁₀\|² − (1/512)\|Φ₁₂₆\|² |
| J₄₅·J₄₅ | (27/256)\|Φ₁₀\|² − (5/512)\|Φ₁₂₆\|² |
| J₂₁₀·J₂₁₀ | −(21/128)\|Φ₁₀\|² − (5/256)\|Φ₁₂₆\|² |
| inverse | \|Φ₁₀\|² = −40 J₁·J₁ + 8 J₄₅·J₄₅; \|Φ₁₂₆\|² = −432 J₁·J₁ − 16 J₄₅·J₄₅ |

All relation residuals are ≤ 10⁻¹⁴ (relative); the coefficients equal the stated rationals to 10⁻¹⁰.

**Method.** (1) Exact weight-multiset characters under D5 × A1 × A1 (complexified Lorentz group) and the Weyl-group
multiplicity formula (`masspairing.algebra.characters`). (2) Explicit sparse Grassmann polynomials on 64 generators
(ψ_{aα} ↦ 2a + α, ψ̄_{aα̇} ↦ 32 + 2a + α̇, weight basis of the 16): |Φ_A|² = Σ_A Φ_A Φ̄_A over the 10 vector and the 252
five-form bilinears, J_A·J_A = Σ_A ε^{α̇β̇}ε^{αβ} J^A J^{A†} with generators orthonormal under Tr(T†T), T¹ = 1/4.
(3) Invariance: vanishing of the derivations of six random so(10) elements and of the six real generators of
sl(2,ℂ). (4) Rank and least-squares relations. |Φ₁₂₆|² is summed over all 252 five-forms (twice a self-dual basis).

**Caveats.** The ψψψ̄ψ̄ type is fixed (U(1)_ψ charge 0); the charge-±4 quartics are K1.2.

**Data.** None (exact algebra).

**Check.** `python claims/K1.3/check.py` (≈ 1 min): the character count, the five Grassmann quartics, invariance,
rank 2, the relation coefficients against the exact rationals and the inverse relations.

**Provenance.** Notebook claim C009 (unhiggsed_notebook @ 3c4a02c).
