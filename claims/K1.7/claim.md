# K1.7 — ℤ₄² = (−1)^F; the partner algebra of a symmetric gap; the composite partner ψ̄ψ̄ψ

**Statement.** (1) The anomaly-free ℤ₄ (U(1)_ψ at angle π/2 on the 16; on the lattice U(1)_ε at π/2, i on even and −i on
odd sites) squares to fermion parity. Hence every fermionic operator ψⁿψ̄ᵐ (n + m odd), elementary or composite, has
odd ℤ₄ charge and every Majorana bilinear O·O of a fermionic operator has charge 2 mod 4: a ℤ₄-symmetric phase has
⟨O_x O_z⟩ ≡ 0 for every fermionic O, and only Dirac-type pairings of charge-conjugate partners (q, −q mod 4) are allowed.
(2) For a fermion χ with Dirac mixing Δ to a partner Ψ that may carry a Majorana mass M_p, G⁻¹ = [[ip, Δ], [Δ, ip + M_p]]:
G_χχ(p → 0) = −M_p/Δ² (finite iff the partner has a Majorana mass); for M_p = 0, |G_χχ(p)| = |p|/(p² + Δ²) (a zero at
p = 0 with gap Δ); for M_p ≫ Δ the light eigenvalue is Δ²/M_p. Higgs–Dirac (elementary partner, M_p = 0), seesaw
(elementary partner, M_p ≫ Δ) and symmetric mass generation (composite partner, M_p = 0 forced by (1)) are the three
corners of this one matrix. (3) The Dirac partner of ψ ∈ 16 ⊗ (½,0) must be a 16̄ ⊗ (½,0) of charge −1 mod 4: ψ̄ψ̄ψ
contains it 3 times and ψψψ (charge 3) 4 times; the pairing ψ·(ψ̄ψ̄ψ) has 3 invariants at character level and 2 as
Grassmann polynomials — the two quartics |Φ₁₀|², |Φ₁₂₆|² of K1.3 — so the partner pairing is the four-fermion vertex.
Sym²(16̄) = 10 ⊕ 126: the composite Majorana mass ΨΨ lives in the same channels as the elementary ψψ. (4) Lattice
(SU(4) ≅ Spin(6)): the ε-vertex partner of χ^a is Ψ^a = ε_abcd χ^bχ^cχ^d ∈ Λ³(4) = 4̄; χ^aΨ^a ∈ 4 ⊗ 4̄ ∋ 1 once (the
vertex, charge 4 ≡ 0); the two-site composite pair Ψ_xΨ_z ∈ Sym²(4̄) ⊕ Λ²(4̄) = 10̄ ⊕ 6 (charge 6 ≡ 2) — the same channels
as the elementary pair χ_xχ_z ∈ 10 ⊕ 6, at degree 6 instead of 2.

**Numbers.**

| quantity | value |
|---|---|
| G_χχ(p = 10⁻⁶) vs −M_p/Δ², 200 random (Δ, M_p) ∈ [0.2, 2]² | max rel. deviation 3.1 × 10⁻⁵ |
| M_p = 0: \|G_χχ(p)\| vs \|p\|/(p² + Δ²), p = 10⁻³, 10⁻², 10⁻¹ | max rel. deviation 4.6 × 10⁻¹⁶ |
| M_p = 1000 × [0.2, 2]: light eigenvalue vs Δ²/M_p | max rel. deviation 5.5 × 10⁻⁵ |
| multiplicity of 16̄ ⊗ (½,0) in ψ̄ψ̄ψ, ψψψ | 3, 4 |
| invariants of ψ·(ψ̄ψ̄ψ): characters, Grassmann | 3, 2 |
| Sym²(16̄) | 10 ⊕ 126 |
| Λ³(4); singlets in 4 ⊗ 4̄; Sym²(4̄), Λ²(4̄); Sym²(4), Λ²(4) | 4̄; 1; 10̄, 6; 10, 6 |

**Method.** Charge arithmetic on degrees; 2 × 2 matrix inversion at fixed seed; exact D5 × A1 × A1 and D3 characters
(`masspairing.algebra.characters`).

**Caveats.** Green's-function zeros of symmetric-mass-generation phases are known (Lu–Zeng–You, PRB 108, 205117 (2023);
Golterman–Shamir, PRL 132, 081903 (2024)); this claim certifies the partner bookkeeping, not a dynamical statement.

**Data.** None.

**Check.** `python claims/K1.7/check.py` (≈ 1 s).

**Provenance.** Notebook claim C149 (unhiggsed_notebook @ 3c4a02c).
