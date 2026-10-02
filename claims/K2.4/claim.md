# K2.4 — the sign-free class is quaternion-linear, hence flavour-democratic

**Statement.** The sign-free class of K2.1 is the set of real-linear operators on the doublet ψ ∈ ℂ^{2V} ≅ ℝ^{4V} that
commute with J_i (multiplication by i) and with the antiunitary τ₂K (J_τ: ψ → τ₂ψ̄). (1) J_i² = J_τ² = −1, {J_i, J_τ} = 0:
they generate a quaternion algebra ℍ on the four real flavours of every site; its commutant in M₄(ℝ) is
span{1, Γ₁, Γ₂, Γ₃} (the σ-coupled units), and the class is exactly ℍ^op ⊗ ℝ^{V×V} = span{1, Γ_a} ⊗ ℝ^{V×V}: every
sign-free operator is quaternion-linear. K ⊗ 1, the Yukawa term yσ·Γ, the link-field term and every flavour-blind
1 ⊗ C are in it. (2) Every class element commutes with the other triplet Γ′ (the anti-self-dual units, the hidden
SU(2)_R of K2.1), which acts on flavour space as right quaternion multiplication; the unit quaternions c₀ + c·Γ′ act
transitively on unit flavour vectors. Hence the restriction (uᵀ ⊗ 1)M(u ⊗ 1) of any class element to any flavour
direction u has a u-independent spectrum: every sign-free source or HS field gives every flavour direction the same
Majorana amplitude and the same spectrum. (3) An antisymmetric class element is 1 ⊗ A (A antisymmetric) +
Σ_a Γ_a ⊗ S_a (S_a symmetric), so every flavour's own Majorana block is the same A; the orthogonal projection of
diag(m) ⊗ C onto the class is m̄ · 1 ⊗ C. The flavour-blind 1 ⊗ C_χ and 1 ⊗ C₀ are inside; the 1-of-4 split, the 2+2
split and any non-democratic diag(m) ⊗ C are outside; the flavour-selective source does not commute with Γ′ ⊗ 1.
(4) The taste-chiral mass, flavour-blind, gives every flavour direction |m| = h on the same doublet: democracy is
compatible with chirality.

**Numbers.**

| quantity | value |
|---|---|
| commutant of ℍ in M₄(ℝ): dimension; residual of 1, Γ_a in it | 4; 1.8 × 10⁻¹⁵ |
| commutant of {J_i ⊗ 1, J_τ ⊗ 1} on 2⁴ (4V = 64) | 1024 = 4V² |
| random A ⊗ Γ_k and K ⊗ 1 in the class (residual); norm fraction of a random matrix kept | 2.5 × 10⁻¹⁴; 0.50 |
| realified D_c on 4⁴ (random σ, link fields, chiral mass h = 0.3): \|[M, Γ′ ⊗ 1]\|; spectra of flavour 1, 4, random u | 0; 3.6 × 10⁻¹⁴ |
| same on the stored 4⁴ P_c configuration (y = 2.41, κ = −0.01) | 0; 1.3 × 10⁻¹⁵ |
| transitivity of SU(2)′ on unit flavour vectors (residual) | 9.8 × 10⁻¹⁸ |
| distance from the class: 1 ⊗ C_χ, 1 ⊗ C₀ | 0, 0 |
| C₀ ⊗ diag(0,0,0,1); C_χ ⊗ diag(0,0,1,1); C_χ ⊗ diag(0.2, 0.5, 1, 1.3) | 0.8660 (= √3/2); 0.7071 (= 1/√2); 0.4949 (= ‖m − m̄‖/‖m‖) |
| \|[C₀ ⊗ diag(0,0,0,1), Γ′ ⊗ 1]\| | 1.00 |
| K − hC_χ restricted to flavour 1, 4, random u (periodic 4⁴, h = 0.5): 8 zeros + 8 at h | max deviation 1.2 × 10⁻⁷ |

**Method.** Real 4 × 4 matrices of J_i, J_τ (`masspairing.algebra.signfree`), commutants as null spaces of the
commutator equations, the class distance as the relative norm of the component orthogonal to span{1, Γ_a} ⊗ ℝ^{V×V};
realified operators M = Re[W D_c W†] (`masspairing.flavour.realify`).

**Caveats.** "Democratic" concerns the exactly sign-free class; a flavour-selective source can be simulated with the
|Pf| weight (I.8), whose sign is a separate question (I.9).

**Data.** `data/derived/algebra/eps_L4_y2.41_k-0.01_final.npz` (final configuration of the 4⁴ ε-model chain at P_c;
raw archive `results/xi_scan/F8_P3_eps/L4_y2.41_k-0.01_g0_g60.npz`). Table: `results/algebra_democracy.json`
(`scripts/table_algebra_democracy.py`).

**Check.** `python claims/K2.4/check.py` (≈ 2 min).

**Provenance.** Notebook claim C165 (unhiggsed_notebook @ 3c4a02c).
