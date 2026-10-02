# K2.1 — τ₂K: det D_c > 0 configuration by configuration; the on-site bilinear structure

**Statement.** The doublet operator D_c(σ) = K ⊗ 1₂ + i y σ·τ (K the real antisymmetric reduced staggered kinetic
matrix) is anti-hermitian and commutes with the antiunitary τ₂K; its spectrum is paired as ±iμ, so
det D_c = det(D_c†D_c)^{1/2} is real and positive on every configuration (strictly positive: V is even). This licenses
the RHMC with a single square root. The on-site propagator block S(x,x) = [D_c⁻¹](x,x) is a quaternion i n(x)·τ with n
real, so tr S(x,x) = 0 identically — the flavour-singlet bilinear Σ_a⟨ψ̄^aψ^a⟩_σ vanishes on every configuration — and
the four-fermion condensate O₄(x) = det S(x,x) = |n(x)|² ≥ 0. The bilinears that can be nonzero are the triplet n(x)
(uniform Σ n, staggered Σ ε n). In the free theory (y = 0) the on-site block vanishes.

**Numbers.** 4⁴ (periodic space, antiperiodic time), dense 512 × 512.

| quantity | value |
|---|---|
| y = 0: anti-hermiticity defect; max \|n(x)\| | 0 (exact); 0 |
| y = 1.3, three random σ: max \|Re eigenvalue\| | 9.8 × 10⁻¹⁵ |
| ±iμ pairing defect of the spectrum | 1.1 × 10⁻¹³ |
| det D_c: max \|sin Im log det\|, min cos Im log det, \|Re log det − ½ log det D†D\| | 4.8 × 10⁻¹⁴, 1.0, 5.7 × 10⁻¹⁴ |
| \|Tr D_c⁻¹\|, max \|tr S(x,x)\| | 6.9 × 10⁻¹⁴, 2.9 × 10⁻¹⁵ |
| quaternion defect τ₂S*τ₂ − S; residual of S = i n·τ; residual of det S = \|n\|² | 4.8 × 10⁻¹⁵, 2.4 × 10⁻¹⁵, 1.4 × 10⁻¹⁵ |

**Method.** Dense eigenvalues and inverse of D_c (`masspairing.operator.DoubletOperator.dense`), on-site 2 × 2 blocks and
n_a(x) = −(i/2) tr[τ_a S(x,x)] (`masspairing.algebra.signfree.onsite_structure`).

**Caveats.** The positivity is a theorem (τ₂K, Kramers pairing); the check verifies it on sample configurations. It
holds for any real antisymmetric flavour-blind addition to K (link fields, same-parity sources; K2.3, K2.5) and is
therefore blind to which such pattern is used (K2.5).

**Data.** None.

**Check.** `python claims/K2.1/check.py` (≈ 2 s).

**Provenance.** Notebook claim C040, structure part (unhiggsed_notebook @ 3c4a02c); the code-validation part of C040 is
I.1.
