# K2.3 — anatomy of the completed (wedge) term; wedge optimality; a sign-free same-parity source

**Statement.** Setting: the completed two-channel term −S ⊃ Σ_Q [g₁(K̃_ν + K̃_ν′)² + g₂(K̃_μ + K̃_μ′)²], g₁ = g₁₀ − g₆,
g₂ = g₁₀ + g₆, whose fermionic content is g₁₀T_Q + g₆T₆ + (g₁₀ − g₆)(K_ν² + K_ν′²) + (g₁₀ + g₆)(K_μ² + K_μ′²) (K2.2;
T_Q = Σ_ab Φ̄Φ the 10-channel plaquette term, T₆ = Σ_ab Λ̄Λ its Λ² partner, K̃_l = s_l K_l, s_νs_ν′ = +1, s_μs_μ′ = −1).
(1) *Edge identities.* On every quad T_Q + T₆ = 4K̃_μK̃_μ′ and T_Q − T₆ = 4K̃_νK̃_ν′: on the wedge edges g₆ = ±g₁₀ the
10-channel term is not a separate operator; the +edge model is 2g₁₀Σ_Q(K̃_μ + K̃_μ′)² = 4g₁₀K̃_μK̃_μ′ + 2g₁₀(K_μ² + K_μ′²),
a parallel-bond attraction plus bond exchange on one pair of parallel links per quad. (2) *Wedge optimality.* In
e^{Σ a_l K̃_l} the two-flavour same-link monomial χ¹_uχ¹_vχ²_uχ²_v has coefficient a_l² and the two-link monomial
a_l a_l′ s_l s_l′; in e^{gT_Q + Σ d_l K_l²} they are 2d_l and 2g s s′ (ν and μ pairs; 0 for ν–μ). Any probability measure
on real flavour-blind link fields reproducing the completed term therefore has E[a_l²] = 2d_l, E[a_ν a_ν′] = 2g s_νs_ν′,
and Cauchy–Schwarz gives d_νd_ν′ ≥ g², d_μd_μ′ ≥ g²: no positive real flavour-blind decoupling reaches below the wedge
edge, and the Gaussian one saturates the bound. (3) *Symmetry patterns.* The columnar dimer order parameter
D_μμ = mean_{μ-links}(−1)^{x_μ}K̃ is invariant under U(1)_ε, the ℤ₄ generator, SU(4) and the one-site shift along the
other direction, and odd under the one-site shift along μ. The Sym²(4) order parameter Φ^{ab}(x, x+δ) (even sites) has
U(1)_ε charge 2, is odd under the ℤ₄ generator, transforms in the 10 of SU(4) (Σ Φ̄Φ invariant) and is mapped by a
one-site shift to Φ̄^{ab} on the odd sublattice (the 10̄). The dimer breaks lattice translations and no internal
symmetry; the Sym² condensate breaks U(1)_ε (ℤ₄ → ℤ₂ with the ε term) and SU(4). (4) *Free-theory weights* on 4⁴ per
quad: four fifths of ⟨T_Q⟩₀ is the product of two mean bond energies and one fifth is exchange (the only genuine
two-pair correlation); the completion Σ_l⟨K_l²⟩₀ is three times the exchange part. (5) *A sign-free same-parity source.*
With C a same-parity (δ = μ̂ ± ν̂), flavour-blind, real antisymmetric matrix, D_c(σ, s) + C ⊗ 1₂ keeps the τ₂K structure:
det is real and positive.

**Numbers.**

| quantity | value |
|---|---|
| edge identities, all 16 quads of 4² (both orientations): max coefficient error | 0 |
| moment bookkeeping (exact coefficients on a quad) | holds |
| D_00 under U(1)_ε, ℤ₄, SU(4), shift₀, shift₁; D_11 under shift₀, shift₁ | 1, 1, 1, −1, 1; 1, −1 |
| Φ⁰¹ under U(1)_ε (α = 0.7), ℤ₄; shift₀ → Φ̄⁰¹ (modulus) | e^{1.4i} = 0.17 + 0.985i, −1; 1 |
| free 4⁴: ⟨K̃⟩₀ | 0.5000000000 |
| ⟨T_Q⟩₀ = bond product + exchange | 1.3075 = 1.0460 + 0.2615 |
| ⟨T₆⟩₀; completion Σ_{l∈Q}⟨K_l²⟩₀ | −0.2087; 0.7845 |
| exchange / T_Q; completion / exchange | 0.200; 3.000 |
| det with a random same-parity source (amplitude 0.3), 10 hot 4⁴ configurations: max \|sign − 1\| | 2.5 × 10⁻¹⁴ |

**Method.** Exact Grassmann algebra on 4² (antiperiodic for the Wick and edge identities, periodic for the symmetry
table, where the one-site shift χ(x) → ζ_μ(x)χ(x + μ̂), ζ_μ(x) = (−1)^{Σ_{ν>μ} x_ν}, needs no boundary sign; orientation
s_l = −sign M_uv). Free weights from the exact 4⁴ doublet propagator (y = 0) realified to the four flavours
(`masspairing.flavour.real_flavour_block`), per quad over the link geometry of `masspairing.wedge.WedgeGeometry`.
Signs by `slogdet` of the dense operator with link fields (g₁₀ = 0.1, g₆ = 0) at y = 2.41.

**Caveats.** Item 5 shows only that such a source is sign-free; its response is a separate measurement (K3.4). The
τ₂K sign test is pattern-blind (K2.5): it does not certify that a given pattern is a mass.

**Data.** None.

**Check.** `python claims/K2.3/check.py` (≈ 10 s).

**Provenance.** Notebook claim C100 (unhiggsed_notebook @ 3c4a02c).
