# K2.5 — C₀ is a nodal pairing (no corner mass, also to O(h⁵)); C_χ is the taste-chiral Majorana mass (ε convention)

**Statement.** Reduced staggered fermions have 16 zero modes per flavour with periodic boundary conditions (the four
Weyl fermions of the flavour at the 16 Brillouin-zone corners p ∈ {0, π}⁴); a same-parity pattern C is a mass iff
K − hC lifts them. (1) *C₀ is nodal.* The flavour-blind same-parity pattern C₀ (`patterns.source_pattern`, the
pair operators χ(x)χ(x + μ̂ ± ν̂) without staggered phase) annihilates the 16 zero modes: they stay exact zero modes of
K − hC₀ at every h (L = 4, 6, 8). The interaction cannot turn it into a corner mass either: under the full lattice
symmetry group of the interacting model (one-site shifts × spatial permutations × reflections × time reflection, 4⁴,
production boundary conditions) the irreducible representations of the six plane masses — the only same-parity,
U(1)_ε-charge-2 corner masses, and exactly the group orbit of C_χ — occur with multiplicity 0 in Sym¹, Sym³ and Sym⁵ of
the orbit span of C₀: no induced corner mass at any odd order ≤ 5 in h. The lowest singular value of K − hC₀ is
h-independent with antiperiodic time only (the lowest free mode has momentum along the time axis, never a "μ" of the
pattern); with antiperiodic x it moves at every h, and with all-antiperiodic bc C₀ lowers it — C₀ acts on modes well
below the cutoff, it is a pairing field, not a mass. (2) *Corner space.* Near the corners K ≈ iΣ_μ k_μΓ_μ with real
Pauli strings Γ_μ = X_{S_μ}Z_{μ}; the strings X_SZ_D anticommuting with all four Γ_μ (scalar masses) are 16: four
one-link (Dirac-type, U(1)_ε charge 0) and, among the antisymmetric two-link ones, exactly one per plane {μ,ν}, with the
phase set PH = {(0,1): (0,2,3), (0,2): (0,3), (0,3): (0), (1,2): (0,1,3), (1,3): (0,1), (2,3): (0,1,2)}. Each plane
operator has a 16-fold degenerate corner block (value 4); a plane commutes with its dual and anticommutes with the other
four. (3) *The chiral mass, ε convention.* C_χ(comp, s) = [C_plane(comp) + s ε_{μνρσ} C_plane(dual)]/8 with
ε₀₁₂₃ = ε₀₃₁₂ = +1, ε₀₂₁₃ = −1. The three self-dual components (0,1)+, (0,2)+, (0,3)+ gap the same taste doublet (their
corner Q_D = ε J_D J_dual coincide; principal cosines of the light subspaces 1), the anti-self-dual ones the other; the
combination without ε, [C(0,2) + C(1,3)]/8, is (0,2)− (Q(0,1) = Q(0,3) = −Q(0,2) without ε). The heavy-doublet triplet is
{P01+P23, P02−P13, P03+P12} (mutually anticommuting). With periodic bc K − hC_χ has exactly 8 zero singular values and 8
at h (L = 4, 6, 8): a Majorana mass for one taste doublet of the flavour, the other doublet exactly massless. With the
production bc the lowest level splits into 16 light modes within 0.05h² of the free value and 16 heavy modes
≥ √(s₀² + (0.8 min(h,1))²), monotone in h. The spectrum is isotropic (antiperiodic t vs x) and the same for all
components. Contrasts: the naive η_ν-phased plane sum has a level crossing; a one-sided (even-only) C_χ leaves
det(K − hC) = det K (not a mass). (4) *Sign.* Flavour-blind, C ⊗ 1₂ is real antisymmetric, so D_c(σ) − hC ⊗ 1₂ is in the
τ₂K class (K2.1): det > 0 for C_χ — and equally for a phase-flipped C_χ and for C₀. The sign check is pattern-blind;
the phase-flipped pattern is caught only by the mass test (16 modes at h/2 instead of 8 zeros + 8 at h).

**Numbers.**

| quantity | value |
|---|---|
| periodic L = 4, 6, 8: zero modes of K; \|C₀P₀\|; zero modes of K − hC₀ (h = 0.1, 0.5, 1, 2) | 16; ≤ 1.0 × 10⁻¹⁴; 16 at every h |
| s_min(K − hC₀), h = 0.1 … 2, L = 4: antiperiodic t / x / all | 1.4142 constant / 1.44–2.45 / 2.83 → min 1.44 |
| same, L = 6 | 1.0000 constant / 1.00–1.17 / 2.00 → min 1.00 |
| \|G\|; dimension of the C₀ orbit span; of the plane-mass span | 24 576; 24; 6 |
| plane-mass multiplicity in Sym¹, Sym³, Sym⁵ of the C₀ orbit span | −8.7 × 10⁻¹⁴, −1.8 × 10⁻¹¹, −1.0 × 10⁻⁹ (= 0) |
| ⟨χ_orbit(C_χ), χ_planes⟩; ⟨χ_orbit(C_χ), χ_orbit(C₀)⟩ | 2 (both taste triplets); 0 |
| scalar-mass strings; antisymmetric two-link; one-link | 16; one per plane with phases PH; 4 |
| (anti-)self-dual sums with ε: corner rank, value | 8, 8 for (0,3)±, (0,1)+, (0,2)+ |
| light-subspace cosines of (0,1)+, (0,2)+ vs (0,3)+; of (0,2)− | 1.0, 1.0; 8.8 × 10⁻¹⁶ |
| periodic L = 4, 6, 8: K − hC_χ zero modes; max \|s − h\| of the next 8 (h = 0.1, 0.5) | 8; ≤ 1.0 × 10⁻¹³ |
| production bc, h = 0.1 / 0.5 / 2: light deviation; heavy min (bound), L = 4 | 0.0001 / 0.0019 / 0.030; 1.4168 (1.4165) / 1.4772 (1.4697) / 2.2168 (1.6248) |
| same, L = 6 | 0.0000 / 0.0006 / 0.0089; 1.0043 (1.0032) / 1.1035 (1.0770) / 2.0332 (1.2806) |
| same, L = 8 | 0.0000 / 0.0002 / 0.0038; 0.7714 (0.7695) / 0.9039 (0.8636) / 1.6226 (1.1072) |
| naive η_ν plane sum, periodic 8⁴: s_min at h = 0.1, 0.3 | 0.2729, 0.0173 |
| isotropy; component independence (max spectral difference) | 1.8 × 10⁻¹⁴; 5.8 × 10⁻¹⁵ |
| even-only C_χ: log det(K − hC) − log det K (h = 0.5, 2) | 0, 0 |
| sign of det(D_c − hC ⊗ 1), h = 0.3, 1, y = 2.41: 4⁴ random (2) and stored P_c, C_χ | +1 (imaginary part ≤ 10⁻¹⁴) |
| stored 6⁴ P_c: C_χ, phase-flipped C_χ, C₀ | +1 for all three |
| mass test, periodic 4⁴, h = 0.5: lowest 16, flipped vs C_χ | 16 at 0.25 vs 8 at 0 + 8 at 0.5 |

**Method.** Dense singular values of K − hC (`masspairing.patterns`); corner blocks P₀ᵀCP₀ on the 16 zero modes;
signed-permutation lattice symmetries found by BFS sign fields (`masspairing.symmetry`), orbit spans and character
inner products (`masspairing.algebra.orbits`; Sym³, Sym⁵ characters from power traces); signs by `slogdet` of the dense
doublet operator (`masspairing.operator.DoubletOperator`).

**Caveats.** Item 1 excludes a corner mass induced on the sourced flavour at odd order ≤ 5 by the exact lattice
symmetry; it does not address spontaneous breaking of that symmetry. The production bc (antiperiodic time) breaks the
hypercubic group; the corner statements are made with periodic bc.

**Data.** `data/derived/algebra/eps_L4_y2.41_k-0.01_final.npz`, `data/derived/algebra/eps_L6_y2.41_k-0.01_final.npz`
(final configurations of the ε-model chains at P_c, y = 2.41, κ = −0.01; raw archive
`results/xi_scan/F8_P3_eps/L4_y2.41_k-0.01_g0_g60.npz`, `.../L6_y2.41_k-0.01_g0_g60.npz`).

**Check.** `python claims/K2.5/check.py` (≈ 4–5 min, dominated by the dense 8⁴ spectra).

**Provenance.** Notebook claims C147, C151 item 4, C152 items 1–3, C160 item 1 (unhiggsed_notebook @ 3c4a02c).
