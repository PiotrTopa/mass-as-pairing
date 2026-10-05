# K5.14 — Assumption (A4) by enumeration of the light corner algebra on 4⁴: with the point group alone twelve antisymmetric, even-in-p, non-mass-type light invariants survive; with the one-site shifts only symmetric-kernel and odd-in-p strings remain

**Statement.** Every corner string X_S Z_D (256 strings, both site symmetrisations: 496 non-zero lattice bilinears) is
averaged over subgroups of the symmetry group of the h C_χ-sourced action on 4⁴. Under the point-group stabiliser of C_χ
and its ℤ₄ extension, light invariants exist that are neither mass-type nor odd in p: on 4⁴ pppp twelve antisymmetric
(flavour-symmetric), even-in-p strings survive the 64-element point group + ℤ₄ (twenty survive the 32-element stabiliser
alone), so (A4) fails with the point group alone. Under the full stabiliser with the one-site shifts every one of them
vanishes. The only survivors of all 496 bilinears are the identity and the taste chirality X₁₃Z₀₁₂₃ — both with symmetric
site kernels, hence zero in the flavour-symmetric channel by Grassmann antisymmetry and sextet-only otherwise, where the
flavour SO(4) has no invariant — and, on 4⁴ aaaa, eight antisymmetric strings that are all odd in p (the four Γ_μ and four
three-link strings). No mass-type string survives. So every light element invariant under the lattice stabiliser with
shifts × SO(4) is mass-type, odd in p, or excluded by Grassmann antisymmetry: (A4) holds for the full group. The
four-link string X₀₂Z₀₁₂₃ = ±Γ₀Γ₁Γ₂Γ₃ anticommutes with every Γ_μ: it is the four-link sextet mass string of K5.12, not
an invariant. The sign fields of all 384 hypercubic site maps are functions of x mod 2 on periodic boxes and x mod 2 times
(−1)^{#{μ reflected: x_μ = 0}} on all-antiperiodic boxes, so the group acts on the (S, D) labels identically on every
even-L box with equal boundary conditions and the 4⁴ result carries over.

**Numbers.** (K5.14 check output)

| box, group | elements | light survivors | antisymmetric, even in p, not mass-type | survivors |
|---|---|---|---|---|
| 4⁴ pppp, point-group stabiliser | 32 | 52 | 20 | incl. the on-site and four-link sextet masses |
| 4⁴ pppp, point group + ℤ₄ | 64 | 20 | 12 (4 LL+RR Γ_μΓ_ν-type, 8 LR) | no mass-type |
| 4⁴ pppp, stabiliser with shifts | 8192 | 2 | 0 | identity, X₁₃Z₀₁₂₃ (symmetric) |
| 4⁴ aaaa, point-group stabiliser | 32 | 236 | 36 | |
| 4⁴ aaaa, point group + ℤ₄ | 64 | 126 | 20 | |
| 4⁴ aaaa, stabiliser with shifts | 8192 | 10 | 0 | identity, X₁₃Z₀₁₂₃; Γ₀…Γ₃ (orbit 4), four three-link (orbit 16), all odd |

The four taste-diagonal pppp invariants of point group + ℤ₄ are (S, D) = ((1),(1,2)), ((0,1,2),(0,3)), ((3),(0,3)),
((0,2,3),(1,2)); the three-link survivors on aaaa are ((),(0,1,3)), ((1),(0,2,3)), ((0,1,3),(0,1,2)), ((2,3),(1,2,3)).
On pppp the light sector is the 16 corner momenta (p ∈ {0, π}⁴), where odd-in-p strings have no light part; on aaaa there
is no p = 0 momentum and they do. Sign fields: 384/384 x-mod-2 functions on 4⁴ and 6⁴ pppp; on 4⁴ and 6⁴ aaaa 24 maps
(no reflection) are, the other 360 carry the boundary-slice factor (8⁴ aaaa the same, notebook run).

**Method.** `masspairing.analysis.corner_invariants`. Strings from `mass_strings.string_op`; groups from
`symmetry.{generators, closure, transform}`, the stabiliser tested on the non-zeros of C_χ; ℤ₄ extension = the C_χ-flipping
point-group elements composed with χ → i^{ε(x)}χ (sign −1 on bilinears with an even number of links); averages under the
8192-element stabiliser computed exactly as orbit means under a four-element generating set. A string survives when
‖P_L Ō P_L‖ or ‖P_L Ō P_R‖ is non-zero; every group element used commutes with P_L. Even/odd in p: the momentum factor of the
string at the corner (cos or sin of p·D). Mass-type: anticommutes with K (= the 16 strings of
`patterns.scalar_mass_strings`).

**Controls.** The pppp point-group survivor list and its ℤ₄ survival reproduce an independent enumeration in the research
notebook; the ten aaaa survivors under the full stabiliser reproduce an independent referee enumeration (10 of 496). The
two sextet masses of K5.12 appear among the point-group survivors and vanish with the shifts.

**Caveats.** A finite computation on 4⁴; the carry-over to other even-L boxes and to infinite volume rests on the sign-field
lemma (verified on 4⁴, 6⁴, 8⁴), not on a separate enumeration per box. (A4) is a statement at p = 0 (pppp) and about the
p-parity of the light elements; it needs the shift subgroup, not only the point group named by Theorem 1.

**Data.** None (exact computation).

**Check.** `python claims/K5.14/check.py` (≈ 4 min, 0.8 GB).

**Provenance.** Notebook claim C218 (unhiggsed_notebook @ 0624ddc).
