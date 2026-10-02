# K5.1 — N1 instrument definitions: taste projectors, the mass test, free baselines, the light-doublet gate

**Statement.** Model N1 puts the flavour-blind taste-chiral Majorana mass h C_χ (K2.5) on one taste doublet ("R",
heavy) of every flavour; the other doublet ("L", light) is the object of K5. The instrument is defined and certified
here:

1. **Taste projectors** (`patterns.TasteProjector`). In the 16-block of a reduced momentum the string Q = ε J_D J_dual =
   X₁₃Z₀₁₂₃ commutes with K(p) and with every plane operator; in position space Q = ζ₁₃(x) F⁻¹[Π_μ sign cos p_μ] F with
   twisted momenta. At momenta with cos p_μ = 0 (periodic axes with L ≡ 0 mod 4, antiperiodic axes with L ≡ 2 mod 4)
   complex conjugation swaps the two eigenspaces, so a real field has no light/heavy split there; with sign(0) := 0, Q is
   real symmetric, Q² = Π_nb (the projector on the momenta where the split exists), and P_R = (Q² + Q)/2,
   P_L = (Q² − Q)/2 are orthogonal projectors with P_L + P_R = Π_nb. On all-antiperiodic boxes with every L ≡ 0 mod 4
   (4⁴, 8⁴, 4³×8, …) the split is complete. Q is the same for the three self-dual components (0,1), (0,2), (0,3) and
   changes sign with the dual sign.
2. **Mass test of the stencilled pattern.** The merged sparse operator equals K − hC; K − hC_χ has exactly 8 zero
   singular values, all in P_L, and 8 at h, all in P_R: C_χ is a Majorana mass of gap h for exactly one doublet.
3. **Taste chirality is a lattice pseudoscalar.** Under every signed-permutation symmetry of K and the one-site
   translations, T Q Tᵀ = ±Q exactly: +Q for the elements that map C_χ into the self-dual triplet (same heavy doublet;
   among them every symmetry of the sourced action), −Q for those that map it into the anti-self-dual triplet and for
   every one-site translation. Every symmetry of the sourced action therefore preserves P_L.
4. **Free baselines** of every key of the N1 measure at L = 4/6/8 under both boundary conditions (frozen in
   `data/derived/free/`), and the **gate**: in the free theory the light doublet is massless beyond the finite-size
   term — GL_p0(h) = 1 exactly on every all-antiperiodic box while GR_p0 falls with h, the light time-slice mass shifts
   by ≤ 0.02 for h ≤ 0.5 at L ≥ 6, the heavy one rises at every h, and the light SSB susceptibility chi_L_sum does not
   change with h (< 1 % for h ≤ 1).

**Numbers.**

| item | result |
|---|---|
| projector algebra (symmetric, Q² = Π_nb, [Q, K] = [Q, planes] = [Q, C_L] = 0, P_L ⊥ P_R, FFT = dense) on 4⁴ pppp/aaaa/pppa, 6⁴ aaaa/pppa, 4³×8 aaaa | ≤ 1.3e-15 |
| boundary (zone-boundary) modes | 4⁴: pppp 240/256, aaaa 0/256, pppa 224/256; 6⁴: aaaa 1040/1296, pppa 432/1296; 4³×8 aaaa 0/512 |
| merged CSR = K − hC (periodic 4⁴, 6⁴): chiral / full | exact; 16 / 32 neighbours |
| K − hC_χ, h ∈ {0.1, 0.5, 2}, periodic 4⁴, 6⁴: 8 zero modes in P_L, 8 at h in P_R | deviation ≤ 7.9e-14 |
| symmetry action on Q, 4⁴ aaaa and pppp: 384 elements + 4 shifts | +Q on 192 (32 fix C_χ), −Q on 192 and on the 4 shifts |
| the same, production bc 4⁴: 96 elements + 4 shifts | +Q on 48 (8 fix C_χ), −Q on 48 and on the 4 shifts |
| GL_p0 on every aaaa box (4⁴, 6⁴, 8⁴), every h | 1 to 1e-15 |
| GR_p0, 8⁴ aaaa, h = 0 / 0.5 / 2 | 1 / 0.9279 / 0.4457 |
| GR_p0, 6⁴ aaaa, h = 0.2 / 0.5 / 1 / 2 | 0.9944 / 0.9660 / 0.8767 / 0.6400 |
| light time-slice mass shift at t* = L_t/2 − 1, h = 0.5: 8⁴ aaaa / 8⁴ pppa / 6⁴ aaaa / 6⁴ pppa / 4³×8 aaaa | +0.0088 / −0.0006 / +0.0000 / +0.0009 / +0.0059 |
| heavy mass rise at t = 1, h = 0.5: 8⁴ aaaa / 8⁴ pppa / 6⁴ aaaa | +0.027 / +0.132 / +0.018 |
| chi_L_sum(h)/chi_L_sum(0) − 1, aaaa, h ≤ 1 | ≤ 0.0044 |
| 8⁴ aaaa: chi_L_sum at h = 0 / 0.5; chi_f_sum at h = 0 / 0.5 / 2 | 0.0464 / 0.0462; 0.052 / 2.59 / 21.3 |
| live regeneration of 14 frozen rows (4⁴ aaaa and pppa, 4³×8, 6⁴ aaaa and pppa at h = 0.5), every key | ≤ 1e-12 |

Recorded, not asserted: at h = 2 the light time-slice shift is +0.113 at 8⁴ aaaa and +0.0135 at 6⁴ pppa, and exactly
0 on aaaa 4⁴/6⁴ (every momentum has equal |p_μ| there): the O(h p²) term ½h(c_D − c_dual) of the light sector at
generic momenta, not a mass (the corner zero modes and GL_p0 = 1 are exact). Every interacting light number is
therefore read as a ratio to its free value at the same L, L_t, bc and h.

**Method.** Free theory: G = (K − hC_χ)⁻¹ ⊗ 1₄ evaluated with the exact N1 measure (`analysis.free.free_point`);
effective masses are log ratios of the positive time-slice correlators CL_t, CR_t. The momentum readout is
G_{L,R}_p0 = 4Σsin²p₀‖G_{L,R}(p₀)‖²_F/32 on the 16-corner × 4-flavour block at the smallest twisted momentum p₀ (1
for a free massless doublet; the free-doublet value is 4Σsin²p₀/(4Σsin²p₀ + m(p₀)²) with m_R = ½h(c_D + c_dual),
m_L = ½h(c_D − c_dual)). Symmetries: site maps g of the hypercubic group and the one-site shifts with the sign field s
fixed by K (BFS on the links of K, `symmetry.sign_field`).

**Controls.** The phase-flipped pattern (plane (0,3) without its ζ phase plus its dual) fails the mass test: no zero
mode, all 16 corner modes at h/2. The nodal pattern C₀ fails it: 16 zero modes stay. The "full" six-plane sum lifts all
16 corner modes at two gaps (0.5, 0.7071 at h = 0.5): not a single-gap mass of one doublet.

**Caveats.** On boxes with zone-boundary modes (pppa, L ≡ 2 mod 4 antiperiodic) the taste observables are exact on the
modes where the split exists; the excluded modes are at the cutoff. On 4⁴ pppa the smallest momentum has spatial
components π/2: the momentum readout is void there (GL_p0 = GR_p0 = 0). The 6³×12 and 8⁴ baselines are frozen
outputs (archive); the check regenerates the 4⁴, 4³×8 and two 6⁴ rows with the same code.

**Data.** `data/derived/free/free_baselines.json` (all boxes of this claim) and the further baseline files
`free_L4_aaaa.json`, `free_L4_pppa.json`, `free_L4x8_aaaa.json`, `free_L6x12_aaaa.json`, `free_L6x12_aaaa_h1.5_3.json`,
`free_L6_aaaa_h1.5_3.json`, `free_L8_aaaa.json`, `free_L8_aaaa_h1.5.json`, `free_L8_aaaa_h3.json`,
`free_flipped_L6_aaaa_h0.0.json`, `free_flipped_L6_aaaa_h0.5.json` used by K5.2–K5.8 and K8 (copied verbatim from the
archive `results/laneK1/`, `results/laneK5S1A/s3_flipped/` by `scripts/derive_n1inst.py free`; generator
`masspairing/analysis/free.py`).

**Check.** `python claims/K5.1/check.py` (≈ 4 min): prints and asserts items 1–4 and the controls.

**Provenance.** Notebook claims C160 items 2–4, C163 items 1–2 (unhiggsed_notebook @ 3c4a02c).
