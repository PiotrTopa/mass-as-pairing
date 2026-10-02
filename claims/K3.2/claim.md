# K3.2 — the complete 10-channel correlator at P_c is free-like and suppressed (≤ 0.5 × free)

**Statement.** (1) In the free theory (y = 0, s = 0) the complete 10-channel correlator χ₁₀ (p = 0, the largest corner)
is short-ranged and UV-dominated: ξ₁₀ ≈ 0.54–0.64 lattice spacings, and its finite-size "exponents" are negative.
(2) At P_c = (2.41, −0.01) — at the five link-field points of K3.1 and in the ε model — χ₁₀ is smaller than free
(χ₁₀ + 2σ < 0.6 × χ₁₀,free at every point and L = 4, 6, 8) and not longer-ranged (ξ₁₀/L ≤ free + 2σ at L = 6, 8), and
every (6,8) exponent of K3.1 is at least the free one within 2σ. The (10,3,1) channel is therefore not near-critical
at P_c: its correlation length is that of the free theory and its amplitude is suppressed by the σ background. The
finite-size exponents of K3.1 and K3.3 are free-like; they say "not IR-relevant" and carry no scaling dimension
(Δ_eff = 2(1 − e) would give 2.26 for the free theory on (6,8), whose Δ is 3). (3) A same-parity flavour-blind pairing
source restricted to one staggered sublattice keeps det D_c real and positive, but in the free theory the pair
amplitude on the sourced sublattice vanishes identically while the other sublattice responds at O(h): with
D = [[0, K_eo], [−K_eoᵀ, hC_oo]] and K_eo invertible the odd–odd block of D⁻¹ is zero (Schur complement). The reduced
staggered sublattices are not chiralities; a one-sublattice source is not a single-chirality pairing term.

**Numbers.**

| | L = 4 | L = 6 | L = 8 |
|---|---|---|---|
| free χ₁₀(p = 0) | 1.6766 | 1.0875 | 0.9365 (stored) |
| free ξ₁₀/L | 0.1607 | 0.0991 | 0.0674 |
| ε model at P_c: χ₁₀ (ratio to free) | 0.439(10) (0.26) | 0.471(4) (0.43) | – |
| ε model at P_c: ξ₁₀/L | 0.123 | 0.083 | – |

- Free exponents d ln χ₁₀/d ln V: −0.267 on (4,6), −0.130 on (6,8); free ⟨T_Q⟩₀ on 4⁴ = 1.3075.
- Largest (χ₁₀ + 2σ)/χ₁₀,free over the K3.1 points and L = 4, 6, 8: 0.54 (at (2.41, 0.02, 0), L = 8); the central
  ratios χ₁₀/χ₁₀,free range over 0.08–0.47 (smallest at (3.0, 0.1, +0.1), `results/K3_chi10_scaling.csv`).
- One-sublattice source: max |sign det D_c − 1| = 3.3 × 10⁻¹⁴ (4⁴, y = 2.41 and 0, 8 hot configurations × h = 0.005,
  0.05, 0.3, odd-only and even-only); free theory with the source on odd pairs: ⟨Φ_odd⟩/h = 0 exactly at L = 4, 6
  (h = 0.01, 0.2), ⟨Φ_even⟩/h = 1.757 (L = 4), 1.665 (L = 6), h-independent.

**Method.** Free baseline: dense propagator of the link-field operator at s = 0, y = 0 with the complete estimator
(`masspairing.measure.ChannelMeasure`, exact), ξ₁₀ from S(0) and S(p_min). P_c values: the K3.1 ensembles and the
exact-propagator ε-model chains at L = 4, 6 (`masspairing.analysis.k3`). One-sublattice source: dense slogdet of
D_c − h C_sub ⊗ 1₂ and the dense free inverse.

**Controls.** The comparison uses the same estimator and the same boundary signs for free and interacting values;
the free 8⁴ value is a stored dense computation (`K3_FREE_L8=1` recomputes it).

**Caveats.** L ≤ 8; two volume pairs. The one-sublattice statement is about the free theory and the sign of the
weight; it is not a statement about an interacting seesaw.

**Data.** `data/derived/k3/F8_P1_wedge*/` (K3.1 ensembles), `data/derived/k3/F8_P3_eps/` (ε model at P_c, archive
`results/xi_scan/F8_P3_eps/`). Table `results/K3_chi10_scaling.csv` (ratios to free).

**Check.** `python claims/K3.2/check.py` (≈ 30 s; free 4⁴/6⁴ recomputed densely, 8⁴ stored unless `K3_FREE_L8=1`):
items (1)–(3) above.

**Provenance.** Notebook claim C130 (unhiggsed_notebook @ 3c4a02c).
