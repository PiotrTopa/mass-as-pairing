# K3.6 — the columnar bond crystal (dimer channel) condenses on the +edge

**Statement.** On the +edge g₆ = +g₁₀ of the sign-free link-field model at κ = −0.01, λ = 1, at the reference point
(y, g₁₀) = (2.41, 0.1) the columnar dimer susceptibility V⟨D_∥²⟩ grows with effective exponent d ln(V⟨D_∥²⟩)/d ln V =
0.69(1) on (4,6) and 0.80(1) on (6,8), rising towards 1 (∝ V); the dimer correlation length of the link field grows
faster than L and D_⊥ stays flat, so the order is columnar — a lattice-symmetry-breaking bond crystal. The ε channel
V⟨|φ_stag|²⟩ grows sub-linearly at all three points studied, and at (2.41, 0.05) the dimer channel overtakes the ε
channel with the volume. On a 6⁴ lattice at the reference point three different starts agree (no hysteresis).

**Numbers.**

| (y, g₁₀) | L | V⟨D_∥²⟩ | V⟨D_⊥²⟩ | V⟨\|φ_stag\|²⟩ | ξ₂ link dimer |
|---|---|---|---|---|---|
| (2.41, 0.05) | 4 / 6 / 8 | 1.52 / 2.67 / 5.08 | 0.05 / 0.04 / 0.02 | 1.15 / 2.25 / 3.09 | 0.35 / 0.67 / 1.22 |
| (3.0, 0.1) | 4 / 6 / 8 | 0.69 / 2.16 / 2.67 | 0.03 / 0.03 / 0.04 | 0.49 / 0.87 / 1.21 | 0.29 / 0.81 / 1.10 |
| (2.41, 0.1) | 4 / 6 / 8 | 5.11 / 15.65 / 39.19 | 0.02 / 0.02 / 0.02 | 0.17 / 0.20 / 0.20 | 0.99 / 2.46 / 5.03 |

- Reference point: ξ₂/L of the link-field dimer = 0.249 → 0.411 → 0.629 (L = 4, 6, 8).
- Exponents (4,6) / (6,8): dimer 0.35(8) / 0.56(11), 0.71(13) / 0.18(15), 0.69(1) / 0.80(1); ε 0.42(4) / 0.28(6),
  0.35(5) / 0.29(6), 0.11(5) / −0.03(5) (points in table order).
- (2.41, 0.05): dimer exponent 0.56(11) vs ε 0.28(6) on (6,8), 2.3σ apart. At (3.0, 0.1) neither channel grows ∝ V
  (the dimer exponent drops from 0.71 to 0.18).

**Method.** L = 4: exact propagators, 400 trajectories; L = 6: 1000, L = 8: 1500 trajectories with 8 noise vectors every
2nd trajectory (Hasenbusch + nested integrator); 8⁴ from the staggered σ start with zero link fields, 6⁴ from hot σ.
Blocked-jackknife analysis (`masspairing.analysis.k3.ensemble`); V-scaled susceptibilities with the errors of their
own series; link-field dimer length from the structure factors at π ê_ρ and π ê_ρ + p_min.

**Controls.** No hysteresis: on 6⁴ at the reference point the chains from hot σ + zero links, staggered σ + columnar
dimer links, and hot σ + hot links agree within 1.3σ on V⟨D_∥²⟩, V⟨D_⊥²⟩, V⟨|φ_stag|²⟩, O₄, ξ₂,stag and the link-field
dimer length.

**Caveats.** Three volumes, one κ, the +edge only (the −edge is K3.5). L = 4 uses exact propagators and L = 6, 8 the
noise estimator. The (6,8) dimer exponent at the reference point (0.80) is below 1: a first-order jump is not
distinguished from a continuous transition with a large anomalous dimension. The 8⁴ chain at (2.41, 0.05) has
acceptance 0.67. The (2.41, 0.05) overtaking is a 2.3σ statement. These chains predate the complete 10-channel
estimator; their 10-channel numbers are not used (the 10-channel statements on the +edge are K3.1).

**Data.** `data/derived/k3/c072/`, `c072_n20/` (L = 4), `c073_main/`, `c073_afmdimer/`, `c073_hothot/` (L = 6, 8), from
the archive chains `results/laneW2/c072/`, `results/laneW2/c073/`.

**Check.** `python claims/K3.6/check.py` (≈ 3 s).

**Provenance.** Notebook claim C073, dimer and ε content (unhiggsed_notebook @ 3c4a02c).
