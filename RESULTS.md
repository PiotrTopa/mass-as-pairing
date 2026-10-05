# Results

One section per paper claim. Every number is printed by the `check.py` of the claim tagged next to it
(`[K5.8]` = `claims/K5.8/check.py`); `make check` runs them all. Errors are one standard deviation (blocked jackknife
unless the claim states otherwise). Lattice units; P_c = (y, κ) = (2.41, −0.01) is the merged critical point of the
ε model reported by the reference study, transferred to these units [I.3] (see the K7 caveats). "Free" means the same
observable of the free theory at the same L, L_t, boundary conditions and h.

The model: four reduced staggered fermion flavours (the lattice proxy of one Spin(10) generation, 16 Weyl fermions as
(4,2,1) ⊕ (4̄,1,2) of Pati–Salam) with the on-site ε vertex decoupled by a σ triplet (ε model); optionally the
sign-free completion of the 10-channel plaquette term ("wedge", link fields g₁₀, g₆) and same-parity sources. Model N1
adds a flavour-blind taste-chiral Majorana mass h C_χ on one taste doublet ("heavy", R) of every flavour and reads the
other doublet ("light", L).

Contents: [K1](#k1-structure-and-partner-algebra) · [K2](#k2-the-sign-free-class-is-flavour-democratic) ·
[K3](#k3-the-z4-odd-elementary-channel-does-not-condense) · [K4](#k4-composite-takeover) ·
[K5](#k5-the-light-half-under-an-explicit-partner-majorana-mass) · [K6](#k6-corner-block-readout) ·
[K7](#k7-order-of-the-transition) · [K8](#k8-an-explicit-mass-on-half-of-the-generation-at-the-critical-point) ·
[M](#m-closed-moonshot) · [N](#n-neutrino-selection-rules-of-the-z4) · [Instruments](#instruments)

---

## K1. Structure and partner algebra

**Statement.** One Weyl 16 of Spin(10) has Majorana channels Sym²(16) = 10 ⊕ 126 only; the unique holomorphic quartic
is Φ₁₀·Φ₁₀ (Φ₁₂₆·Φ₁₂₆ ≡ 0); ℤ₄ ⊂ U(1)_ψ is the largest anomaly-free remnant for one generation; ℤ₄² = (−1)^F, so every
Majorana bilinear carries ℤ₄ charge 2 and a ℤ₄-symmetric phase has no Majorana (self-conjugate) pole for any fermion,
elementary or composite; a Dirac pole pairing ψ (charge 1) with ψ̄ψ̄ψ ∈ 16̄ (charge 3 ≡ −1) is allowed — the pole of
symmetric mass generation (SMG). Higgs–Dirac, seesaw and SMG are the three corners of one partner matrix
[[ip, Δ], [Δ, ip + M_p]], whose SMG corner with the composite partner ψ̄ψ̄ψ is that of Eichten and Preskill (1986) and
Golterman and Shamir (arXiv:2505.20436); the matrix organises the three mechanisms, and ℤ₄ forces M_p = 0 for a partner
with ψ's quantum numbers.

**Numbers.**
- Bilinear spans ψᵀCΓ^{[k]}ψ, k = 0…5: 0, 10, 0, 120, 0, 126; a generic 126 vev has rank 16 (500/500 samples), the
  SU(5)-singlet direction rank 1 [K1.1].
- S_{(2,2)}(16) = 1 ⊕ 54 ⊕ 210 ⊕ 1050 ⊕ 4125 (one singlet); 126 ⊗ 126 = 54 ⊕ 945 ⊕ 1050 ⊕ 2772 ⊕ 4125 ⊕ 6930; no
  charge-6 invariant, one charge-8 invariant (Φ₁₀·Φ₁₀)²; (Φ₁₂₆Φ₁₂₆)₅₄ = −10 (Φ₁₀Φ₁₀)₅₄ [K1.2].
- Fierz: J₁·J₁ = −(1/256)|Φ₁₀|² − (1/512)|Φ₁₂₆|², J₄₅·J₄₅ = (27/256)|Φ₁₀|² − (5/512)|Φ₁₂₆|² [K1.3].
- Dai–Freed: ξ(ℝP⁵, Spin-ℤ₄) = 1/16; ξ(S⁵/ℤ₄, Spin-ℤ₈) = 5/32, 3/32 (ν = 16 anomalous, phase −1); minimal anomaly-free
  ν = 16 / 32 / 64 for Spin-ℤ₄ / ℤ₈ / ℤ₁₆; T(16)/T(10) = 2, four instanton zero modes [K1.4].
- Pati–Salam: the lattice ε channel is 0.500000 in the 10 and 0.500000 in the 126; the two-site Sym²(4) channel is
  entirely in the 126 (fraction in the 10 ≤ 5.8 × 10⁻³¹) [K1.5].
- The lattice U(1)_ε lies outside spin(10) ⊕ u(1)_ψ (residual 1.0000), Tr Q_ε = Tr Q_ε³ = 0, Tr Q_ε T_L T_L = 2 [K1.6].
- Partner algebra: G_χχ(p → 0) = −M_p/Δ², seesaw pole Δ²/M_p, SMG zero |G| = |p|/(p² + Δ²) [K1.7].

**Links.** `claims/K1.1/` … `claims/K1.7/`; code `masspairing/algebra/`; tables `results/algebra_anomaly.csv`,
`results/algebra_branching.csv`, `results/algebra_fierz.json` (`scripts/table_algebra_*.py`).

**Caveats.** That lens spaces generate Ω₅^{Spin-ℤ_{2m}} is cited, not computed. U(1)_ψ is not a symmetry of real reduced
staggered fields: the lattice ℤ₄ equals U(1)_ψ's ℤ₄ up to a Spin(10) element [K1.6]. The partner matrix is bookkeeping:
it fixes which pairings ℤ₄ allows, not which one a dynamics selects.

## K2. The sign-free class is flavour-democratic

**Statement.** The doublet operator D_c is anti-hermitian and τ₂K-invariant, so det D_c > 0 configuration by
configuration. The pure Sym² (10-channel) term has no sign-free formulation along any route examined; its
sum-of-squares completion is sign-free. Every sign-free (τ₂K-class) operator is quaternion-linear and commutes with a
hidden SU(2) acting transitively on flavour directions: every sign-free Majorana source gives all four flavours the same
|mass|; 1-of-4 and 2+2 splits lie outside the class. The nodal pattern C₀ is a pairing field, not a mass; C_χ is the
taste-chiral Majorana mass.

**Numbers.**
- τ₂K pairing of the spectrum to 1.1 × 10⁻¹³; |tr S(x,x)| ≤ 2.9 × 10⁻¹⁵ [K2.1].
- Pure Sym² term: exact two-site Hamiltonian weights negative in 47 %, 43 %, 17 %, 45 % of the HS schemes' draws;
  10-plet Yukawa Pfaffian negative in 42 % (real) / 55 % (imaginary) fields; 2³ bag weights of the completed vertex all
  positive (552 configurations, min 4.68) while 4-link monomials are negative (−21/256) [K2.2].
- Free 4⁴ per quad: ⟨T_Q⟩₀ = 1.3075; the completion is 0.7845 [K2.3].
- Class distance of a flavour-selective source: 0.8660 (1-of-4), 0.7071 (2+2), 0 (flavour-blind); flavour-resolved
  spectra equal to 3.6 × 10⁻¹⁴ [K2.4].
- C₀ keeps all 16 corner zero modes at every h; no induced corner mass at odd order ≤ 5 (multiplicity 0 under the
  24 576-element lattice group) [K2.5].

**Links.** `claims/K2.1/` … `claims/K2.6/`; code `masspairing/algebra/`, `masspairing/patterns.py`; tables
`results/algebra_mtr.csv`, `results/algebra_democracy.json`.

**Caveats.** "No sign-free formulation" covers the routes listed in K2.2; bag positivity of the completed vertex is
empirical. The τ₂K sign test cannot distinguish a mass from a wrong pattern: the corner-spectrum mass test is needed
[K2.5]. The 2+2 split is positive in the bag and conjugate-pair representations [K2.6], not in the σ-triplet RHMC.

## K3. The ℤ₄-odd elementary channel does not condense

**Statement.** On the sign-free models that carry the 10-channel interaction (the completed term on the whole wedge
|g₆| ≤ g₁₀, its per-link completion, the ε model) the complete SU(4)-invariant 10-channel correlator χ₁₀ — the
(10,3,1) ⊂ 126 channel — does not grow with the volume up to L = 8, is free-like and suppressed, its response to the pure
plaquette term does not grow with L, and a same-parity pairing source finds no spontaneous breaking. Where the model
orders, it is the columnar bond crystal; where U(1)_ε is exact and no crystal forms, the fermion stays gapless.

**Numbers.**
- χ₁₀ volume exponents d ln χ₁₀/d ln V on (6,8) at five wedge points: −0.12 … −0.03, exponent + 2σ ≤ 0.10; positive
  control (dimer) 0.71(1), 0.80(1) [K3.1].
- Free χ₁₀ = 1.6766 / 1.0875 / 0.9365 at L = 4 / 6 / 8; at P_c (χ₁₀ + 2σ)/free ≤ 0.54 [K3.2].
- Response to the pure plaquette term, dense 8⁴: R(8)/R(6) = 1.022(45) (g = 0.01), 0.973(23) (g = 0.02) → irrelevant
  [K3.3].
- Same-parity pairing source: (6,8) exponents of ⟨Φ_src⟩/h −0.055(55), −0.030(48), 0.30–0.42 × free → no SSB [K3.4].
- U(1)_ε-exact family (y = 0 and the −edge): χ₁₀ 0.12–0.49 × free; bond crystal at g_j = 0.2 (dimer exponents 2.33(12),
  2.17(10)); free-like gapless fermion at g_j ≤ 0.1 [K3.5].
- +edge bond crystal at (2.41, 0.1): V⟨D_∥²⟩ = 5.11 / 15.65 / 39.19 at L = 4 / 6 / 8 on the K3.6 chains [K3.6];
  4.97 / 15.72 / 39.28 on the K3.1 chains (the dimer positive control) [K3.1].

**Links.** `claims/K3.1/` … `claims/K3.6/`; data `data/derived/k3/`; tables `results/K3_chi10_scaling.csv`,
`results/K3_linear_response.csv`, `results/K3_source_ssb.csv`, `results/K3_dimer.csv`; figure
[`figures/k3_chi10_response.pdf`](figures/k3_chi10_response.pdf) (`scripts/fig_k3.py`).

**Caveats.** Two volume pairs; these are finite-size statements at L ≤ 8, not scaling dimensions. The two sets of
V⟨D_∥²⟩ at (2.41, 0.1) are independent replicas of the same point (different chains and seeds), both listed. The 8⁴ χ₁₀ of K3.1,
K3.4 and K3.5 come from the stochastic estimator. R is a finite-g response. The source of K3.4 is a nodal pairing field,
not a Majorana mass. The on-site 6 channel has no readout at y = 0.

## K4. Composite takeover

**Statement.** Across SYM → P_c → SMG the response to a ℤ₄-odd field moves from the elementary fermion to its ε-vertex
partner. In the SMG phase the elementary Majorana channel is screened and the composite carries the response, linear in
h and lattice-independent, with the nodal pairing source on one flavour (4⁴/6⁴) and with the taste-chiral mass C_χ up to
8⁴.

**Numbers.**
- Nodal source on flavour 4: |φ_T|/φ_heavy ≤ 0.0055 in SYM (y = 2.0), ≥ 5.8 in SMG (y = 3.0); SMG φ_T/h = −0.0877 (4⁴),
  −0.0896 (6⁴) [K4.1].
- C_χ, stage 0 (4⁴, 4³×8): SMG φ_T_R/h = −0.00174 … −0.00183 [K4.2].
- C_χ, 8⁴: φ_T_R/h = −0.001952(14), −0.001969(16), −0.001955(15) at h = 0.5, 1, 2; ρ = |φ_T_R/φ_R| > 4 on all nine SMG
  rows (8⁴, 6⁴, 6³×12), elementary ≤ 0.011 × free [K4.3].
- P_c with C_χ on all-antiperiodic boxes: ρ = 0.004–0.012 (elementary side) [K4.3]; on 6⁴ pppa vs aaaa the momentum
  readout is bc-independent while ρ(h = 0.5) = 0.0378(34) vs 0.0087(4) [K4.4].

**Links.** `claims/K4.1/` … `claims/K4.4/`; tables `results/K4_takeover.csv`, `results/K4_composite_L8.csv`; figure
[`figures/k4_takeover.pdf`](figures/k4_takeover.pdf) (`scripts/fig_n1inst_takeover.py`).

**Caveats.** The nodal source of K4.1 is a pairing field, not a Majorana mass. The "intermediate" P_c ratio of K4.1 is a
production-boundary-condition feature at 4⁴/6⁴; K4 at P_c is quoted from K4.3/K4.4 with the bc stated. L ≤ 8.

## K5. The light half under an explicit partner Majorana mass

**Statement.** With h C_χ on one taste doublet of every flavour, every Lorentz-scalar mass of the other (light) doublet —
Majorana or Dirac, flavour-symmetric or flavour sextet — is an exact symmetry zero of the action under its full symmetry
group (the lattice stabiliser of C_χ with the one-site shifts, and the flavour SO(4)) on hypercubic-symmetric boxes and
in infinite volume; the point group alone removes the flavour-symmetric masses but not two sextet strings. The
assumption of the general statement that no other invariant light bilinear is even in p holds for the same full group
(an enumeration of the whole light corner algebra), not for the point group alone. More
generally: if no light mass bilinear is invariant under the symmetry the explicit mass leaves, and that symmetry is not
broken spontaneously, every mass-type part of the light two-point function vanishes, for elementary operators and for
composite partners with the light quantum numbers alike; a seesaw light mass Δ M⁻¹ Δᵀ is such a part, so it needs
spontaneous breaking ("no seesaw without re-Higgsing"). An interacting 0+1-dimensional model realises the statement
exactly and shows what it does not fix: a symmetric light gap can still fall like 1/M with a vanishing mass-type part,
so the scaling of a gap does not identify a Majorana mass — the mass-type part of the propagator does. What the lattice
data decide is the light gap and SSB: in the SMG phase (y = 3.0) the pre-registered criteria read **(a)** at h = 1
and 2 — the light half keeps its gap, nearly unchanged in size, with a small drift toward free, and the re-specified
symmetry-breaking clause does not fire — on the stage-1b extensions and on two independent 8⁴ replicas per h. Whether a
symmetry breaks at (3.0, h = 2) is not resolved at L ≤ 8 (neither established nor excluded) in the channel every SSB
clause reads, the anti-self-dual (0,3) light Majorana channel: the growth of the light pair susceptibility relative to
free on (6⁴, 8⁴) compares a 6⁴ box on which the taste split exists on 19.8 % of the momenta with every mode of 8⁴, and on
source-free light modes — 4⁴, whose light doublet carries no source, against the source-free modes of 8⁴ — the
susceptibility relative to free falls with V, faster than with all 8⁴ light modes.

**Numbers.**
- Symmetry zeros, point group: 4⁴ aaaa group 384, 32 elements fix C_χ, 16 of them flip the light Majorana mass; group
  averages of the seven flavour-symmetric light scalar masses ≤ 5.1 × 10⁻¹⁸ (zero to rounding) [K5.2].
- Symmetry zeros, all 16 scalar mass strings (10 flavour-symmetric, 6 sextet), 4⁴ pppp and aaaa: under the 32-element
  point-group stabiliser the on-site ε(x)χΣχ and four-link sextet strings average to 1.000 and 0.354; with the one-site
  shifts (stabiliser 8192) every light part averages to zero; flavour SO(4) has no invariant in Λ²(4) [K5.12].
- 0+1D (8 light + 8 heavy Majoranas): with the residual symmetry exact the light mass-type part is ≤ 5.1 × 10⁻¹⁵ at every
  heavy mass and coupling, with one symmetry-breaking term ≥ 7.8 × 10⁻⁵; without a light quartic the symmetric light gap
  falls like 1/M (slopes −1.09, −1.13) with zero mass-type part [K5.10].
- Free-theory gate: the light momentum readout GL_p0 = 1 on every all-antiperiodic box at every h; the heavy one falls,
  GR_p0(8⁴) = 0.9279 / 0.4457 at h = 0.5 / 2 [K5.1].
- y = 3.0, verdict (a): light pair-channel cosh-mass ratio r_L^cosh(8⁴) = 0.977(2) (h = 1), 0.921(2) (h = 2) on the
  pooled independent replicas (stage 1b: 0.976(2), 0.920(2)); G_L(p_min)/free = 0.0030 / 0.0034 [K5.8, K5.11].
- Recorded with the verdict, h = 2, on (6⁴, 8⁴): the light pair susceptibility grows faster with V than free,
  e − e_free = 0.600(104) on the replicas (5.8σ; 0.456(137) on stage 1b, 0.551(93) pooled with it), its 2σ lower edge
  0.39 below the threshold 0.5 of the re-specified clause; χ_L/free = 1.7 % at 8⁴; at h = 1 not resolved, 0.30(17) [K5.8, K5.11].
- R_peak = [χ_L(0)/χ_L(p_min)]/free at h = 2 rises from 0.61(5) (6⁴) to 0.88(8) (8⁴), 2.8σ, and stays below 1 [K5.8, K5.13].
- Taste-split coverage on all-antiperiodic boxes: complete on 4⁴, 4³×8, 8⁴, 12⁴; 19.8 % of the momenta on 6⁴, 29.6 % on 6³×12 [K5.13].
- Pair (4⁴, 8⁴), same observable, all light modes: e − e_free = −0.32(4) at h = 2 and −0.44(7) at h = 1, against +0.51(9) on (6⁴, 8⁴) with the same reader (8.9σ apart) [K5.13].
- Light block of the source, ‖P_L C_χ P_L‖/‖C_χ‖ (aaaa): 0 on 4⁴ and 6⁴, 0.3684 on 8⁴ (1536 of 4096 light momenta source-free), 0.1951 on 4³×8, 0.1730 on 6³×12 [K5.15].
- Restricted-mode control, (3.0, h = 2), 90 stored 8⁴ configurations: χ_L/free = 0.0156(21) with all light modes, 0.0056(6) on the source-free modes, 0.0024(3) on the 256 momenta nearest the corners; e − e_free (4⁴, 8⁴) = −0.306(58), −0.676(49), −0.976(50); (6⁴, 8⁴) = +0.540(134), −0.352(110), −1.076(113) [K5.15].
- P_c, pair (4⁴, 8⁴): e − e_free = −0.114(14), −0.080(8), −0.024(4), +0.025(4) at h = 0, 0.5, 1, 2; g(8⁴)/g(4⁴) = 0.878, 0.920, 0.926, 0.933 [K5.15].
- (A4) on 4⁴: with the point group and its ℤ₄ extension twelve antisymmetric, even-in-p, non-mass-type light strings survive (pppp); with the one-site shifts only the identity and the taste chirality X₁₃Z₀₁₂₃ (symmetric kernels) and eight odd-in-p strings survive, 10 of 496 on aaaa [K5.14].
- h = 2, χ_L/free on the complete boxes: 0.0365 (4⁴) > 0.0255 (4³×8) > 0.0151 (8⁴); 6⁴: 0.0084, below all three [K5.13].
- Stage-1c integrity: nine chains intact; the replicas agree with each other and with stage 1b (smallest p 0.16); the
  per-mille drift of the 8⁴ h = 1 chain lies in its stored stage-1 stretch (p = 0.005) [K5.9].
- Stage 1 by its letter: no verdict at L ≤ 8 — the gap-floor proxy (t* log ratio) on 6³×12 is 0.066(3) / 0.086(1) < 0.3;
  the SSB clause (b) (e − 2σ > 0.5 on (6⁴, 8⁴)) fired at (3.0, h = 2): e = 0.786(129), e − 2σ = 0.527, e_free = 0.387 [K5.5].
- Stage 0 gates as written: G1 passes, G2 fails at h = 2 on r_L = 1.020(6) (+3.35σ), G3 fails [K5.3].

**Links.** `claims/K5.1/` … `claims/K5.15/`; code `masspairing/analysis/stage1b.py`, `stage1c.py`,
`mass_strings.py`, `complete_pair.py`, `corner_invariants.py`, `restricted_modes.py`, `masspairing/algebra/ed01.py`,
`scripts/run_k5_control.py`; data `data/derived/n1stage/`, `data/derived/free/`, `data/derived/n1inst/K512_mass_strings.json`,
`data/derived/k5control/`; tables `results/K5_verdict_stage1.csv`,
`results/K5_verdict_1b.csv`, `results/K5_verdict_1c.csv`, `results/K5_mass_strings.csv`,
`results/K5_complete_pair.csv`, `results/K5_taste_coverage.csv`; figures
[`figures/K5_h2_replicas.pdf`](figures/K5_h2_replicas.pdf),
[`figures/K8_light_pair_gap_vs_h.pdf`](figures/K8_light_pair_gap_vs_h.pdf) (right panel: y = 3.0),
[`figures/K5_E8_no_plateau.pdf`](figures/K5_E8_no_plateau.pdf) (`scripts/fig_n1stage.py`).

**Caveats.** L ≤ 8. CL_t is a zero-momentum pair channel: its gap is a pair-channel effective mass at the stated t (no
plateau at L_t ≤ 12) [K5.7], not a fermion mass; G_L(p_min) and α_L (K6) are the single-fermion readouts. **Box shape:**
on L³×L_t boxes (6³×12) and on pppa boxes the (0,3) light Majorana component is not a symmetry zero; there ⟨φ_L⟩ is a
free-sized shape effect (free 6³×12: −0.00085 per flavour at h = 2, 5 % of the heavy amplitude), read against free
[K5.2, K5.7]. The symmetry statement is the selection rule of the Weinberg operator in a Higgs language; what is
specific here is its instance in an SMG phase and the measurement. Verdict (a) is read at y = 3.0 only; at P_c the light
half moves toward free with h (K8). **Volume pairs:** on all-antiperiodic boxes the taste split exists on every momentum
for L ≡ 0 mod 4 and on 19.8 % of them on 6⁴ (the infrared-most fifth of the modes), so every (6⁴, 8⁴) light-sector ratio
compares two different mode sets; the h = 2 growth of K5.8/K5.11 is such a ratio and is not reproduced on (4⁴, 8⁴),
where χ_L/free falls with V at every h [K5.13]. A complete taste split is not a source-free light doublet: the source has
a light block on 8⁴ (0.368 of its norm), none on 4⁴ and 6⁴ [K5.15], so (4⁴, 8⁴) with all modes also compares unlike light
doublets; restricted to the source-free 8⁴ modes the fall steepens (control at h = 2 only, 90 configurations) [K5.15].
4⁴ is about 3ξ across in the SMG phase; a 12⁴ point at (3.0, 2) is the clean test. Whether a symmetry breaks at
(3.0, h = 2) is not resolved at L ≤ 8 in the measured (0,3) channel. **Criteria:** the stage-1
SSB clause (b) fired by its letter at (3.0, h = 2) [K5.5]; the stage-1b order re-specified (b) relative to free (e −
e_free − 2σ > 0.5 together with R_peak and χ_L/free clauses), before any new trajectory and after the stage-1 row at the
same (y, h) had been read; stage 1c applies the re-specified clauses unchanged to independent seeds. The stage-1
clauses stand as written: their SSB threshold 0.5 lies just above the free exponent 0.39–0.45 of the pair [K5.5].

## K6. Corner-block readout

**Statement.** In the sign-free class the frequency winding of the corner-block propagator vanishes identically, and
its quantisation would need gaps above the bandwidth at L_t ≤ 8: there is no integer order parameter. The odd-part
frequency exponent α separates SMG from SYM configuration by configuration, and, ensemble-averaged, shows the light half
in SMG as a saturated Luttinger zero at every h.

**Numbers.**
- Every loop step within 10⁻¹² of {0, π}, N_det = 0; quantisation needs m ≥ 7.87 / 5.57 / 4.26 at L_t = 4 / 6 / 8 [K6.1].
- 39 stored configurations: none quantised; α(SMG) > α(SYM) with margins 0.42 (L = 6) and 0.17 (L = 8) [K6.2].
- α_L in SMG: 0.928–0.977 at every h and lattice (free 0.05–0.58), h-stable to 0.021 [K6.3].

**Links.** `claims/K6.1/` … `claims/K6.3/`; tables `results/K6_alpha.csv`, `results/K6_alpha_L.csv`; figure
[`figures/k6_alpha.pdf`](figures/k6_alpha.pdf) (`scripts/fig_n1inst_alpha.py`).

**Caveats.** α cannot tell a pole from a zero of the same gap on all-antiperiodic boxes; it does not identify the AFM
sliver; one configuration per chain in K6.2.

## K7. Order of the transition

**Statement.** At L ≤ 8 there is no first-order signature at P_c: no start dependence on 6⁴ or 8⁴, no two-peak histogram,
an energy-cumulant deficit falling like 1/V, a staggered susceptibility growing far slower than V, and a ξ₂/L crossing
at P_c. The pre-registered rule returns "continuous (consistent with, L ≤ 8)". This is "not first order at L ≤ 8", not
"walking": ξ(y) at L ≤ 12 cannot separate walking from a power law.

**Numbers.**
- Hysteresis: largest of 15 start-pair pulls 2.66σ (6⁴) [K7.1], 2.42σ (8⁴, 2000 trajectories per start) [K7.2].
- Energy-cumulant ratio (2/3 − V_e)₈/₆ = 0.28(2) (1/V law: 0.32) [K7.1].
- Pooled P_c: γ/ν_eff = 1.58(16), η_eff = +0.42(16), 16σ from the first-order value [K7.2].
- ξ₂,stag/L at y = 2.41: 0.377(10) (6⁴) / 0.415(19) (8⁴), Δ = +1.8σ [K7.1]; pooled 0.373(7) / 0.399(9), Δ = +2.4σ [K7.2].
- Walking vs power law: Δχ² ≤ 1.2 at L_max ≤ 12 with 7 points at 1 % [K7.3].

**Links.** `claims/K7.1/` … `claims/K7.3/`; data `data/derived/k7/`; tables `results/K7_fss.csv`,
`results/K7_hysteresis.csv`, `results/K7_verdict.json`, `results/K7_discrimination.csv`; figure
[`figures/k7_xi_over_L.pdf`](figures/k7_xi_over_L.pdf) (`scripts/fig_k7_xi_over_L.py`).

**Caveats.** Two volumes. The crossing sits slightly above y = 2.41 on the pooled data (+2.4σ) [K7.2]. The 8⁴ rule first
returned UNDECIDED at 1000 trajectories per start (chain scoring); the starts were extended to 2000 and the unchanged rule
applied [K7.2].

**κ_c and the AFM boundary.** κ_c = −0.01 is the value of the reference study (arXiv:2608.18239), transferred to these
units; the κ scan of the calibration places the closing of the AFM window in (−0.03, 0.05) [I.3].
Our own κ lines bracket κ_c in (−0.04, +0.02): no AFM order on κ = −0.04 at any y, AFM order on κ = +0.02 at y = 2.452 [K7.1].
Developed AFM order at P_c is excluded at L ≤ 8: |Σ_stag|₆/|Σ_stag|₈ = 1.37 (ordered: 1.01), γ/ν_eff = 1.6–1.8 (AFM: 3.89), R4 = 1.42–1.51 (AFM: 1.09) [K7.1, K7.2].
A thin antiferromagnetic band at P_c, ordered only beyond L = 8, is not excluded; every result at P_c is a result at the
transferred point.

## K8. An explicit mass on half of the generation at the critical point

**Statement.** At P_c an explicit Majorana mass on half of the generation detunes the critical point. The critical ε/σ
channel collapses: the staggered structure factor falls monotonically with h, to the size of its SMG value by h = 2
and below it at h = 3. The light pair-channel effective mass rises monotonically toward its free value and stays below
it; the light single-fermion readout and the light pair susceptibility move toward their free values. Nothing falls
(no seesaw) and nothing breaks spontaneously. The explicit heavy mass moves the critical line to larger y: in the
large-N (Gaussian) approximation it removes the heavy doublet from the fermion loop that drives the staggered-σ
instability, so the critical coupling rises, and the sign survives in the full mean-field potential and in the chain
runner's own operator. The light observables move in that direction — at P_c toward the symmetric point, and inside
SMG toward free — while the large-N size of the shift is too large. At P_c the light half therefore de-criticalises
toward free; the rise of its pair-channel effective mass is the loss of the critical mode, not a symmetric gap beyond
free. The response is a crossover to saturation with one scale h₀ ≈ 2, not a power of h; its small-h onset is
consistent with the h² form that the h → −h symmetry requires. No specific functional form is claimed: generic
saturating curves describe the data as well as the large-N shape.

**Numbers.**
- Lemma: λ = √(Δ² + M²/4) − M/2 ≤ Δ, strictly decreasing in M [K8.1].
- Light pair-channel midpoint cosh mass at P_c on 8⁴: 0.197(9) / 0.211(13) / 0.332(16) / 0.614(5) at h = 0 / 0.5 / 1 / 2
  [K8.1]; 0.764(4) at h = 3, every step 5–24σ on three lattices [K8.2].
- Against free (same box, h and bc): m/m_free = 0.129 → 0.442 on 8⁴ and 0.407 → 0.670 on 6⁴ for h = 0 → 3, below free at
  every h; G_L(p_min)/free 0.730 → 0.867 and χ_L/free 0.517 → 0.920 on 8⁴ [K8.4].
- The pair-channel mass does not tell the phase: on 6⁴ it is 0.821 × free at y = 2.0 and 0.793 × free in SMG, while
  G_L(p_min)/free is 0.914 vs 0.0125 [K8.4].
- ε/σ channel at 8⁴: S(π) = 26.0(2.3) → 2.91(12) from h = 0 to 2, against 3.02(12) in SMG [K8.1]; at h = 3,
  S(π)/S_SMG = 0.73 / 0.69 / 0.71 on 6³×12 / 6⁴ / 8⁴ [K8.2].
- Tier 2 as pre-registered: exponent κ = 0.44(6) on 6³×12 (window [0.7, 1.3]) → FAIL; no single power on 6⁴/8⁴ (χ²/dof
  13 / 27) [K8.2].
- Small-h onset on 6⁴ (h ≤ 1, with replicas): m² − m₀² = (c h^p)² gives p = 1.14(37), consistent with the h² onset
  (p = 1) at 0.4σ; the pre-registered bands returned UNDECIDED and are recorded as mis-specified [K8.3].
- Direction, pre-registered on 6⁴: the positions of P_c between the symmetric point (y = 2.0) and SMG move toward the
  symmetric point, f_g 0.198 → 0.070 (15σ) and f_c 0.363 → 0.074 (29σ); the third clause (S(π)) holds at 1.9σ only,
  so the verdict as written is MIXED; at (y, h) = (3.0, 3) deep SMG, g = 0.0204 [K8.5].
- The same test on 8⁴ with a new y = 2.0 pair: f_g 0.190 → 0.084 (2.8σ), f_c 0.419 → 0.051 (15σ), dS = 0.300(203) (1.5σ):
  MIXED as written; volume trend at h = 3 split — g at P_c loses more from 6⁴ to 8⁴ than at the symmetric point
  (0.9776(27) vs 0.9936(19)), c does not (1.0455(114) vs 1.0204(62)) [K8.9].
- SMG side: G_L(p_min)/free and χ_L/free rise with h on all three lattices (8⁴: 0.0028 → 0.0033 and 0.0022 → 0.0121)
  and on all four independent 8⁴ replicas [K8.5].
- ε vertex, h-matched on 6⁴: the position of P_c falls 0.250 → 0.056 from h = 0 to 3; the heavy mass alone removes 46 %
  of the vertex at y = 2.0 [K8.7].
- Large N: y_MF(h)/y_MF(0) = 1.056 / 1.159 / 1.27 at h = 1 / 2 / 3 on 8⁴, the same from the chain runner's operator;
  the full mean-field transition moves from y* = 2.02 to 2.36 on 4⁴ (h = 0 → 3) [K8.6].
- Fit weight: 1/S(π) = s₀ + b δ_L(h) fits on three lattices (χ²/dof 0.55 / 1.11 / 1.42) where a single power fails
  [K8.6]; generic saturating curves with h₀ = 2.10 / 2.21 / 2.05 fit as well (ΔAIC +0.0 / −1.4 / −2.8) [K8.8].
- Fixed σ: on 8⁴ the fixed-σ fraction of the rise at h = 2 is 0.34(9) (mostly dynamical) [K8.2].

**Links.** `claims/K8.1/` … `claims/K8.9/`; code `masspairing/analysis/stage1b.py`, `stage1c.py`, `direction.py`,
`largen.py`; data `data/derived/n1stage/` (incl. `TH/`), `data/derived/largen/`; tables `results/K8_gap_vs_h.csv`,
`results/K8_onset_6x6.csv`, `results/K8_free_compare.csv`; figures
[`figures/K8_light_vs_free.pdf`](figures/K8_light_vs_free.pdf),
[`figures/K8_onset_6x6.pdf`](figures/K8_onset_6x6.pdf),
[`figures/K8_light_pair_gap_vs_h.pdf`](figures/K8_light_pair_gap_vs_h.pdf) (`scripts/fig_n1stage.py`).

**Caveats.** L ≤ 8 and two volumes: h ∈ [0.5, 3] is not a scaling window and no exponent is claimed. The pair channel has
no plateau at L_t ≤ 12. G_L(p_min) and χ_L are read at p_min ≈ π/L: a symmetric light gap well below π/L would leave them
free-like, so whether the light half at (2.41, h > 0) is gapless or has a small symmetric gap in infinite volume is not
determined; on 8⁴ the single-fermion readout at P_c(h = 3) shows a small excess volume loss against the symmetric point
(difference −0.0159(33), 4.8σ) that the pair susceptibility does not show [K8.9], so a small symmetric light gap below π/L is not
excluded. The direction test has two volumes (6⁴, 8⁴); its S(π) clause is direction-blind (S peaks at the critical point
and falls on both sides) and is short on both. The large-N sign locates the staggered-σ instability; that the line between the symmetric
phase and SMG follows it for h > 0 is an assumption supported by the direction of the data, not derived. The fits of
[K8.6] were formed after the data were known. The lemma's premise (a Dirac mixing with a Majorana-massive partner) is
absent in N1 at tree level: it excludes one mechanism and is not evidence about the rise. The σ-channel test T3 is
undecided and its pass clause lacks a resolution requirement (recorded, no outcome affected) [K8.2]. Two σ-channel
thresholds of K8.1 were widened after a first run (disclosed in the claim).

## M. Closed moonshot

A critical seesaw at P_c (the light gap falling with h only at the critical point) is not seen at L ≤ 8: the light
pair-channel ratio rises, r_L (t* log ratio / free) on 8⁴ = 1.15(17), 2.30(23), 5.84(52) at h = 0.5, 1, 2 [M.1]. No
larger-volume run was made. `claims/M.1/`.

## N. Neutrino selection rules of the ℤ₄

On matter the ℤ₄ is X = 5(B−L) − 4Y mod 4: every field of the 16 has X ≡ 1, the Higgs X ≡ 2, and after electroweak
breaking 5(B−L) − 4Q mod 4 stays exact. While it is exact there is no Majorana neutrino mass and no |ΔL| = 2 amplitude
(neutrinoless double-beta decay included); a type-I seesaw needs a condensate with B−L = 2; ν^c alone cannot be gapped
symmetrically (anomaly); and a ℤ₄-preserving heavy ν^c mass — a Dirac pairing with a partner of charge 3 — leaves one
exactly massless neutral fermion per generation, not a seesaw-suppressed one [N.1]. The formula is
García-Etxebarria–Montero's and the absence of a Majorana mass with exact ℤ₄ is known; the counting is the addition.
`claims/N.1/`.

## Instruments

The machinery the K claims rest on, each certified against an independent exact computation with a sabotage control:
RHMC exact against exact-determinant Metropolis [I.1]; Hasenbusch/multiple-time-scale integrator [I.2]; calibration
y = √2 y_ref, P_c = (2.41, −0.01) transferred from the reference study [I.3]; link-field HMC of the completed term against the all-orders bag expansion [I.4];
the link-field resonance at ωτ ≈ nπ and the trajectory-length jitter that removes it [I.5]; the complete 10/6-channel
estimator [I.6]; the same-parity source [I.7]; the flavour-selective source with its |Pf| RHMC [I.8] and Pfaffian sign
[I.9]; the N1 RHMC against exact-determinant Metropolis with the phase-flipped pattern rejected [I.10]; the N1 taste
observables [I.11]. See each `claims/I.*/claim.md`.
