# Results

One section per paper claim. Every number is printed by the `check.py` of the claim tagged next to it
(`[K5.8]` = `claims/K5.8/check.py`); `make check` runs them all. Errors are one standard deviation (blocked jackknife
unless the claim states otherwise). Lattice units; P_c = (y, κ) = (2.41, −0.01) is the merged critical point of the
ε model in these units [I.3]. "Free" means the same observable of the free theory at the same L, L_t, boundary
conditions and h.

The model: four reduced staggered fermion flavours (the lattice proxy of one Spin(10) generation, 16 Weyl fermions as
(4,2,1) ⊕ (4̄,1,2) of Pati–Salam) with the on-site ε vertex decoupled by a σ triplet (ε model); optionally the
sign-free completion of the 10-channel plaquette term ("wedge", link fields g₁₀, g₆) and same-parity sources. Model N1
adds a flavour-blind taste-chiral Majorana mass h C_χ on one taste doublet ("heavy", R) of every flavour and reads the
other doublet ("light", L).

Contents: [K1](#k1-structure-and-partner-algebra) · [K2](#k2-the-sign-free-class-is-flavour-democratic) ·
[K3](#k3-the-z4-odd-elementary-channel-does-not-condense) · [K4](#k4-composite-takeover) ·
[K5](#k5-the-light-half-under-an-explicit-partner-majorana-mass) · [K6](#k6-corner-block-readout) ·
[K7](#k7-order-of-the-transition) · [K8](#k8-anti-seesaw-at-the-critical-point) · [M](#m-closed-moonshot) ·
[Instruments](#instruments)

---

## K1. Structure and partner algebra

**Statement.** One Weyl 16 of Spin(10) has Majorana channels Sym²(16) = 10 ⊕ 126 only; the unique holomorphic quartic
is Φ₁₀·Φ₁₀ (Φ₁₂₆·Φ₁₂₆ ≡ 0); ℤ₄ ⊂ U(1)_ψ is the largest anomaly-free remnant for one generation; ℤ₄² = (−1)^F, so a
ℤ₄-symmetric phase has no pole mass for any fermion, elementary or composite. Higgs–Dirac, seesaw and symmetric mass
generation (SMG) are the three corners of one partner matrix [[ip, Δ], [Δ, ip + M_p]]; the SMG partner of ψ is
ψ̄ψ̄ψ ∈ 16̄.

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
staggered fields: the lattice ℤ₄ equals U(1)_ψ's ℤ₄ up to a Spin(10) element [K1.6].

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
- +edge bond crystal at (2.41, 0.1): V⟨D_∥²⟩ = 5.11 / 15.65 / 39.19 at L = 4 / 6 / 8 [K3.6].

**Links.** `claims/K3.1/` … `claims/K3.6/`; data `data/derived/k3/`; tables `results/K3_chi10_scaling.csv`,
`results/K3_linear_response.csv`, `results/K3_source_ssb.csv`, `results/K3_dimer.csv`; figure
[`figures/k3_chi10_response.pdf`](figures/k3_chi10_response.pdf) (`scripts/fig_k3.py`).

**Caveats.** Two volume pairs; these are finite-size statements at L ≤ 8, not scaling dimensions. The 8⁴ χ₁₀ of K3.1,
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
Majorana or Dirac — is an exact symmetry zero of the action on hypercubic-symmetric boxes and in infinite volume: a light
mass needs spontaneous breaking ("no seesaw without re-Higgsing"). What the data decide is the symmetric gap and SSB: in
the SMG phase (y = 3.0) the light half keeps its own gap — verdict **(a)** of the pre-registered stage-1b criteria at
h = 1 and 2 — and nowhere does it go gapless or re-Higgs.

**Numbers.**
- Symmetry zeros: 4⁴ aaaa group 384, 32 elements fix C_χ, 16 of them flip the light Majorana mass; group averages of
  every light scalar mass ≤ 5.1 × 10⁻¹⁸ (zero to rounding) [K5.2].
- Free-theory gate: the light momentum readout GL_p0 = 1 on every all-antiperiodic box at every h; the heavy one falls,
  GR_p0(8⁴) = 0.9279 / 0.4457 at h = 0.5 / 2 [K5.1].
- Stage 1b at y = 3.0 (new samples): light pair-channel cosh-mass ratio r_L^cosh(8⁴) = 0.976(2) (h = 1), 0.920(2) (h = 2);
  G_L(p_min)/free = 0.0030 / 0.0034; e − e_free = 0.42(26) / 0.456(137), no p = 0 peak sharper than free → (a) [K5.8].
- Recorded with the verdict: at h = 2 the light pair susceptibility grows relative to free at 3.3σ with no SSB-scale
  exponent, on a susceptibility χ_L/free = 0.0142(20) [K5.8].
- Stage 1 by its letter: no verdict at L ≤ 8 — the gap-floor proxy (t* log ratio) on 6³×12 is 0.066(3) / 0.086(1) < 0.3;
  the SSB clause flagged once at the 2σ edge, e(2) = 0.786(129) at y = 3.0 [K5.5].
- Stage 0 gates as written: G1 passes, G2 fails at h = 2 on r_L = 1.020(6) (+3.35σ), G3 fails [K5.3].

**Links.** `claims/K5.1/` … `claims/K5.8/`; data `data/derived/n1stage/`, `data/derived/free/`; tables
`results/K5_verdict_stage1.csv`, `results/K5_verdict_1b.csv`; figures
[`figures/K8_light_pair_gap_vs_h.pdf`](figures/K8_light_pair_gap_vs_h.pdf) (right panel: y = 3.0),
[`figures/K5_E8_no_plateau.pdf`](figures/K5_E8_no_plateau.pdf) (`scripts/fig_n1stage.py`).

**Caveats.** L ≤ 8. CL_t is a zero-momentum pair channel: its gap is a pair-channel effective mass at the stated t (no
plateau at L_t ≤ 12) [K5.7], not a fermion mass. **Box shape:** on L³×L_t boxes (6³×12) and on pppa boxes the (0,3) light
Majorana component is not a symmetry zero; there ⟨φ_L⟩ is a free-sized shape effect (free 6³×12: −0.00085 per flavour at
h = 2, 5 % of the heavy amplitude), read against free [K5.2, K5.7]. **Partial outcome at h = 2**, recorded above. The
criteria of each stage were fixed before its data were read; the stage-1 criteria proved mis-specified for these data and
stand as written [K5.5].

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

## K8. Anti-seesaw at the critical point

**Statement.** At P_c an explicit Majorana mass on half of the generation gaps the other half symmetrically: the light
pair-channel effective mass rises monotonically with h on three lattices while the critical ε/σ channel collapses onto
(and below) its SMG value, with no spontaneous breaking. A partner Majorana mass can only lower a light gap (lemma), and a
light mass term is a symmetry zero (K5), so this is the opposite of a seesaw. The rise is a **crossover** from a
quadratic-like onset to saturation, **not a power law**: the pre-registered exponent test fails.

**Numbers.**
- Lemma: λ = √(Δ² + M²/4) − M/2 ≤ Δ, strictly decreasing in M [K8.1].
- Light pair-channel midpoint cosh mass at P_c on 8⁴: 0.197(9) / 0.211(13) / 0.332(16) / 0.614(5) at h = 0 / 0.5 / 1 / 2
  [K8.1]; 0.764(4) at h = 3, every step 5–24σ on three lattices [K8.2].
- ε/σ channel at 8⁴: S(π) = 26.0(1.9) → 2.91(15) from h = 0 to 2, against 3.02(10) in SMG [K8.1].
- Tier 2 as pre-registered: exponent κ = 0.44(6) on 6³×12 (window [0.7, 1.3]) → FAIL; no single power on 6⁴/8⁴ (χ²/dof
  13 / 27) [K8.2].
- Fixed σ: on 8⁴ the fixed-σ fraction of the rise at h = 2 is 0.34(9) (mostly dynamical) [K8.2].

**Links.** `claims/K8.1/`, `claims/K8.2/`; table `results/K8_gap_vs_h.csv`; figure
[`figures/K8_light_pair_gap_vs_h.pdf`](figures/K8_light_pair_gap_vs_h.pdf) (`scripts/fig_n1stage.py`).

**Caveats.** Crossover, not exponent: h ∈ [0.5, 3] is not a scaling window. L ≤ 8; no plateau of the pair channel at
L_t ≤ 12. The σ-channel test T3 is undecided and its pass clause lacks a resolution requirement (recorded, no outcome
affected) [K8.2]. Two σ-channel thresholds of K8.1 were widened after a first run (disclosed in the claim).

## M. Closed moonshot

A critical seesaw at P_c (the light gap falling with h only at the critical point) is not seen at L ≤ 8: the light
pair-channel ratio rises, r_L (t* log ratio / free) on 8⁴ = 1.15(17), 2.30(23), 5.84(52) at h = 0.5, 1, 2 [M.1]. No
larger-volume run was made. `claims/M.1/`.

## Instruments

The machinery the K claims rest on, each certified against an independent exact computation with a sabotage control:
RHMC exact against exact-determinant Metropolis [I.1]; Hasenbusch/multiple-time-scale integrator [I.2]; calibration
y = √2 y_ref, P_c = (2.41, −0.01) [I.3]; link-field HMC of the completed term against the all-orders bag expansion [I.4];
the link-field resonance at ωτ ≈ nπ and the trajectory-length jitter that removes it [I.5]; the complete 10/6-channel
estimator [I.6]; the same-parity source [I.7]; the flavour-selective source with its |Pf| RHMC [I.8] and Pfaffian sign
[I.9]; the N1 RHMC against exact-determinant Metropolis with the phase-flipped pattern rejected [I.10]; the N1 taste
observables [I.11]. See each `claims/I.*/claim.md`.
