# I.8 — Flavour-selective source in the real basis: |Pf| RHMC and real-basis observables exact

**Statement.** The flavour-selective source is realised as M(h) = Re[W D_c(σ, s) W†] − h C₀ ⊗ P, a real antisymmetric
operator on the four real flavours χ^a (P = diag(p_a) a flavour mask, P = diag(0, 0, 0, 1) for the source on flavour
4), applied real-linearly on the packed doublet ψ_p = χ^{2p} + iχ^{2p+1} (`operator.DoubletOperator` with `h_sel`,
`mask`; apply = M, apply_dag = Mᵀ = −M). The RHMC with one complex pseudofermion, S_pf = Re φ†(MᵀM)^{−1/2}φ and heat
bath φ = (MᵀM)^{1/4}η, samples the weight det(MᵀM)^{1/4} = |Pf M(h)|; the sign of Pf M is not in the weight and is
reweighted (I.9). This is exact: the operator, the action and the forces are correct, the RHMC reproduces an
exact-determinant Metropolis, and the real-basis fermion observables (`measure.N1Measure`) — ε channel, links, the
per-flavour pair channel O^(a) = ½ Σ C₀ χ^a χ^a (amplitude phi_f, sublattice parts, corner momenta, susceptibility chi_f
with all Wick terms, connected response chi_f_conn, p_min value), the flavour-resolved point-source correlators Gf_f,
Cf_f and the six-fermion channel (phi_T, C_T) — are exact.

**Numbers.**
1. Operator, 4⁴ (4 hot configurations): M(h) antisymmetric and equal to Re[W D_c W†] − hC₀ ⊗ P to 0; packed apply = Mχ
   (single vectors and batches) and apply_dag = −apply to 5.3e-15; the all-flavour mask equals the flavour-blind operator
   D_c − hC₀ ⊗ 1₂ to 5.4e-15; Pf M(0) = det D_c > 0 by Householder and Parlett–Reid to 1.8e-15 (relative, in the log).
2. 4⁴ wedge model (g₁₀, g₆) = (0.25, 0.05), h = 0.1 on flavour 4: Lanczos S_pf vs dense φᵀ(MᵀM)^{−1/2}φ 8.3e-14; σ- and
   s-forces vs 4-point finite differences 7.9e-8 (six components); reversibility 1.4e-13 / 1.1e-12.
3. Exact Metropolis vs RHMC on the 2³×4 all-antiperiodic box, ε model (y, κ, λ) = (2, 0, 1), h = 0.3 on flavour 4: 17
   sign-reweighted observables (σ², |Σ_stag|, O₄, |φ_stag|², E_bond, dimer, light and heavy pair amplitudes with the
   sublattice parts of the heavy one, light and heavy susceptibilities, heavy connected response, φ_T, the point-source
   correlators Cf_f(1) heavy and light, C_T(1)) agree within 1.39σ (largest pull, σ²); e.g. phi_f_heavy 0.05035(29) vs
   0.05028(30), chi_f_heavy 0.3261(17) vs 0.3248(19), phi_f_light −0.00006(24) vs +0.00007(30). No negative Pfaffian in
   either chain at this h (1866 + 1200 measured configurations).
4. Observables:
   - h = 0, 4⁴ wedge configuration: phi, phi_stag, O4, E_bond, dimer_sq and Σ_a phi_f (even/odd parts) equal the doublet
     channel measure's phi_src (C₀ direction) to 3.8e-16; Σ_a Cf_f / 2 = Cf and Σ_a Gf_f / 4 = Re Gf at the same point
     source (CG solves) to 1.1e-16; the vectorised 6 × 6 Pfaffian of the six-fermion channel equals the Householder
     Pfaffian to 1e-15.
   - chi_f_conn equals ∂⟨O^(a)⟩/∂h_a on a fixed configuration (flavours 4 and 2, h = 0.1) to 5.4e-6 relative.
   - Free theory (y = 0), source on flavour 4, L = 4 (h = 0.01, 0.2) and L = 6 (h = 0.2): the light flavours carry no
     amplitude (phi_f^{1..3} = 0, chi_f_conn^{1..3} = their h = 0 value, phi_T = 0: bipartite propagator), the heavy
     amplitude equals the flavour-blind phi_src / 4 (phi_f^4/h = 0.878 at L = 4, h = 0.01; 0.681 at L = 4, h = 0.2;
     0.694 at L = 6, h = 0.2), and C_T(t) = −(1/V) Σ_{x₀,x⃗} wrap G(x, x₀)³ (≤ 1.3e-19).
   - Stochastic (n_noise = 4, 16 seeds) vs dense on 158 quantities, 4⁴ (y, g₁₀) = (2.41, 0.05), h = 0.1 on flavour 4:
     largest pull 3.46σ, rms 1.01; point-source Gf_f / Cf_f (CG) vs dense 3.2e-11. Dense values: phi_f = (−0.0011,
     −0.0018, −0.0002, +0.0784), chi_f = (0.849, 0.851, 0.856, 2.253), phi_T = −0.0002.

**Method.** Metropolis: single-site Gaussian proposals of σ (width 0.6), accept with |det M|^{1/2} e^{−S_B} (dense
log-determinant of the 128 × 128 M per proposal), two chains of 3000 sweeps, measurements every third sweep after 200,
Householder Pfaffian sign at every measurement. RHMC: τ = 1 with jitter ±30 %, 8 Omelyan steps, 200 thermalisation
trajectories, 1200 measured trajectories, sign from the Schur route (I.9). Errors: sign-reweighted blocked jackknife
with 20 blocks.

**Controls.** The RHMC with C₀ of the wrong sign on the odd sublattice (in the operator and in the measure, 400
trajectories) is rejected: largest pull 111σ (chi_f_light), phi_f_heavy −0.0179(6) (104σ), chi_f_heavy 0.0902(19) (95σ),
E_bond 6.3σ. The connected response with the sign of its second
exchange term flipped misses the derivative by 100 %.

**Caveats.** On the 2³×4 box the kinetic matrix has time hopping only (L = 2 axes cancel); the chain comparison tests
the algorithm and the measure, not 4D physics — the 4⁴ comparison with full hopping is I.10. The sign is exercised
trivially at h = 0.3 on 2³×4 (no negative Pfaffian); its non-trivial cases are in I.9.

**Data.** `data/derived/n1inst/I8_chains.npz` (the four chains: series, acceptance, dH, boson-action trace;
`scripts/derive_n1inst.py i8`).

**Check.** `python claims/I.8/check.py` (≈ 8 min): items 1, 2 and 4 live; item 3 from the frozen chains, with the first
206 Metropolis sweeps and the first 202 trajectories of both RHMC chains regenerated live and required bit-identical.

**Provenance.** Notebook claims C140, C142 (unhiggsed_notebook @ 3c4a02c).
