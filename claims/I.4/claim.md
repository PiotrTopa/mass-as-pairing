# I.4 — the link-field (wedge) HMC is exact; the per-link completion model is exact

**Statement.** The sign-free link-field model — D_c(σ, s) = (K − Â(s)) ⊗ 1₂ + i y σ·τ with weight
det D_c = det(D_c†D_c)^{1/2} > 0, real Hubbard–Stratonovich fields s_{Q,j} with prior s²/(4g_j), g₁ = g₁₀ − g₆ (ν pairs),
g₂ = g₁₀ + g₆ (μ pairs), on every square of the lattice ("all") or the checkerboard set — is simulated exactly by the
RHMC of `masspairing` (Lanczos actions, Zolotarev forces, Hasenbusch split, nested integrator), and it is exactly the
model whose fermion-bag expansion is the sum-of-squares-completed vertex. The per-link completion model ("links": one
field per link with prior s²/(4c_l g), c_l the number of squares through link l) induces c_l g K_l² per link, i.e. it is
the "all" model at (g, g₆ = 0) with the plaquette term g T_Q removed, exactly; its HMC is exact too.

**Numbers.**
1. Geometry: orientation s_l = −sign K[u,v] = sign of the free propagator on every link of 2³, 4², 4³, 2⁴, 4⁴;
   checkerboard quads = plaquette quads; dense K − A(s) = independent construction (0.0); the identity
   T_Q + Σ_l K_l² = (K̃_ν + K̃_ν′)² + (K̃_μ + K̃_μ′)² and its g₆ = 0.4 version as Grassmann identities on all 16 quads of
   4² and 24 random quads of 4³ of both orientations (0.0). Per-link model: fields = links, c_l = 6 / 3 / 2 / 2 on
   4⁴ / 2⁴ / 2³ / 4², incidence diag(lsign); E_{s∼N(0,2cg)} e^{sK̃} = e^{cgK²} to all orders (0.0);
   Σ_Q[squares − T_Q] = Σ_l c_l K_l² on 4² (192 monomials, 0.0).
2. Exact 2³ (y = 0, staggered units g_bag = g/4): HS partition function by polynomial algebra vs all-orders bag
   enumeration — interior (0.4, 0.1): Z/Z₀ = 157.1025296778 (1970 configurations), relative difference 9.4 × 10⁻¹⁵,
   ⟨Σ term⟩ = 3.35632885 (1.6 × 10⁻¹⁴); edge (0.5, 0.5): 658.8123910465 (1212 configurations), 3.7 × 10⁻¹⁴. Per-link model
   g = 0.3: 7.6193227922 (3095 configurations), 7.9 × 10⁻¹⁵.
3. HMC on 2³ (interior, 3000 trajectories after 300, 5 steps) vs exact: ⟨Σ s²⟩ 15.584(120) vs 15.42896, ⟨Σ term⟩
   3.444(63) vs 3.35633, ⟨Σ s⟩ 7.626(45) vs 7.57243, fermionic bond estimator Σ_f 2g_f⟨B_f⟩ 7.525(18) vs 7.57243:
   pulls +1.30, +1.39, +1.18, −2.69σ.
4. 4⁴ at (1.7, −0.01, 1), g₁₀ = 0.25, g₆ = 0.05: forces vs 4-point finite differences 1.1 × 10⁻⁶ (rational degree 20);
   reversibility 1.2 × 10⁻¹⁴ / 7.4 × 10⁻¹⁴ (CG tolerance 10⁻¹²) and 2.6 × 10⁻⁹ / 6.0 × 10⁻⁸ (10⁻⁸); ⟨|ΔH|⟩ = 0.0896 / 0.0173
   / 0.0043 at 6 / 12 / 24 steps (ratios 5.18, 3.98); Hasenbusch forces 2.2 × 10⁻⁸, 9.6 × 10⁻⁸, 1.0 × 10⁻⁸; nested
   (4, 2, 2) reversibility 2.2 × 10⁻¹⁵ / 2.5 × 10⁻¹⁵. Per-link model at y = 2.41, g = 0.1: det D_c > 0 on 10 hot
   configurations (log det 544.7–575.2), forces 1.1 × 10⁻⁸, reversibility 1.6 × 10⁻¹⁴ / 4.9 × 10⁻¹⁴.
5. 4⁴ thermalised configuration: stochastic estimators of the channel measure (24 noise vectors × 4 seeds) vs dense on
   O₄, |φ_stag|², |φ|², χ₁₀, χ₆, E₁₀, E₆, D_∥², D_⊥² and the four bond energies: largest |pull| 3.64 (a bond energy);
   χ₁₀ 0.208(7) vs 0.2133.
6. 2⁴ (all antiperiodic, (y, κ, λ) = (2, 0, 1)), exact-determinant Metropolis (10000 sweeps) vs RHMC (2000 trajectories):
   wedge model (g₁₀, g₆) = (0.4, 0.1): largest pull 1.89σ over σ², O₄, |Σ_stag|, |φ_stag|², ⟨s²⟩, bond energy, D_∥², χ₆,
   E₁₀, E₆; per-link model g = 0.15 (τ-jitter 0.3): 1.45σ, ⟨s²⟩/(6g) = 1.135.

**Method.** Exact side: P(s) = Pf(K − Â(s))/Pf(K) for one real flavour as a polynomial in the link fields by Wick's
theorem, Gaussian averages of P⁴ in the Hermite basis (`masspairing.analysis.k3_exact.hs_exact`); bag enumeration of
⟨Π_Q T_Q^{k_Q}/k_Q!⟩₀ (`bag_reference`, `masspairing.algebra.bags`). Metropolis: single-variable updates of (σ, s) with
the weight from a dense log-determinant, no pseudofermions and no molecular dynamics (`k3_exact.metropolis`).

**Controls.** (2) a wrong sign in one K̃ term of every ν pair changes Z at O(1) (39.68 vs 157.10); the uncompleted vertex
g₁₀T_Q + g₆T₆ (552 configurations) gives 104.66 vs 658.81; for the per-link model the "all" model with the plaquette term
(61.76) and the prior width 2g (c_l = 1, 3.0012) differ. (6) The RHMC with the wrong K̃ sign is rejected (largest pull
18.4σ, on E₆; 8–14σ on O₄, |φ_stag|², bond energy, D_∥², χ₆, E₁₀); the per-link RHMC with prior s²/(4g) is rejected
(98.3σ on ⟨s²⟩, 33σ on the bond energy).

**Pre-registration and deviations.** The estimator comparisons (5) and the χ₆, E₁₀, E₆ observables of (6) use the
complete 10/6-channel estimator (I.6); the χ₁₀ of the complete estimator vanishes identically on L = 2 (x + δ and x − δ
coincide) and is not compared there. The Metropolis and RHMC series that do not depend on the estimator reproduce the
notebook's runs bit by bit (same seeds).

**Caveats.** (6) compares two samplers of the same operator; the map to the four-fermion term rests on (1)–(2). The
exact 2³ test is at y = 0.

**Data.** `data/derived/k3/bags_2x2x2.json` (all-orders bag references, `scripts/derive_k3.py bags`, about half an hour on one core);
`data/derived/k3/chains/wedge_2x3_hmc.npz`, `wedge_2x4_*.npz`, `links_2x4_*.npz` (`scripts/derive_k3.py chains`).

**Check.** `python claims/I.4/check.py` (≈ 2.5 min): all items; a prefix of every frozen chain is regenerated and must be
bit-identical; the per-link bag enumeration is recomputed live.

**Provenance.** Notebook claims C070, C111 (unhiggsed_notebook @ 3c4a02c).
