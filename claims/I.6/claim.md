# I.6 — the complete SU(4)-invariant 10/6-channel estimator (dense = stochastic; fermionic boundary sign)

**Statement.** `masspairing.measure.ChannelMeasure` computes

    Σ_ab⟨Φ̄^{ab}(y,y′)Φ^{ab}(x,x′)⟩ = disc₁₀ − 2K₁K₂ − 2tr(G₁G₂) + 2tr(G₃G₄) + 2K₃K₄,
    Σ_ab⟨Λ̄^{ab}(y,y′)Λ^{ab}(x,x′)⟩ = disc₆ − 2K₁K₂ + 2tr(G₁G₂) + 2tr(G₃G₄) − 2K₃K₄

(G₁ = G(y,x), G₂ = G(y′,x′), G₃ = G(y,x′), G₄ = G(y′,x), K = tr G over the four real flavours) as class-averaged structure
factors S(p) = (1/V)Σ_{x even, y odd} cos(p·(x−y)) w(y)w(x)⟨pair(y) pair(x)⟩ at the 16 corner momenta and at corner +
p_min, the local energies ⟨T_Q⟩, ⟨T₆⟩, and the source-direction one-point function with its Wick-connected h-derivative,
densely or from Z2 noise vectors (U-statistic over ordered pairs of distinct vectors, unbiased since every term is a
product of two propagators). A pair operator χ(x)χ(x + δ) that straddles an antiperiodic boundary carries the fermionic
sign w; without it the correlators are not translation invariant.

**Numbers.**
1. Dense estimator = an independent real-flavour Wick evaluation (`masspairing.analysis.k3_exact.channel_reference`) on
   three stored configurations (4⁴ at (2.41, 0.1, 0) and (2.41, 0.05, +0.05), 6⁴ at (2.41, 0.1, +0.1)): 7.4 × 10⁻¹⁷,
   1.1 × 10⁻¹⁶, 1.0 × 10⁻¹⁶ over 32 corner values and both local energies; S(p + π(1,1,1,1)) = −S(p) to 2 × 10⁻¹⁷ (8
   independent corners).
2. Boundary sign: a hot 4⁴ configuration translated by two sites in time changes the structure factors by 1.8 × 10⁻¹⁶
   with the sign and by 4.0 × 10⁻² without it (local energies and source derivative 3.9 × 10⁻¹⁵). Free 4³ × 8, plane
   (0,3), pairs two time steps apart: Σ_ab⟨Φ̄Φ⟩ = −0.4931 on every time slice with the sign; without it +0.4931 on the
   two slices where an odd number of the four sites wrap.
3. Stochastic (8 noise vectors) vs dense, 34 quantities per configuration (χ₁₀, χ₆, E₁₀, E₆, φ_src, ∂φ_src/∂h, corners
   and corner + p_min 1–7 of both channels): 4⁴ (48 seeds) max |pull| 2.33, rms 0.92 and 1.89, 0.87; 6⁴ (32 seeds)
   1.94, rms 0.99. Noise per configuration of χ₁₀: 0.073 / 0.111 (4⁴), 0.036 (6⁴), i.e. 0.23 / 0.34 / 0.17 of the value.

**Method.** Dense: inverse of the 2V × 2V doublet operator; real flavours G = Re[W S W†]. Stochastic: two solves per
noise vector (doublet dilution).

**Controls.** Omitting the K₃K₄ exchange term moves the dense χ₁₀ by 0.14 / 0.095 / 0.056 and the stochastic χ₁₀ by
−19.5σ, −6.1σ, −7.6σ (0.1678 vs 0.3189, 0.2377 vs 0.3302, 0.1646 vs 0.2122): rejected. Replacing tr(G₃G₄) by
Σ_ab G₃^{ab}G₄^{ab} moves the dense χ₁₀ by 1.2 × 10⁻³ / 5.7 × 10⁻³ / 4.3 × 10⁻⁵ — detected by the dense comparison
(tolerance 10⁻¹⁰), below the noise of the stochastic one.

**Caveats.** No dense reference at L = 8; the noise per configuration is larger on softer configurations (K3.1).

**Data.** `data/configs/k3/c072_L4_y2.41_g0.1_g60.npz`, `c072_L4_y2.41_g0.05_g60.05.npz`, `c073_L6_y2.41_g0.1_g60.1.npz`
(final configurations of the archive chains `results/laneW2/c072/…`, `results/laneW2/c073/main/…`).

**Check.** `python claims/I.6/check.py` (≈ 4 min).

**Provenance.** Notebook claim C110 (unhiggsed_notebook @ 3c4a02c).
