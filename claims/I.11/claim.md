# I.11 — The light/heavy taste observables of the N1 measure are exact

**Statement.** With the taste-chiral mass C_χ on one taste doublet ("R", heavy) the N1 measure (`measure.N1Measure`,
`taste=True`) splits every fermion observable into the heavy and light doublets with the taste projectors P_R, P_L
(K5.1): heavy pair channel phi_f/chi_f (pattern C_χ) with the flavour-summed susceptibility chi_f_sum; light pair channel
phi_L, chi_L, chi_L_conn, chi_L_corners, chi_L_pmin and chi_L_sum = V⟨φ_L²⟩ (pattern C_L = P_L C_asd P_L); the light
(6,1,1) channel phi6L, phi6L_stag, phi6L_sq, phi6L_stag_sq, O4L on G_L = P_L G P_L; the momentum readouts GL_p0, GR_p0,
G_p0 at the smallest twisted momentum; the all-source time-slice correlators CL_t, GL_t, CR_t, GR_t; the composites
phi_T_L, C_T_L, phi_T_R, C_T_R; and, on the stochastic path, the point-source correlators of `taste_correlator`. These
estimators are exact: free-theory identities hold to rounding, the connected susceptibilities are the responses to the
corresponding sources, and the stochastic estimators agree with the dense ones.

**Numbers.**

1. Free theory (y = 0), all-antiperiodic 4⁴ and 4³×8, h ∈ {0, 0.5}:
   - light/heavy split Σ_t(CL_t + CR_t) = Σ_t C_full(t), ½(GL_p0 + GR_p0) = G_p0: ≤ 2.4e-14;
   - h = 0: light = heavy (CL_t = CR_t, GL_t = GR_t, GL_p0 = GR_p0 = 1): ≤ 1.2e-15;
   - h = 0.5: momentum readouts = 4Σsin²p₀/(4Σsin²p₀ + m(p₀)²) with m_R = ½h(c_D + c_dual), m_L = ½h(c_D − c_dual) to
     1.4e-15: 4⁴ GL_p0 = 1.00000, GR_p0 = 0.99225; 4³×8 GL_p0 = 0.99978, GR_p0 = 0.98753;
   - phi6L = 0 and phi_R = the doublet-code phi_src of the channel measure (≤ 1.7e-18); on 4⁴ phi_L = 0 exactly (the
     symmetry zero of K5.2); on 4³×8 phi_L = +5.0e-4 (recorded: the box is not hypercubic-symmetric, K5.2).
   - the composites phi_T_R, phi_T_L are non-zero at h ≠ 0 (4⁴: −9.3e-7; 4³×8: −1.2e-6, −1.6e-10): allowed one-point
     functions in the source direction (K5.2).
2. Response identities on an interacting 4⁴ all-antiperiodic configuration (y = 2.41, κ = −0.01, h = 0.3):
   chi_f_sum_conn = ∂⟨Σ_a O^(a)⟩/∂h (0.02978 vs 0.02978), chi_L_sum_conn = ∂⟨O_L⟩/∂h_L with a light source h_L C_L added
   to the operator (0.03000 vs 0.03000): relative deviation 2.7e-8.
3. Stochastic (n_noise = 4, 16 seeds) vs dense on that configuration: 135 quantities, largest pull 3.31σ, rms 1.22;
   the point-source taste correlators (CG) equal the dense P G P columns to 3.3e-12. Dense values: phi_R = 0.0089,
   chi_L_sum = 0.0300, GL_p0 = 1.002, GR_p0 = 0.998.
4. The taste channels are added only for the chiral pattern: the C₀ measure has no taste keys and its pair channel is
   the C₀ one.

**Method.** Dense propagator G = M⁻¹ in the real 4V basis; P_L, P_R applied by FFT (K5.1). The response identities use
central differences (step 10⁻³) of the configuration-wise one-point functions. The stochastic estimators use Z2 noise
diluted over the four real flavours, U-statistics over distinct samples for every product of two propagators.

**Controls.** The connected susceptibility with the sign of its second exchange term flipped misses both responses by
100 %. The light/heavy split and the h = 0 identity hold only if P_L and P_R are complementary projectors that commute
with K.

**Caveats.** The all-antiperiodic boxes with L ≡ 0 mod 4 have no zone-boundary modes (P_L + P_R = 1); on other boxes
the taste observables are exact on the modes where the split exists (K5.1).

**Data.** None (configurations generated in the check with fixed seeds).

**Check.** `python claims/I.11/check.py` (≈ 1 min): prints and asserts every number above.

**Provenance.** Notebook claim C162 (unhiggsed_notebook @ 3c4a02c).
