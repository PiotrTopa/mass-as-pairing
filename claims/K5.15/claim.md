# K5.15 — The light block of the source per box, and the restricted-mode control: with the 8⁴ light doublet restricted to its source-free modes the light pair susceptibility relative to free falls faster with the volume on (4⁴, 8⁴); no SSB-scale growth on (4⁴, 8⁴) at P_c

**Statement.** The source h C_χ has a light-doublet block that depends on the box: ‖P_L C_χ P_L‖/‖C_χ‖ = 0 on 4⁴ and 6⁴,
0.368 on 8⁴, 0.195 on 4³×8 and 0.173 on 6³×12 (all-antiperiodic), with ‖P_L C_χ P_L v_p‖ = 2^{−3/2}·||cos p₀cos p₃| −
|cos p₁cos p₂|| on every momentum where the light/heavy split exists. On 4⁴ and 6⁴ every light momentum is source-free; on
8⁴ 1536 of 4096 are, on 4³×8 and 6³×12 none. A complete taste split (K5.13) is therefore not a source-free light doublet.
The clean volume comparison uses source-free light modes on both boxes: with χ_L on stored 8⁴ (3.0, h = 2)
configurations restricted to the 1536 source-free momenta, the light pair susceptibility relative to free falls faster
with the volume on (4⁴, 8⁴) than with all light modes, e − e_free = −0.68(5) against −0.31(6); restricted to the 256
momenta nearest the corners (the 8⁴ analogue of the 6⁴ mode set) it falls on (6⁴, 8⁴) as well, −1.08(11). The
source-dressed 8⁴ modes carry more than their free share of χ_L, so they do not cause the fall of K5.13. At P_c the
(4⁴, 8⁴) pair shows no SSB-scale growth: e − e_free = −0.114(14), −0.080(8), −0.024(4), +0.025(4) at h = 0, 0.5, 1, 2.

**Numbers.** (K5.15 check output)

| box (aaaa) | ‖P_L C_χ P_L‖/‖C_χ‖ | momenta with a split | source-free |
|---|---|---|---|
| 4⁴ | 0 | 256 / 256 | 256 |
| 6⁴ | 0 | 256 / 1296 | 256 |
| 8⁴ | 0.3684 | 4096 / 4096 | 1536 |
| 4³×8 | 0.1951 | 512 / 512 | 0 |
| 6³×12 | 0.1730 | 768 / 2592 | 0 |

Control at (3.0, h = 2), 8⁴ stage-1c replicas rA + rB (45 + 45 configurations):

| 8⁴ light modes | χ′/free rA / rB | pooled | e − e_free (4⁴, 8⁴) | e − e_free (6⁴, 8⁴) |
|---|---|---|---|---|
| all (stored χ_L) | 0.0186(33) / 0.0108(27) | 0.0156(21) | −0.306(58) | +0.540(134) |
| source-free (1536 momenta) | 0.0065(9) / 0.0041(7) | 0.0056(6) | −0.676(49) | −0.352(110) |
| IR class (256 momenta) | 0.0029(4) / 0.0017(3) | 0.0024(3) | −0.976(50) | −1.076(113) |

Per configuration the IR class carries 5.4 % (median 7.6 %) of χ_L against 44.8 % in the free theory, the source-free class
17.5 % (median 22.7 %) against 54.1 %.

P_c = (2.41, −0.01):

| h | e − e_free (4⁴, 8⁴) | g(8⁴)/g(4⁴) | e − e_free (6⁴, 8⁴) | g(8⁴)/g(6⁴) |
|---|---|---|---|---|
| 0 | −0.114(14) | 0.878 | −0.086(34) | 0.992 |
| 0.5 | −0.080(8) | 0.920 | −0.049(22) | 1.008 |
| 1 | −0.024(4) | 0.926 | +0.021(10) | 1.006 |
| 2 | +0.025(4) | 0.933 | +0.053(10) | 0.972 |

**Method.** `masspairing.analysis.restricted_modes`. (a) The light block on the V twisted plane waves, matrix-free (P_L by
FFT, C_χ sparse); the closed form evaluated separately; ‖·‖_F² = Σ_p ‖P_L C_χ P_L v_p‖². (b) The restricted χ′ = V⟨φ_L′²⟩
with the light pattern C_L′ = W c Wᵀ, W an orthonormal basis of P_L restricted to a momentum subset, from one sparse LU of
the doublet operator per configuration (`chi_reduced`); the per-configuration values (`data/derived/k5control/`) were
produced by `scripts/run_k5_control.py` from the stored configurations of the raw archive (12.7 CPU-hours). χ′/free uses the
free value of the same restricted observable (σ = 0); the free exponent of a pair is ln(χ′_free(8⁴)/χ_free(L⁴))/ln(V₈/V_L).
4⁴: stage 0, trajectories ≥ 150 (n = 225); 6⁴: stage 1b new trajectories (n = 250). Errors as in K5.13: 10 blocks per
member (20 pooled), raised to σ_naive √(2τ_int). (c) The P_c pair with the same reader from the derived stage-0 (4⁴, ≥ 150)
and stage-1 (6⁴, 8⁴, ≥ 100) chains; g = G_L(p_min)/free.

**Controls.** The reduced-basis estimator with the full light basis (and on 6⁴ with the IR-class basis, which there is the
whole split) reproduces the stored χ_L of stored 4⁴ and 6⁴ configurations to 7·10⁻¹⁵. The 90 control rows carry, at their
trajectories, exactly the stored χ_L of the derived stage-1c chains, and the full-mode row on these 90 configurations
is consistent with the K5.13 reading over all 325 (0.0151; −0.317(39) on (4⁴, 8⁴)). The live light block equals the closed form on every momentum of every box.

**Caveats.** h = 2 only; 90 of the 325 pooled 8⁴ configurations at that h. The IR class on 8⁴ sits at |cos p| = cos(π/8),
the 6⁴ modes at cos(π/6): not identical momenta (closer to the corner on 8⁴, the direction of stronger screening), so the
(6⁴, 8⁴) −1.08 bounds the like-for-like fall rather than measuring it. At P_c the 8⁴ chains have 45 measurements.
Exponents are finite-size ratios at L ≤ 8. Spontaneous breaking at (3.0, h = 2) in the measured (0,3) channel is not
resolved by this control.

**Data.** `data/derived/k5control/control_L8_y3_h2.npz` (per configuration: trajectory, χ′ and φ′ of both restrictions,
stored χ_L), `data/derived/k5control/free_L8_h2_restricted.json`; chains as in K5.13; stored configurations
`data/configs/n1_L4_y3_h2_aaaa.npz`, `n1_L6_y3_h2_aaaa.npz`.

**Check.** `python claims/K5.15/check.py` (≈ 25 s).

**Provenance.** Notebook claim C219 (unhiggsed_notebook @ 0624ddc).
