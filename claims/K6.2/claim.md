# K6.2 — The odd-part exponent α: validation, and SYM/SMG separated per configuration

**Statement.** Of the real-valued readouts that the frequency loop of the corner-block propagator carries (K6.1), the
odd-part frequency exponent α = ln(‖O(p₃)‖/‖O(p₁)‖)/ln(sin p₃/sin p₁) between the two lowest shells (O the p-odd part of
the corner block; α = −1 for a pole at the corner, → +1 for a Luttinger zero with a large gap) is validated on known
propagators, each case with a sabotage, and on 39 stored equilibrium configurations it separates the SMG from the SYM
phase configuration by configuration, with no overlap at L = 6 and 8. The loop carries no integer (no configuration is
quantised), and the Majorana fraction f_M tracks the local mass y|σ| and does not identify the AFM phase.

**Numbers.**
1. Synthetic partner propagator G(p) = (ip + M_p)/((ip)(ip + M_p) − Δ²), p → 2 sin p₀ (odd part on Γ₀, even part on the
   ε-mass structure X₁₁₁₁):
   - M_p = 0 (a zero-type gap): f_M = 0 and α > 0 iff Δ² > 4 sin p₁ sin p₃, i.e. Δ > 1.414 (L_t = 6), 1.189 (L_t = 8);
     α rises monotonically with Δ to +0.935 / +0.951 at Δ = 8 (L_t = 6 / 8): the "zero" reading needs a gap above ≈ 1.2
     at L_t = 8.
   - M_p ≠ 0 (a partner Majorana mass): f_M > 0, and the reading is "majorana-pole" iff the light eigen-mass of the
     two-state system exceeds 2 sin p₁ (e.g. Δ = 4, M_p = 2: f_M = 0.699 / 0.805). The seesaw corner (M_p = 3, Δ = 0.5,
     light mass 0.081) reads "gapless" (f_M = 0.005 / 0.010): a light pole below the loop resolution.
2. Free operator: massless — reading "gapless", α = −1 exactly, and quantised (trivially, N_odd = 32, f_M = 0; K6.1);
   Dirac mass m: f_M = m²/(m² + 4 sin²p₁) to 1e-8 (L = 6: 0.2000 at m = 0.5, 0.8000 at m = 2).
3. K − hC_χ (one taste doublet gapped; heavy half = range of the p = 0 corner block of C_χ, rank 8): heavy f_M within
   −25 / −11 / −5 % (L = 4 / 6 / 8, h = 0.5) and −20 / −7 / −3 % (h = 1) of h²/(h² + 4 sin²p₁) (the two-link form factor at
   p₀ = π/L_t); light half f_M ≤ 0.011 and α = −1.007 … −1.081 at L ≥ 6 (massless).
4. The 39 stored configurations (27 at L = 6, 12 at L = 8; the final configuration of each chain of the phase-labelled
   set, labels from the archive):
   - none quantised: N_odd ∈ [8.2, 30.4] (n = 32), f_M ∈ [0.09, 0.82]; 0/5 (L = 6) and 0/4 (L = 8) scored gapped
     (AFM/SMG, not near-critical) configurations; det G_B real on all 39 (|Im det|/|det| ≤ 1.8e-11), half turns on 3, N_det
     ∈ ½ℤ: never a winding.
   - f_M rises monotonically with y through SYM → SMG (it measures the local mass y|σ|); the AFM files (0.685 at L = 6,
     0.527 at L = 8, both near-critical) lie inside the SYM/SMG range (up to 0.821 / 0.807): f_M does not identify AFM.
   - α separates SMG from SYM among scored, non-near-critical files: L = 6 min α(SMG) = −0.492 (5 files) > max α(SYM) =
     −0.911 (6 files), margin 0.42; L = 8 −0.605 (4) > −0.779 (3), margin 0.17. Every such SYM file has α = −1 ± 0.25 (the
     corner pole survives the O(1) random σ configuration by configuration); α reaches +0.636 at (y, κ) = (2.792, −0.01),
     L = 8. The six critical (P_c) files scatter over α ∈ [−1.16, −0.17].
   - Volume trend (recorded): (2.792, −0.01): α = +0.380 → +0.636 from L = 6 to 8; (2.537, −0.01): −0.001 → −0.044 (a gap
     at the loop resolution: the zero-type α of item 1 is 0 at Δ ≈ 1.2–1.4).

**Method.** D_c(σ) = K ⊗ 1₂ + i y σ·τ with the stored y and bc, one sparse LU per configuration, the corner block at every
p₀ of the loop at p⃗ = 0 (`corner.analyse`); readings by the pre-registered rule (`corner.classify`: f_M ≥ 0.5 →
majorana-pole, else α > 0 → zero, else gapless) and quantisation test (`corner.quantised`).

**Pre-registration and deviations.** The quantisation test, the AFM test by f_M and the SYM/SMG test by α were fixed
before any stored configuration was read (notebook commit 2ead635).

**Controls.** Item 3: the un-phased plane pair (not a mass) gives f_M = 0 (< 1e-28) at h = 1, and the nodal source C₀
gives f_M = 0 and α = −1 at h = 0.5 and 2 (gapless at every h). Item 4: with the SYM/SMG labels swapped the α test fails;
with the scored near-critical files included the separation holds but narrows (L = 6: −0.837 vs −0.844; L = 8: −0.736
vs −0.779).

**Caveats.** One configuration per chain; no ensemble statistics. The physical propagator is the ensemble average of the
blocks; α is a per-configuration diagnostic of whether the corner pole survives, not a gap measurement.

**Data.** `data/derived/n1inst/K6_winding_stored.json` (the 39 records, regenerated from the archive chains by
`scripts/derive_n1inst.py winding`; phase labels from `results/laneK2/c051/phase_labels.json`; the chain files are listed
in `data/MANIFEST.tsv`); `data/configs/n1inst/L6_y2.028_eps_pppa.npz`, `L6_y2.792_eps_pppa.npz` (two records recomputed
live). Table `results/K6_alpha.csv` (`scripts/table_n1inst_alpha.py`), figure `figures/k6_alpha.pdf`
(`scripts/fig_n1inst_alpha.py`).

**Check.** `python claims/K6.2/check.py` (≈ 2 min): items 1–3 live; item 4 from the frozen records with two 6⁴ records
recomputed live (≤ 1e-10).

**Provenance.** Notebook claims C182, C183 (unhiggsed_notebook @ 3c4a02c).
