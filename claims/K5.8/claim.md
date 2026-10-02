# K5.8 — stage 1b at y = 3.0 (SMG), by the pre-registered clauses on the new samples: (a), the light half keeps its own gap at h = 1 and h = 2

**Statement.** With the explicit taste-chiral Majorana mass h C_χ on the heavy doublet, in the SMG phase (y = 3.0), the
stage-1b clauses applied to the new samples only (8⁴: 100 measurements at h = 1 and at h = 2; 6⁴: 250 each), with the
stored h = 0 chains as denominators, read class **(a)**: the light pair-channel midpoint cosh mass relative to h = 0 and to
the free ratio is r_L^cosh(8⁴) = 0.976(2) at h = 1 and 0.920(2) at h = 2 (≥ 0.7 − 2σ), the light momentum readout is a
Luttinger-zero-like 0.3 % of free (h-independent, 1.3–1.5 % on 6⁴: falling with V), and the SSB clause (b) does not fire
(no SSB-scale volume exponent, no p = 0 peak sharper than free). Recorded at h = 2, as the order prescribes: the partial
outcome "growth relative to free, no SSB-scale exponent" (e − e_free > 0 at 3.3σ while the (b1) threshold fails) on a
light pair susceptibility that is 1.4 % of free.

**Numbers.** (K5.8 check output)

| clause | h = 1 | h = 2 |
|---|---|---|
| r_L^cosh(8⁴) [6⁴] | 0.976(2) [0.986(4)] | 0.920(2) [0.923(4)] |
| g = G_L(p_min)/free, 8⁴ [6⁴] | 0.0030 [0.013] | 0.0034 [0.015] |
| e − e_free on (6⁴, 8⁴) | 0.42(26) (null: within 3σ of 0) | 0.456(137); e − e_free − 2σ = 0.18 < 0.5 |
| R_peak(8⁴) = [χ_L(0)/χ_L(p_min)]/free | 0.93(26) | 0.80(12) |
| χ_L/free 8⁴ vs 6⁴ | 0.0054(15) vs 0.0033(4), 1.3σ | 0.0142(20) vs 0.0084(6), 2.8σ |
| class | (a) | (a) |

- Floor (recorded, the stored 6³×12 h = 0 chain): midpoint cosh mass 0.423(3) ≥ 0.3 at 2σ.
- Free gate: G_L^free(p_min) = 1 to 1e-9 on 6⁴/8⁴ at h ∈ {1, 1.5, 2, 3} (6³×12: 0.9949–0.9994, boundary modes); free
  midpoint cosh-mass shift at 8⁴: 0.037 (h = 1), 0.123 (h = 2), removed by the free normalisation; 0 at 6⁴.
- h = 2 at 8⁴: condensate bound |⟨φ_L⟩| ≤ √(χ_L/V) = 3.9·10⁻⁴ (free 3.3·10⁻³); explicit heavy amplitude φ_R = 7.5·10⁻⁴
  (free 0.072): the elementary sector stays screened.
- Pooled (stored + new; labelled, not the verdict): class (a); e − e_free = 0.20(22) / 0.453(102); the pooled values of
  8⁴ h = 1 and 6⁴ h = 2 are not used (K5.7, S7′).
- Recorded beside the clauses: the light midpoint cosh mass moves by −0.5 % (8⁴) / −7.7 % (6⁴) from h = 0 to 2 (new
  samples); the 3-point cosh fit is accepted at (8⁴, h = 2) (0.755(1), χ²/dof 2.8) and rejected at h = 1 (χ²/dof 11): the
  pair channel is not a single cosh. α_L on the new configurations (K6.3 method, 8⁴ subsampled): 0.944(8) / 0.920(10)
  (6⁴, h = 1 / 2), 0.955(6) / 0.959(6) (8⁴), p-odd norm 5 % / 3 % of free.

**Method.** `masspairing.analysis.stage1b.verdict_y3`: primary gap estimator the midpoint cosh mass of the mean light
correlator CL_t (a two-fermion pair channel; delete-one-block jackknife with the error-model inflation);
r_L^cosh = [m(h)/m(0)]/[m^free(h)/m^free(0)]; (a) r_L ≥ 0.7 − 2σ and g ≤ 0.1 + 2σ at every h, not (b); (b) = (b1)
e − e_free − 2σ > 0.5 and (b2) R_peak(8⁴) − 2σ > 1 and sharper than 6⁴ and (b3) χ_L/free rising with V at 2σ; (c)
r_L + 2σ < 0.3 with g ≥ 0.8 − 2σ.

**Pre-registration and deviations.** Criteria fixed before any new trajectory (notebook commit 27ea0ee); implementation
fixed before any stage-1b observable was printed (notebook commit 645486b). The free gate "= 1 to 1e-9" is applied on
6⁴/8⁴ and recorded on 6³×12; the 6³×12 chains were not extended, so no 6³×12 clause enters (the order's (a) is written for
8⁴). The floor is the order's post-hoc re-specification of the stage-1 floor, recorded.

**Controls.** S2′ (injected SSB: χ_L(8⁴, h = 2) × V₈/V₆ and χ_L(p_min) × 0.5) → (b) with all three clauses; S3′ (injected
free return: g(2) := 1, m(2) := 0.25 m(0)) → (c); S6′ (free null: r_L = 1, χ_L = free, g = 1) → UNDECIDED (not (a), not
(b)); the stage-0 reader's synthetic self-test passes.

**Caveats.** L ≤ 8; the light pair-channel gap has no plateau at L_t ≤ 12 (K5.7 E8), so it is an effective mass at the
stated t, not a single-fermion mass; the verdict concerns the symmetric gap and SSB only (a light scalar mass term is a
symmetry zero, K5.2). The h = 2 partial outcome is a real 3.3σ growth of a susceptibility at 1.4 % of free.

**Data.** `data/derived/n1stage/S1b/{L8,L6}/*_y3_*h{1,2}_*.npz` (extensions), `data/derived/n1stage/S1/*/*.npz` (stored
references), `data/derived/n1stage/alpha/S1b/*.npz`, free baselines in `data/derived/free/`. Raw archive:
`results/xi_scan/K_N1_S1/{L8,L6}/*_y3_*h{1,2}_*.npz`, `results/laneK5S1bA/alpha/*__new.npz`.

**Check.** `python claims/K5.8/check.py` (≈ 1.5 min): the clauses, the pooled reading and the three controls live.
`tests/equivalence/test_n1stage.py` verifies bit-identity with the notebook's frozen stage-1b rows, verdict and controls.

**Provenance.** Notebook claim C201 (unhiggsed_notebook @ 3c4a02c).
