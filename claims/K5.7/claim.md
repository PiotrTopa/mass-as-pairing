# K5.7 — stage 1b integrity, stored-vs-new consistency (S7′), the stage-1 rows with the stage-1b estimators (E1, E8), and the box-shape facts of the light Majorana pattern (E9)

**Statement.**
1. The ten stage-1b chains — four stage-1 chains extended in place to 2000 trajectories (8⁴ and 6⁴ at y = 3.0, h = 1, 2)
   and six new P_c chains (8⁴, 6³×12, 6⁴ at h = 1.5 and 3) — are complete (md5 as delivered), seam-continuous (the first
   1000 trajectories of every extension equal the pre-extension copy), carry one code revision, have no repeated trajectory
   index, good acceptance and a silent kill (A(h) < 0 beyond 3σ nowhere: A > 0 at 95–1001σ at P_c, |A| ≤ 1.9σ at
   y = 3.0, screened); their error model is benign.
2. S7′ (3σ rule): new-only and stored-only values of the extended chains agree for χ_L and G_L(p_min) on all four and for
   the light pair-channel midpoint cosh mass on two; on 8⁴ h = 1 and 6⁴ h = 2 the midpoint cosh mass shifts by −0.6 % /
   −1.1 % at 0.1 % precision (3.4σ / 3.2σ) — recorded as non-stationarity; their pooled values are not used.
3. E1 (recorded): the stored stage-1 rows re-read with the stage-1b estimators reproduce the light pair-channel midpoint
   cosh mass at P_c (8⁴: 0.197 → 0.614 for h = 0 → 2) and its h-stability at y = 3.0; the free exponent e_free = 0.39–0.45
   on (6⁴, 8⁴). E8 (recorded): the 6³×12 light correlator at y = 3.0, h = 0 has no plateau (cosh effective mass 0.91 → 0.42
   over t = 1 … 5; 14 of 15 single-cosh windows with χ²/dof > 3); the free 6³×12 midpoint has no cosh solution (raw r_L there).
4. E9 (box shape): on 4⁴ aaaa the stabiliser of C_χ (32 of 384 elements) splits 16 / 16 into elements keeping and flipping
   the (0,3) anti-self-dual pattern C_asd(0,3), and every flipping element exchanges a spatial axis with time; on 4³×8 the
   box stabiliser (8) keeps it. Hence on L³×L_t boxes the light Majorana one-point function ⟨φ_L⟩ is not a symmetry zero: in
   the free theory it vanishes on hypercubic boxes and is a shape-induced finite-size number on L³×L_t (6³×12: −0.000846 per
   flavour at h = 2, set by the spatial box, falling with L within each L mod 4 class). In the chains ⟨φ_L⟩ is zero within
   errors on 6⁴/8⁴, 0.63–0.88 × free on 6³×12 at P_c, and ≤ 3 % of free at y = 3.0 on 6³×12 (screened).

**Numbers.** (K5.7 check output)

| chain | kind | trajectories | new samples | cadence | min acceptance (new) |
|---|---|---|---|---|---|
| 8⁴ y = 3.0 h = 2 / h = 1 | extension | 2000 | 100 / 100 | 10 | 0.98 / 0.98 |
| 6⁴ y = 3.0 h = 2 / h = 1 | extension | 2000 | 250 / 250 | 4 | 0.96 / 0.94 |
| 8⁴ P_c h = 1.5 / h = 3 | new | 1000 | 46 / 45 | 20 (one 10-step at a restart) / 20 | 0.98 / 0.98 |
| 6³×12 P_c h = 1.5 / h = 3 | new | 1000 | 112 / 112 | 8 | 0.94 / 0.98 |
| 6⁴ P_c h = 1.5 / h = 3 | new | 1000 | 225 / 225 | 4 | 0.98 / 0.98 |

- Code revision efcc72d on every chain; τ_B/Δ ≤ 0.40, inflation ≤ 1.29, half-split pulls ≤ 2.49σ (primary series).
- S7′: 8⁴ h = 1: 0.7201(11) → 0.7160(5), −3.4σ (−0.6 %); 6⁴ h = 2: 0.7125(18) → 0.7044(19), −3.2σ (−1.1 %); 8⁴ h = 2 −0.1σ,
  6⁴ h = 1 +0.1σ; χ_L, G_L(p_min) within 3σ on all four.
- E1: 8⁴ P_c midpoint cosh mass 0.197(9), 0.211(13), 0.332(16), 0.614(5) at h = 0, 0.5, 1, 2; y = 3.0, h = 0 → 2: −0.5 % (8⁴),
  −6.7 % (6⁴), +8.2 % (6³×12); e_free = 0.446, 0.442, 0.431, 0.387 at h = 0, 0.5, 1, 2.
- E8: 6³×12 (3.0, h = 0) cosh effective mass at t = 1 … 5: 0.91, 0.83, 0.60, 0.49, 0.42.
- E9 (group tables recomputed live = frozen): 4⁴: |G| = 384, stabiliser 32 = 16 keep + 16 flip, mean action 0; 4³×8:
  |G| = 96, stabiliser 8, all keep, mean action +1.000. Free ⟨φ_L⟩ per flavour at h = 2 (live): 6³×12 −0.000846,
  4³×8 +0.000505 (= the free baselines), 4⁴ and 6⁴ zero (≤ 1e-19). Frozen scan: 6³×24 −0.000843; 4³×8 +0.000505 →
  8³×16 +0.000045; 6³×12 −0.000846 → 10³×20 −0.00064; 4⁴/6⁴/8⁴ ≤ 3e-19; for scale, the free heavy amplitude per flavour at 6³×12, h = 2 is
  +0.01768 (the light one-point function is ≈ 5 % of it). Chains: max |pull| from zero 1.5σ on 6⁴/8⁴; 6³×12 P_c ⟨φ_L⟩/free = 0.63, 0.69, 0.80, 0.76, 0.88
  at h = 0.5, 1, 1.5, 2, 3 (each > 5σ from zero); 6³×12 y = 3.0: 0.012, 0.026, −0.003 at h = 0.5, 1, 2.

**Method.** Stage-1b reader `masspairing.analysis.stage1b` (new samples: trajectory ≥ 1000 on extensions, ≥ 100 on new
chains; the stage-1 error model; the light pair-channel midpoint cosh mass as the primary gap estimator); md5 and seam
from the derived files (the delivered md5 is recorded in each derived file's meta; the seam compares every stored series
and the configuration digests of the extension's first 1000 trajectories with the pre-extension copy). E9:
`stage1b.element_table` (signed hypercubic symmetries of K, stabiliser of C_χ) and `stage1b.free_phiL`
(⟨φ_L⟩ = (1/2V) Σ C_asd ∘ P_L (K − hC_χ)⁻¹ P_L, dense).

**Pre-registration and deviations.** The stage-1b order (cuts, estimators, S7′ rule) was fixed before any new trajectory
(notebook commit 27ea0ee); the implementation before any stage-1b observable was printed (notebook commit 645486b). R2
(8⁴, h = 1.5) has 46 instead of 50 new measurements with one 10-step at a restart seam. The E1, E8, E9 tables are recorded,
not criteria.

**Caveats.** On L³×L_t and pppa boxes ⟨φ_L⟩ carries a free-sized shape-induced part: every 6³×12 light reading (stage 1
and 1b) includes it, in numerator and denominator alike; the symmetry zeros of the light scalar masses hold on boxes with
the full hypercubic group and in infinite volume (K5.2).

**Data.** `data/derived/n1stage/S1b/*/*.npz` (ten chains), `data/derived/n1stage/S1/*/*.npz`,
`data/derived/n1stage/e9_box_shape.json` (frozen free-theory scan incl. 8⁴, 8³×16, 10³×20), free baselines in
`data/derived/free/`. Raw archive: `results/xi_scan/K_N1_S1b/{L8,L6x12,L6}/*.npz`, `results/xi_scan/K_N1_S1/{L8,L6}/*_y3_*h{1,2}_*.npz`
(extended files), `results/xi_scan/K_N1_S1b/pre/*/*.npz`, `results/laneK5S1bA/e9_box_shape.json`.

**Check.** `python claims/K5.7/check.py` (≈ 2 min): integrity, S7′, E1, E8 from the derived chains; the E9 group tables and
the small-box free values recomputed live.

**Provenance.** Notebook claim C200, with the domain note of C199 item 2 (E9) (unhiggsed_notebook @ 3c4a02c).
