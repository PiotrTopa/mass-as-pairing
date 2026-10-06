# K8.4 — at P_c the light half approaches its free values as h grows and never overshoots them; the light pair-channel effective mass measures the distance from criticality, not the phase

**Statement.** At P_c = (2.41, −0.01), model N1 with h C_χ on the heavy doublet, all-antiperiodic boxes, every light
observable is read against the free theory on the same box, h and boundary conditions:
1. On 8⁴ and 6⁴ the light pair-channel midpoint cosh mass m stays below the free value m_free at every h ∈ [0, 3] (2σ)
   and rises toward it: m/m_free = 0.129 → 0.442 on 8⁴ and 0.407 → 0.670 on 6⁴ for h = 0 → 3.
2. The light single-fermion readout g = G_L(p_min)/free and the light pair susceptibility c = χ_L/free rise from h = 0 to
   3 beyond 2σ on 8⁴, 6⁴ and 6³×12, never exceed 1 (2σ), and exceed 0.8 at h = 3 (8⁴: g 0.730 → 0.867, c 0.517 → 0.920).
3. On 6³×12 (where the free correlator has no cosh solution at the midpoint) the ratio estimator
   dmeff = −ln[R(t+1)/R(t)], R = C/C_free, falls from +0.303 to +0.146 — also toward free. It is positive already at the
   critical h = 0 point (boundary modes of the free 6³×12 correlator) and is not read as a gap.
4. On 6⁴ (h = 0) the pair-channel mass is 0.821 × free at y = 2.0 (symmetric phase) and 0.793 × free at y = 3.0 (SMG),
   while g is 0.914 vs 0.0125: m measures how far a point is from criticality on either side and cannot tell which side.

The rise of the light pair-channel effective mass at P_c (K8.1, K8.2) is therefore the light half becoming free-like as
the critical point is detuned; there is no light gap beyond free at L ≤ 8.

**Numbers.** (K8.4 check output)

| lattice | h | m | m_free | m/m_free | g | c |
|---|---|---|---|---|---|---|
| 8⁴ | 0 | 0.197(9) | 1.519 | 0.129 | 0.730 | 0.517 |
| 8⁴ | 1 | 0.332(16) | 1.556 | 0.213 | | |
| 8⁴ | 2 | 0.614(5) | 1.643 | 0.374 | | |
| 8⁴ | 3 | 0.764(4) | 1.731 | 0.442 | 0.867 | 0.920 |
| 6⁴ | 0 | 0.392(8) | 0.962 | 0.407 | 0.735 | 0.568 |
| 6⁴ | 3 | 0.645(2) | 0.962 | 0.670 | 0.887 | 0.880 |
| 6³×12 | 0 / 3 | | | | 0.677 / 0.867 | 0.563 / 0.813 |
| 6⁴ y = 2.0 | 0 / 3 | 0.791 / 0.824 | 0.962 | 0.821 / 0.856 | 0.914 / 0.952 | 0.892 / 0.951 |
| 6⁴ y = 3.0 | 0 / 2 / 3 | | 0.962 | 0.793 / 0.740 / 0.633 | 0.0125 → 0.0204 (h = 3) | 0.0019 → 0.0164 |

**Method.** `masspairing.analysis.direction.free_compare`: rows by the stage-1b reader (`stage1b.read_row`, cut at
trajectory 100, error model C0); stored stage-1 rows at h ∈ {0, 0.5, 1, 2}, stage-1b rows (new samples) at h ∈ {1.5, 3};
the 6⁴ y = 2.0 (h = 0, 3) and y = 3.0 (h = 3) chains of data/derived/n1stage/TH; free values from the free baselines at
the same (L, L_t, bc, h). dmeff: delete-one-block jackknife over the CL_t blocks with the error-model inflation.

**Controls.** A synthetic 6⁴ h = 3 row 10 % above free (a gap beyond free) is flagged by criterion 1; a synthetic 8⁴ h = 3
row with the SMG value of g is flagged by criterion 2.

**Caveats.** Post hoc (a re-analysis of certified rows; the criteria were written with the rows known). g and c are
readouts at p_min ≈ π/L: a symmetric light gap well below π/L would leave them free-like, so this is a finite-volume
statement (L ≤ 8). The pair channel has no plateau at L_t ≤ 12 (K5.7): m is an effective mass at t = L_t/2 − 1. The 6⁴
y = 2.0 / 3.0 (h = 3) chains have 88 measurements each.

**Data.** `data/derived/n1stage/{S1,S1b,S1c}/`, `data/derived/n1stage/TH/{sym6,y3h3}/*.npz`, free baselines in
`data/derived/free/`. Raw archive: `results/xi_scan/K_N1_S1{,b,c}/`, `results/laneTH/{sym6,y3h3}/*.npz` (bundle
`n1-k8`).

**Check.** `python claims/K8.4/check.py` (≈ 15 s).

**Provenance.** Notebook claim C210 (unhiggsed_notebook @ ea0ce47).
