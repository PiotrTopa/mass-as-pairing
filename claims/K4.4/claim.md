# K4.4 — production-bc continuity at P_c: 6⁴ pppa vs 6⁴ aaaa

**Statement.** At P_c (y = 2.41) on 6⁴, h ∈ {0, 0.5}, with every number read against its free value at the same bc:
the light momentum readout G_L(p_min)/free and the light h-response are boundary-condition independent within 1.3σ;
the light pair susceptibility relative to free is about half as large on the pppa box (its p⃗ = 0 mode is suppressed
more strongly); the light t* log-ratio is bc-dependent as in the free theory; the composite/elementary ratio ρ keeps a
4.4× bc-dependence (9σ) with both values far below 0.2 (elementary side); the pppa h = 0 chain is the slow chain of
the stage-1 set.

**Numbers.** (K4.4 check output)

| 6⁴, y = 2.41 | χ_L/free | G_L(p_min)/free | m_L(t* = 2) [free] | cosh mass t = 2 | ρ (h = 0.5) | φ_R/h × free | A(0.5) [free] |
|---|---|---|---|---|---|---|---|
| pppa h = 0 | 0.31 | 0.740(32) | 0.028(4) [0.213] | 0.20(2) | — | — | — |
| aaaa h = 0 | 0.57 | 0.735(8) | 0.085(3) [0.405] | 0.39(1) | — | — | — |
| pppa h = 0.5 | 0.36 | 0.783(21) | 0.025(2) [0.214] | 0.19(1) | 0.0378(34) | 0.41 | 0.030(11) [0.046] |
| aaaa h = 0.5 | 0.61 | 0.765(9) | 0.089(2) [0.406] | 0.41(1) | 0.0087(4) | 0.63 | 0.038(13) [0.034] |

1. pppa h = 0: bosonic τ_int = 24 trajectories, acceptance 0.66 per 50 trajectories (aaaa ≥ 0.84), 6 of its 10 primary
   series fail the 3σ half-split (aaaa partner: 0); τ_eff = τ_B/Δ = 6.1 measurements.
2. G_L(p_min)/free pulls pppa − aaaa: +0.14σ (h = 0), +0.80σ (h = 0.5); χ_L(0.5)/χ_L(0) = 1.17(11) vs 1.07(2);
   r_L(0.5) (t* log-ratio over free) = 0.88(13) vs 1.04(4).
4. ρ(pppa)/ρ(aaaa) = 4.4×, 9σ; φ_T_R/h = −0.00087 vs −0.00025.
5. A(0.5) pull −0.5σ.

**Method.** Stage-1 reader and error model (`masspairing.analysis.stage1`); free values from the free baselines at the
same bc (free χ_L = 0.0522 (pppa) vs 0.0278 (aaaa) at h = 0).

**Pre-registration and deviations.** The comparison was fixed before the stage-1 data were read (notebook commit
eaa3692) as a recorded comparison; no pass/fail beyond the pulls.

**Caveats.** One volume; the pppa h = 0 chain carries τ_eff = 6.1 and its errors are correspondingly large.

**Data.** `data/derived/n1stage/S1/L6pppa/*.npz`, `data/derived/n1stage/S1/L6/L6_y2.41_*{chiral,h0.5_chiral}_bcaaaa.npz`;
`data/derived/free/free_baselines.json`. Raw archive: `results/xi_scan/K_N1_S1/{L6,L6pppa}/*.npz`.

**Check.** `python claims/K4.4/check.py` (≈ 2 s).

**Provenance.** Notebook claim C197 (unhiggsed_notebook @ 3c4a02c).
