# K8.9 — the pre-registered direction test repeated on 8⁴: the light clauses again point toward the symmetric side, the S(π) clause is again short (MIXED as written); the volume trend at h = 3 is split (g flags a small excess loss at P_c, c does not)

**Statement.** The test P1 of K8.5 applied unchanged to 8⁴ rows (P1′), with a new 8⁴ all-antiperiodic N1 pair at the
symmetric point y = 2.0, h = 0 and 3 (500 production trajectories each, 40 dense measurements after the cut, acceptance
≥ 0.98 per 50 trajectories; data sufficiency FULL), the stored 8⁴ P_c (h = 0), stage-1b P_c (h = 3) and SMG (y = 3.0, h = 0)
rows as the other references:
1. Clause (i) holds: the positions of P_c between the symmetric point and SMG move toward the symmetric point,
   f_g 0.190 → 0.084 (Δ = −0.106(38), 2.8σ) and f_c 0.419 → 0.051 (Δ = −0.367(24), 15σ).
2. Clause (ii) fails: dS = |S_Pc(3) − S_SMG(0)| − |S_Pc(3) − S_SYM(3)| = +0.300(203), 1.5σ. The verdict as written is
   **mixed / undecided**, as on 6⁴ (1.9σ there); no "toward SMG" clause holds. S(π) peaks at the critical point and
   falls on both sides, so this clause is weak by construction.
3. Amendment 1 (volume trend at h = 3, v_X = O_X(8⁴)/O_X(6⁴)) is **split**: for g, v_Pc = 0.9776(27) against
   v_SYM = 0.9936(19), a difference of −0.0159(33) (4.8σ) → "gap beyond SYM dressing"; for c, v_Pc = 1.0455(114) against
   v_SYM = 1.0204(62) → "free-like volume trend". The light single-fermion readout at P_c(h = 3) loses 1.6 % more from 6⁴
   to 8⁴ than at the symmetric point while the pair susceptibility does not: a small symmetric light gap well below π/L
   at P_c(h = 3) is not excluded at L ≤ 8.
4. Recorded: on 8⁴ the h-matched O4_Pc(3)/O4_SYM(3) = 1.74 (6⁴: 1.74, K8.7); the light pair-channel mass is 0.63 / 0.65 ×
   free at the symmetric point (h = 0 / 3), 0.47 × free in SMG and 0.44 × free at P_c(3) — below both phases.

The claim certifies the outcome as it fell (MIXED; amendment split), not a "toward SYM" verdict.

**Numbers.** (K8.9 check output)

| 8⁴ row | y | h | g | c | S(π) | O4 | m (free) |
|---|---|---|---|---|---|---|---|
| P_c | 2.41 | 0 | 0.730(34) | 0.517(19) | 26.0(2.3) | 0.0355 | 0.197 (1.519) |
| P_c | 2.41 | 3 | 0.867(2) | 0.920(9) | 2.135(72) | 0.0104 | 0.764 (1.731) |
| SMG | 3.0 | 0 | 0.0028 | 0.0022(11) | 3.022(122) | 0.0863 | 0.716 (1.519) |
| SYM | 2.0 | 0 | 0.900(2) | 0.888(2) | 2.573(143) | 0.0114 | 0.965 (1.519) |
| SYM | 2.0 | 3 | 0.946(2) | 0.970(6) | 1.548(76) | 0.0060 | 1.123 (1.731) |

**Method.** `masspairing.analysis.direction.p1prime`: rows by the stage-1b reader (cut at trajectory 100, error model
C0); clauses, Gaussian Monte Carlo and classifier controls as in K8.5; amendment 1 with the 6⁴ P_c(3) and y = 2.0, h = 3
rows of K8.5.

**Pre-registration and deviations.** P1′ (the clauses of P1, unchanged; data-sufficiency levels; a time box for the
chains) was committed before the two chains started; amendment 1 was committed while they ran, before any output was
read (notebook lane THP). The chains stopped at the time box after 500 production trajectories each (FULL).

**Controls.** P1 classifier: a synthetic SMG-ward row returns "toward SMG", a row at the symmetric point "toward SYM".

**Caveats.** 40 dense measurements per new chain; the 8⁴ P_c h = 0 anchor has 45. Two volumes. g and c are read at
p_min ≈ π/L.

**Data.** `data/derived/n1stage/TH/sym8/*.npz` (the 8⁴ y = 2.0 pair), `data/derived/n1stage/{S1,S1b}/L8/`,
`data/derived/n1stage/{S1,S1b}/L6/`, `data/derived/n1stage/TH/sym6/`, free baselines in `data/derived/free/`. Raw
archive: `results/laneTHP/sym8/h{0,3}/*.npz` (bundle `n1-k8`).

**Check.** `python claims/K8.9/check.py` (≈ 15 s).

**Provenance.** Notebook claim C215 (unhiggsed_notebook @ f5e5047).
