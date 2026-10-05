# K5.13 — Taste-projector coverage: the light/heavy split exists on 19.8 % of the momenta on 6⁴ and on all of them on 4⁴, 4³×8, 8⁴; on the complete pair (4⁴, 8⁴) at y = 3.0 the light pair susceptibility relative to free falls with the volume

**Statement.** On all-antiperiodic boxes the light/heavy taste split exists only on the momenta where every cos p_μ ≠ 0;
with p_μ = (2n+1)π/L this fails iff L ≡ 2 mod 4. The split is complete on 4⁴, 4³×8, 8⁴ and 12⁴; on 6⁴ it exists on
(4/6)⁴ = 19.8 % of the momenta (the four of six per axis nearest a corner), on 6³×12 on 29.6 %, on 10⁴ on 41.0 %. Every
6⁴ light-sector observable is therefore built from the infrared-most fifth of the modes and every 8⁴ one from all of them,
so a (6⁴, 8⁴) ratio of a light-sector observable compares two different mode sets. On the complete pair (4⁴, 8⁴) at
y = 3.0 the light pair susceptibility χ_L = V⟨φ_L²⟩ relative to free falls with the volume: e − e_free = −0.32(4) at
h = 2 and −0.44(7) at h = 1 (e = d ln χ_L/d ln V), against +0.51(9) on (6⁴, 8⁴) with the same reader — the two pairs
differ by 8.9σ. At h = 2 χ_L/free falls monotonically over the three complete boxes, 0.0365 (4⁴) > 0.0255 (4³×8) > 0.0151
(8⁴), and the 6⁴ value 0.0084 lies below all three. The (6⁴, 8⁴) growth of K5.8 and K5.11 is not reproduced on a pair
with a complete taste split.

**Numbers.** (K5.13 check output)

| h | χ_L/free 4⁴ | 8⁴ pooled (n) | e_free(4,8) | e − e_free (4,8) | e − e_free (6,8), same reader |
|---|---|---|---|---|---|
| 0 | 0.0145 | 0.0027 (45) | 0.143 | −0.61(15) | +0.32(37) |
| 0.5 | 0.0096 | 0.0029 (45) | 0.141 | −0.44(14) | +0.23(34) |
| 1 | 0.0152 | 0.0045 (325) | 0.136 | −0.44(7) | +0.11(15) / +0.27(15) |
| 2 | 0.0365 | 0.0151 (325) | 0.118 | −0.32(4) | +0.59(9) / +0.51(9) |

(6,8) columns: 6⁴ stage 1 / stage-1b new trajectories against the pooled 8⁴. The h = 0 and 0.5 rows rest on the
45-measurement stage-1 8⁴ chains (recorded, not asserted). 8⁴ members at h = 2: stage 1 0.0116, stage-1b new 0.0142,
replicas 0.0186 / 0.0149.
Recorded with the stage-1c reader of K5.11: R_peak = [χ_L(0)/χ_L(p_min)]/free at h = 2 rises from 0.61(5) on 6⁴ to
0.88(8) on 8⁴ (2.8σ); both are below 1.

**Method.** `masspairing.analysis.complete_pair`. Coverage from `TasteProjector.n_boundary`. χ_L = ts_chi_L_sum of the
y = 3.0, κ = −0.01 chains: 4⁴ and 4³×8 from stage 0 (trajectories ≥ 150, i.e. the first 75 of 300 measurements cut),
8⁴ pooled over stage 1 (≥ 100), the stage-1b new trajectories (≥ 1000) and the stage-1c replicas rA, rB (≥ 100), 6⁴ from
stage 1 and stage 1b; free values of the same observable on the same box and h. Errors: 10 blocks per chain (10 per
member for the pooled series), raised to σ_naive √(2τ_int) where larger — a simpler error model than the stage-1b/1c
reader; on (6⁴, 8⁴) it gives +0.590(90) / +0.513(85) against K5.11's 0.600(104).

**Controls.** The (6⁴, 8⁴) pair with the same reader reproduces the positive K5.8/K5.11 value (the reader sees a growth
where the incomplete box gives one). Sabotage: the pooled 8⁴ χ_L rescaled to the SSB-scale growth χ_L(4⁴)·V₈/V₄ gives
e ≈ 1 and e − e_free(4,8) = +0.86 / +0.88 at h = 1 / 2, and the falling-with-V clause fails on it.

**Caveats.** 4⁴ is a small box (the SMG gap ≈ 0.7 makes ξ ≈ 1.4, so 4⁴ is about 3ξ across); the 4³×8 point and the
monotone three-box series support the trend; a 12⁴ point at (3.0, 2) is the clean test. The 8⁴ data were not re-analysed
on the 6⁴ mode set (this needs a dense re-measurement of stored configurations). The exponents are finite-size ratios
at L ≤ 8, not scaling dimensions.

**Data.** `data/derived/n1stage/S0/L4/L4_y3_*_bcaaaa.npz`, `data/derived/n1stage/S0/L4x8/L4x8_y3_*_h2_*.npz`,
`data/derived/n1stage/{S1,S1b}/{L8,L6}/*_y3_*.npz`, `data/derived/n1stage/S1c/L8_h{1,2}_r{A,B}/*.npz`, free baselines
`data/derived/free/free_L4_aaaa.json`, `free_L8_aaaa.json`, `free_L4x8_aaaa.json`, `free_baselines.json`; table
`results/K5_complete_pair.csv`, coverage `results/K5_taste_coverage.csv` (`scripts/table_k5_complete_pair.py`).

**Check.** `python claims/K5.13/check.py` (≈ 10 s).

**Provenance.** Notebook claim C217 (unhiggsed_notebook @ 698af9a).
