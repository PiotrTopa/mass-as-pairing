# K5.9 — stage 1c integrity and replica consistency

**DRAFT — prose pending analysis 2026-10-05.**

**Statement.** The nine stage-1c chains are intact (md5 as delivered, seeds and h as ordered, 1000 trajectories, no
duplicates, 90 (8⁴) / 225 (6⁴) rows after the cut at trajectory 100, acceptance ≥ 0.5, A(h) never below −3σ). At
y = 3.0 on 8⁴ the two independent replicas per h agree with each other and with the stage-1b extension: CONSISTENT at
h = 2 and h = 1 (3-member χ², every p ≥ 0.16). With the stored stage-1 stretch (trajectories 100–999) of the 8⁴ h = 1
chain as a fourth member the midpoint cosh mass fails (p = 0.005, labelled): the drift recorded in K5.7 lies in that
stretch. The 6⁴ P_c replicas at h = 0.25, 0.5 agree with the first chains (p ≥ 0.07), so the pooled points enter K8.3.

**Numbers.** (K5.9 check output)

| test | value |
|---|---|
| smallest p: 3-member; rA vs rB; pooled vs stage 1b | 0.16; 0.24; 0.22 |
| 4-member with the stored h = 1 stage-1 stretch, m_cosh | p = 0.005; 0.72006(111) vs 0.7160–0.7172 |
| half-split pulls, nine chains | max 2.3 |
| control: 3σ_c shift into rB (largest p) | 6·10⁻⁴ |
| 6⁴ replica test, h = 0.25 / 0.5 (smallest p) | 0.07 |
| sabotage: 3σ_c shift into a 6⁴ replica (largest p) | 3.3·10⁻⁵ |

**Data.** `data/derived/n1stage/S1c/{L8_h1_rA,L8_h1_rB,L8_h2_rA,L8_h2_rB,L6,L6_r2}/*.npz`,
`data/derived/n1stage/{S1,S1b}/`, `data/derived/free/free_L6_aaaa_h0.25_0.5_0.75.json`. Raw archive:
`results/xi_scan/K_N1_S1c/`, `results/laneK1/free_L6_aaaa_h0.25_0.5_0.75.json` (bundle
`mass-as-pairing-data_n1-stage1c.tar.gz`).

**Check.** `python claims/K5.9/check.py`. `tests/equivalence/test_n1stage.py::test_stage1c` verifies bit-identity with
the notebook's frozen stage-1c rows, verdict and sensitivity.

**Provenance.** Notebook claim C203 (unhiggsed_notebook @ fd0c934).
