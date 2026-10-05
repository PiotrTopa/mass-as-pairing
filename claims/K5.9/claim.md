# K5.9 — stage 1c integrity and replica consistency

**Statement.** The nine stage-1c chains are intact: md5 as delivered, seeds and h as ordered, 1000 trajectories each, no
duplicate trajectory index, 90 (8⁴) / 225 (6⁴) rows after the cut at trajectory 100, acceptance ≥ 0.5 in every
50-trajectory window, the light–heavy asymmetry A(h) never below −3σ. At y = 3.0 on 8⁴ the two independent replicas per
h agree with each other and with the stage-1b extension of the stored chain: **CONSISTENT** at h = 2 and h = 1 (three
members, midpoint cosh mass, χ_L and G_L(p_min); every p ≥ 0.16). With the stored stage-1 stretch (trajectories 100–999)
of the 8⁴ h = 1 chain as a fourth member the midpoint cosh mass fails (p = 0.005; labelled, not a criterion): the
per-mille drift recorded in K5.7 lies in that stretch. On 6⁴ at P_c the replicas at h = 0.25 and 0.5 agree with the first
chains (p ≥ 0.07), so the pooled points enter the onset fit of K8.3.

**Numbers.** (K5.9 check output)

| test | value |
|---|---|
| min acceptance (50-window), 8⁴ / 6⁴ | 0.96–0.98 / 0.90–0.94 |
| smallest p: 3-member; rA vs rB; pooled vs stage 1b | 0.16; 0.24; 0.22 |
| h = 2 midpoint cosh mass rA / rB / stage 1b | 0.71394(94) / 0.71300(84) / 0.71238(88) |
| 4-member with the stored h = 1 stage-1 stretch, m_cosh | p = 0.005; 0.72006(111) vs 0.7160–0.7172 |
| the same 4-member test at h = 2 | p = 0.11 |
| half-split pulls, nine chains | max 2.3 |
| control: 3σ_c shift into rB (largest p) | 6·10⁻⁴ |
| 6⁴ replica test, h = 0.25 / 0.5 (smallest p) | 0.07 |
| sabotage: 3σ_c shift into a 6⁴ replica (largest p) | 3.3·10⁻⁵ |

**Method.** `masspairing.analysis.stage1c`: the stage-1b reader per chain (cut at trajectory 100, error model C0);
consistency by χ² over the members (dof = members − 1) on the midpoint cosh mass, ⟨χ_L⟩ and ⟨G_L(p_min)⟩; pooled values
with each member's jackknife × its own inflation (conservative by construction).

**Pre-registration and deviations.** The integrity list, the consistency test (p > 0.01 on every observable) and the
controls were fixed before the stage-1c chains started (notebook order K-N1-S1c, commit 56a8e32). The four-member test
with the stage-1 stretch is post hoc and labelled.

**Controls.** A 3σ_c shift injected into rB is flagged in the three-member and the pairwise test at both h; a 3σ_c shift
of a 6⁴ replica's cosh mass is flagged at h = 0.25 and 0.5. The one-member pool reproduces the single-chain reader.

**Caveats.** 90 measurements per 8⁴ chain at cadence 10; the error inflation is driven by the bosonic τ_B.

**Data.** `data/derived/n1stage/S1c/{L8_h1_rA,L8_h1_rB,L8_h2_rA,L8_h2_rB,L6,L6_r2}/*.npz`,
`data/derived/n1stage/{S1,S1b}/`, `data/derived/free/free_L6_aaaa_h0.25_0.5_0.75.json`. Raw archive:
`results/xi_scan/K_N1_S1c/`, `results/laneK1/free_L6_aaaa_h0.25_0.5_0.75.json` (bundle
`mass-as-pairing-data_n1-stage1c.tar.gz`).

**Check.** `python claims/K5.9/check.py` (≈ 5 s). `tests/equivalence/test_n1stage.py::test_stage1c` verifies bit-identity
with the notebook's frozen stage-1c rows, verdict and sensitivity.

**Provenance.** Notebook claim C203 (unhiggsed_notebook @ fd0c934).
