# K4.3 — the composite takeover with C_χ on 8⁴, 6⁴ and 6³×12: the elementary heavy channel is screened in SMG (≤ 1.1 % of free), the composite is linear in h and lattice-independent to 2 %

**Statement.** On the stage-1 chains of model N1 (all-antiperiodic 8⁴, 6⁴, 6³×12; y ∈ {2.41, 3.0}; h ∈ {0.5, 1, 2}):
(i) in the SMG phase (y = 3.0) the composite/elementary ratio ρ = |φ_T_R/φ_R| exceeds 4 on all nine rows (lower bounds
where φ_R is unresolved) and the elementary amplitude is φ_R/h ≤ 0.011 × free, growing with h (a second-order response);
(ii) at P_c (y = 2.41) on all-antiperiodic boxes ρ = 0.004–0.012 (elementary side);
(iii) the SMG composite amplitude per unit h, φ_T_R/h ≈ −0.00195, is h-independent within 0.9 % on 8⁴ and equal between
8⁴, 6⁴ and 6³×12 within 1.9 % (≥ 44 × free on 6⁴, 6³×12; stage 0 at L = 4 reads 8–10 % lower);
(iv) the ordering does not reverse with the volume;
(v) at P_c the composite falls 2.1–3.1× from h = 0.5 to 2 while φ_R/h = 0.59–0.84 × free, and the on-configuration
asymmetry exceeds free (8⁴, h = 2: A = 0.268(2) vs 0.180).

**Numbers.** (K4.3 check output)

| lattice | y | h | φ_R/h (× free) | φ_T_R/h | ρ |
|---|---|---|---|---|---|
| 8⁴ | 2.41 | 0.5 / 1 / 2 | 0.59 / 0.72 / 0.84 | −0.000359(25) / −0.000214(13) / −0.000117(1) | 0.012 / 0.007 / 0.004 |
| 8⁴ | 3.0 | 0.5 / 1 / 2 | 0.001 / 0.004 / 0.011 | −0.001952(14) / −0.001969(16) / −0.001955(15) | > 12.9 / 11.5(3.2) / 5.1(6) |
| 6⁴ | 3.0 | 0.5 / 1 / 2 | −0.001 / 0.005 / 0.011 | −0.001945(12) / −0.001956(10) / −0.001938(10) | > 18 / 9.4(1.8) / 4.9(4) |
| 6³×12 | 3.0 | 0.5 / 1 / 2 | 0.002 / 0.004 / 0.010 | −0.001949(21) / −0.001932(16) / −0.001945(16) | > 9.7 / 11.9(5.7) / 5.4(6) |

Min ρ in SMG 4.9; P_c ρ range 0.004–0.012 on 8⁴, 6⁴, 6³×12; min φ_T_R/free (6⁴, 6³×12, SMG) 44.

**Method.** Stage-1 reader (`masspairing.analysis.stage1.read_chain`): measurements at trajectory ≥ 100; error
σ = max{σ_block, σ_naive √(2τ_eff)} with τ_eff = max(τ_int, τ_B/Δ) (τ_B of the per-trajectory bosonic series); φ_R =
Σ_f φ_f; ρ resolved when |φ_R| > 2σ, else the lower bound |φ_T_R|/(|φ_R| + σ). Criteria `stage1.k4`: SMG ρ > 4 and
φ_R/h ≤ 0.05 × free; P_c ρ < 0.2; φ_T_R/h equal across h and lattices within max(10 %, 2σ); the y-label swap must fail.

**Pre-registration and deviations.** The composite criteria were fixed before the stage-1 data were read (notebook
commit eaa3692). The 8⁴ free baseline carries no composite, so φ_T_R/free is quoted on 6⁴ and 6³×12 only.

**Controls.** With the y labels swapped (2.41 ↔ 3.0) the criteria (i)/(ii) fail at 8⁴.

**Caveats.** ρ at h = 0.5 in SMG is a lower bound (φ_R unresolved). At P_c the elementary-side reading holds on
all-antiperiodic boxes; the production-bc box has a 4× larger ratio (K4.4). The screening is a small-quasiparticle-weight
statement on L ≤ 8.

**Data.** `data/derived/n1stage/S1/{L8,L6x12,L6,L6pppa}/*.npz` (26 chains); `data/derived/free/free_L8_aaaa.json`,
`free_L6x12_aaaa.json`, `free_baselines.json`. Raw archive: `results/xi_scan/K_N1_S1/*/*.npz` (the four chains later
extended in place are read from their pre-extension copies `results/xi_scan/K_N1_S1b/pre/*/*.npz`).

**Check.** `python claims/K4.3/check.py` (≈ 2 s). The reader is bit-identical to the notebook's frozen stage-1 rows
(`tests/equivalence/test_n1stage.py`).

**Provenance.** Notebook claim C195 (unhiggsed_notebook @ 3c4a02c).
