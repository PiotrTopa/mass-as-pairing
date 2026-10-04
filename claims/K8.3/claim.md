# K8.3 — small-h onset at P_c on 6⁴: undecided, with a power-limit control

**DRAFT — prose pending analysis 2026-10-05.**

**Statement.** The pre-registered onset fit F1, m² − m₀² = (c h^p)² on 6⁴ aaaa at P_c = (2.41, −0.01) with
h ∈ {0.25, 0.5 (pooled replicas), 0.75, 1} and m₀ the stored h = 0 chain, gives p = 1.14(37), χ²/dof 0.17,
I = p ± 2σ = [0.41, 1.87]. I meets both the quadratic band [1.6, 2.4] and the linear band [0.7, 1.3], and σ_p > 0.3:
**UNDECIDED**. Power-limit control: a synthetic quadratic onset with the real errors also returns UNDECIDED
(p = 2.00(48)); the linear synthetic returns "linear" (1.00(20)); the saturating one is not called quadratic
(0.44(2)).

**Numbers.** (K8.3 check output)

| h | m(t = 2) on CL_t | origin |
|---|---|---|
| 0 | 0.3919(84) | stage 1 |
| 0.25 | 0.3921(72) | stage 1c, pooled |
| 0.5 | 0.4041(52) | stage 1c, pooled (stored stage 1: 0.4066(87), −0.26σ) |
| 0.75 | 0.4270(57) | stage 1c |
| 1 | 0.4527(43) | stage 1 |

- Labelled fits: F1′ 1.128(362), F2 1.38(85), F1f ≡ F1; all UNDECIDED.
- Monotonicity pulls 0 → 0.25 → 0.5 → 0.75 → 1: 0.0σ, 1.4σ, 3.0σ, 3.6σ.
- Local quadrature exponents 0.5 → 3: 1.34, 1.01, 0.91, 0.80, 0.53.
- Recorded (post hoc): Gaussian-MC σ_p = 0.44; P(p ≥ 1.6) = 0.16; P(0.7 ≤ p ≤ 1.3) = 0.60.

**Data.** `data/derived/n1stage/S1c/{L6,L6_r2}/*.npz`, `data/derived/n1stage/{S1,S1b}/L6/`,
`data/derived/free/free_L6_aaaa_h0.25_0.5_0.75.json`. Raw archive: `results/xi_scan/K_N1_S1c/{L6,L6_r2}/`,
`results/laneK1/free_L6_aaaa_h0.25_0.5_0.75.json` (bundle `mass-as-pairing-data_n1-stage1c.tar.gz`).

**Check.** `python claims/K8.3/check.py`.

**Provenance.** Notebook claim C205 (unhiggsed_notebook @ fd0c934).
