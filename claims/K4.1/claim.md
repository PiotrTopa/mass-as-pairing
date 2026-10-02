# K4.1 — Composite takeover across SYM → P_c → SMG (nodal source on one flavour, 4⁴/6⁴)

**Statement.** A charge-2, ℤ₄-odd pairing source on one flavour — the same-parity nodal pattern C₀ on flavour 4,
h C₀ ⊗ diag(0, 0, 0, 1) — is answered by the elementary field χ⁴ in the symmetric (SYM) phase and by its ε-vertex
partner Ψ⁴ = χ¹χ²χ³ in the SMG phase. In SYM the composite amplitude is below 2 % of the elementary one; in SMG the
elementary response per unit h has collapsed by ≈ 50× and the composite carries the response, linearly in h and
volume-stably. At P_c, on the production boundary conditions, the ratio is intermediate. The unsourced flavours carry
no elementary amplitude beyond the O(κ) level.

**Numbers.** Per chain: φ_heavy = ⟨O⁽⁴⁾⟩/V (elementary amplitude of the sourced flavour in the source direction),
φ_light = the same for flavours 1–3 (mean), φ_T = (1/V) ½ Σ_{xz} C₀[x,z] ⟨T(x) T(z)⟩ with T = χ¹χ²χ³ (the same-direction
amplitude of the composite partner, charge 6 ≡ 2). Ranges over the rows (errors per row in the check output and in
`results/K4_takeover.csv`):

| phase | rows | φ_heavy/h | φ_T/h | \|φ_T\|/φ_heavy |
|---|---|---|---|---|
| SYM, y = 2.0 | 4⁴ h = 0.05–0.5 (5), 6⁴ h = 0.3 | 0.70 → 0.36 (falls with h); 0.567(1) at 6⁴ | −0.0039 … −0.0015; −0.0016 at 6⁴ | 0.003–0.006 |
| P_c, y = 2.41 | 4⁴ h = 0.02–0.5 (8), 6⁴ h = 0.1–0.3 (3) | 0.24–0.41 | −0.012 … −0.047 (4⁴); −0.013 … −0.026 (6⁴) | 0.032–0.166 (h ≤ 0.3); 0.049–0.068 (h = 0.5) |
| SMG, y = 3.0 | 4⁴ h = 0.05–0.5 (5), 6⁴ h = 0.3 | 0.001–0.015 | −0.0877 (4⁴ mean), −0.0896(5) (6⁴) | 5.8–121 |

1. SYM: |φ_T|/φ_heavy < 0.02 on every row (max 0.0055).
2. SMG: |φ_T|/φ_heavy > 4 and φ_heavy/h < 0.05 on every row (min ratio 5.8, max φ_heavy/h 0.0151).
3. SMG: φ_T/h is constant within 3.4 % over h ∈ [0.05, 0.5] at 4⁴ (linear response) and 4⁴ (h = 0.2, 0.5) agrees with
   6⁴ (h = 0.3) within 2.3 %.
4. P_c: 0.032 ≤ |φ_T|/φ_heavy ≤ 0.166 on every row with h ≤ 0.3. φ_T/h does not grow with the volume: −0.040 … −0.047
   at 4⁴ (h ≤ 0.1), −0.026 at 6⁴ (h = 0.1).
5. Unsourced flavours: |φ_light/h| < 0.02 + 3σ on all 22 rows with h ≥ 0.05 (max 0.014).

**Method.** Stored chains of the ε model at κ = −0.01, g₁₀ = 0, production boundary conditions (periodic space,
antiperiodic time), RHMC on |Pf M(h)| with the source C₀ on flavour 4 (I.8), exact (dense) fermion estimators, the
Pfaffian sign recorded per measurement (I.9) and used as a reweighting factor (⟨s⟩ = 0.73, 0.84 on the two 4⁴ P_c
h = 0.5 chains and 0.88 on 6⁴ P_c h = 0.3; 1 elsewhere). 4⁴: y ∈ {2.0, 2.41, 3.0} × h ∈ {0.02 … 0.5}, 18 chains, several
(y, h) points with two independent runs (labels a, b); 6⁴: y ∈ {2.0, 2.41, 3.0} at h = 0.3 and y = 2.41 at h = 0.1, 0.2.
First quarter of each chain discarded; sign-weighted means; errors from 8 equal blocks.

**Pre-registration and deviations.** The criteria (a)–(e) with thresholds 0.02, 4, 0.05, 6 %, (0.02, 0.2) were fixed
before the 6⁴ rows were read.

**Controls.** The SMG ordering (ratio > 4) never occurs in a SYM row and the SYM ordering (ratio < 0.02) never in an SMG
row: both amplitudes come from the same dense propagator, so the ordering is a property of the phase, not of the
estimator.

**Caveats.**
- The source is the nodal pattern C₀, a pairing field on the UV modes, not a Majorana mass of the corner fermions
  (K2.5). The claim is about which channel answers a ℤ₄-odd perturbation; the taste-chiral Majorana mass is K4.2–K4.4.
- The intermediate P_c ratio of item 4 is a feature of the production boundary conditions at 4⁴/6⁴: with the
  taste-chiral mass on all-antiperiodic boxes the P_c ratio is 0.004–0.012, on the elementary side, and 0.038–0.10 on
  the production bc (K4.2, K4.4). K4 at P_c is quoted from K4.3/K4.4 with the boundary condition stated, not from item 4.
- Only 4⁴ and 6⁴; the SMG rows are deep in the phase (y = 3.0); T is the SU(3)-singlet product of the three unsourced
  flavours.

**Data.** `data/derived/n1inst/takeover/*.npz` (23 files: ts_phi_T, ts_phi_f, ts_pf_sign; `scripts/derive_n1inst.py
takeover`) from the archive chains `results/laneS/chains_L4/`, `results/xi_scan/S_prod/L4/`, `results/xi_scan/S_prod/L6/`,
`results/laneS/pilot/L6/` (`*_f0001.npz`, listed in `data/MANIFEST.tsv`). Table `results/K4_takeover.csv`
(`scripts/table_n1inst_takeover.py`), figure `figures/k4_takeover.pdf` (`scripts/fig_n1inst_takeover.py`).

**Check.** `python claims/K4.1/check.py` (≈ 5 s): prints the per-chain table and asserts items 1–5 and the control.

**Provenance.** Notebook claim C150 (unhiggsed_notebook @ 3c4a02c).
