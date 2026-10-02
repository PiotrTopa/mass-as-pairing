# K5.4 — stage 1: data integrity, cadence, the order's kills (silent), recorded instrument checks, error model and free baselines

**Statement.** The 26 stage-1 chains of model N1 (8⁴ aaaa, 6³×12 aaaa and 6⁴ aaaa × y ∈ {2.41, 3.0} × h ∈ {0, 0.5, 1, 2};
6⁴ pppa at y = 2.41, h ∈ {0, 0.5}; 1000 trajectories after 100 of thermalisation; exact dense measurements with stored
configurations) are complete and consistent: one code revision and source pattern, the thinned measurement cadence as
recorded, no repeated trajectory index and no repeated configuration, acceptable acceptance; the order's kills are
silent; the recorded instrument checks hold except the h = 0.5 asymmetry check on 8⁴ and 6⁴ (where the free asymmetry is
itself small); the error model's inflation is modest; the free baselines reproduce the free-theory facts, and the free
6³×12 light correlator has no decaying midpoint, so the free-ratio normalisation of the t* log-ratio is void there.

**Numbers.** (K5.4 check output)

1. Integrity: 26 chains, code revision 29f33bd, pattern C_χ (0,3), dual sign +1; 24 aaaa + 2 pppa. Cadence after the cut:
   8⁴ every 20 trajectories (45 measurements), 6³×12 every 16 (56), 6⁴ every 4 (225). No repeated trajectory index; the
   251-row 6⁴ (3.0, h = 0) chain has 251 distinct indices and 251 distinct stored configurations (a resume re-phased the
   cadence 44, 48, 50, 54, before the cut). Acceptance per 50-trajectory window ≥ 0.60 (≥ 0.84 on aaaa).
2. Kills: at h = 0 (7 chains) CL_t = CR_t within 4σ at every t (max pull 2.79) and |m_L − m_R| ≤ 0.89σ; A(h) < 0 beyond 3σ
   nowhere (min pull −1.5).
3. Recorded checks: y = 2.41: φ_R/h > 0 on all 9 rows (≥ 67σ); A(h) > 0 for h ≥ 1 (≥ 15.6σ); at h = 0.5 A passes 3σ on
   6³×12 (3.4σ) and fails as written on 8⁴ (2.1σ) and 6⁴ (3.0σ) (free A(0.5) = 0.019–0.034). y = 3.0: φ_T_R/h ≠ 0
   (≥ 92σ) and φ_R/h ≥ 0 within 2σ on all 9 rows; φ_R/h ≤ 0.011 × free and |A| ≤ 0.009 on 6⁴, 6³×12, 8⁴.
4. Error model: τ_int of every primary series ≤ 2.9 measurements; τ_B/Δ ≤ 1.04 on aaaa, 6.1 on 6⁴ pppa h = 0 (τ_B = 24
   trajectories); inflation σ/σ_block ≤ 1.93 on aaaa; half-split pulls > 3σ on 10 of 260 series (6 on the pppa h = 0
   chain), recorded, nothing re-estimated.
5. Free baselines: |G_L^free(p_min) − 1| ≤ 1.8e-15 (6⁴), 6.7e-16 (8⁴), 2.3e-3 (6³×12), 3.3e-3 (6⁴ pppa; boundary modes);
   free light shift at t*: ≤ 0.0088 at h = 0.5, ≤ 0.113 at h = 2 (6⁴, 8⁴); free 6³×12 light log-ratio at t* = 5:
   −0.251, −0.244, −0.222, −0.140 at h = 0, 0.5, 1, 2.

**Method.** `masspairing.analysis.stage1.read_chain`: cut at trajectory index 100; σ = max{σ_block(10 or 20 blocks),
σ_naive √(2τ_eff)}, τ_eff = max(τ_int(series), τ_B/Δ), τ_B the Madras–Sokal time of the per-trajectory |Σ_stag| and
σ² series, Δ the measurement spacing; duplicates detected by trajectory index and by the md5 digest of every stored
configuration (`cfg_digest`, written by `scripts/derive_n1stage.py`).

**Pre-registration and deviations.** Reader, cut and error model were fixed before any stage-1 observable was read
(notebook commit eaa3692). The measurement cadence was thinned relative to the order (every 20 / 16 / 4 trajectories).

**Caveats.** 45 measurements per 8⁴ chain; the pppa h = 0 chain is slow (K4.4).

**Data.** `data/derived/n1stage/S1/*/*.npz` (26 chains with `cfg_digest`); `data/derived/free/free_L8_aaaa.json`,
`free_L6x12_aaaa.json`, `free_baselines.json`. Raw archive: `results/xi_scan/K_N1_S1/*/*.npz`; the four chains extended
later in place (8⁴ and 6⁴ at y = 3.0, h = 1, 2) are read from their pre-extension copies
`results/xi_scan/K_N1_S1b/pre/{L8,L6}/*.npz`.

**Check.** `python claims/K5.4/check.py` (≈ 2 s). `tests/equivalence/test_n1stage.py` verifies that the stage-1 rows
are bit-identical to the notebook's frozen output.

**Provenance.** Notebook claim C193 (unhiggsed_notebook @ 3c4a02c).
