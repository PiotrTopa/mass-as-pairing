# K5.3 — stage 0: the pre-registered gates G1–G4 applied as written, with controls of the gate logic

**Statement.** On the 26 stage-0 chains of model N1 (4⁴ aaaa × y ∈ {2.0, 2.41, 3.0} × h ∈ {0, 0.1, 0.2, 0.5, 1, 2};
4³×8 aaaa × y ∈ {2.41, 3.0} × h ∈ {0, 0.5, 2}; one 4⁴ pppa pair at y = 2.41, h ∈ {0, 0.5}), read with the
pre-registered reader against the free baselines at the same L, L_t, bc, h, the gates give: G1 (readability) PASS;
G2 (SYM control at y = 2.0) passes at h ≤ 1 and fails at h = 2 on the light ratio alone (a 2 % shift); G3 (heavy half
responds) fails in two distinct ways — at h = 0.5 nothing is resolved (the free shift itself is below the resolution and
negative on 4³×8), and at y = 3.0 the elementary heavy channel does not respond at any h; G4 (recorded) reproduces the
composite ordering SYM ≪ 1 ≪ SMG, with P_c on the elementary side.

**Numbers.** (K5.3 check output)

| gate | as pre-registered | readout |
|---|---|---|
| G1 | 4³×8: rel. error of m_L, m_R < 15 %, of χ_L < 20 % at every (y, h) | PASS: max 1.0 %, 1.0 %, 15.3 % |
| G2 | y = 2.0: r_L within 2σ of the free ratio; χ_L/free ∈ [0.5, 2]; G_L(p_min)/free ∈ [0.7, 1.3] | passes at h ≤ 1 (max \|pull\| 1.61σ); fails at h = 2: r_L = 1.020(6) vs 1.000, +3.35σ; χ_L/free 0.90–0.93, G_L/free 0.94–0.96 at every h |
| G3 | m_R(h) − m_R(0) > 0 at > 3σ for h ≥ 0.5 at every y; φ_R/h > 0 at > 5σ | fails: (i) h = 0.5: max +2.2σ, free shifts +0.0078 (4⁴) / −0.0073 (4³×8); (ii) y = 3.0: shift −13.2 … −0.2σ (4⁴: −3.2σ / −13.2σ at h = 1 / 2 where the free shift is +0.03 / +0.12), φ_R/h ≤ 0.025 × free, > 5σ only at h = 2; passes at y = 2.0/2.41, h ≥ 1 (≥ 3.6σ; φ_R/h = 0.74–0.93 × free) |
| G4 (recorded) | φ_T_R/φ_R: < 0.02 at y = 2.0, ≫ 1 at y = 3.0 | ≤ 0.0015 at y = 2.0; 2.6–5.8 at y = 3.0 where φ_R is resolved (> 2σ, 4 rows), φ_T_R/h > 50 × free at every h; 0.0045–0.0066 at P_c |

- 26 chains at 600 trajectories, acceptance 0.96–1.00; τ_int of the m_L, χ_L, G_L(p_min), φ_R series ≤ 0.93
  measurements (20-block errors valid).
- G2 rows: r_L = 0.9999(53), 1.0016(54), 0.9999(58), 1.0099(61), 1.0203(60) at h = 0.1, 0.2, 0.5, 1, 2 (free ratio 1).

**Method.** Reader (`masspairing.analysis.stage0`): m_L, m_R = ln C(t*)/C(t*+1) of the light / heavy pair-channel
time-slice correlators at t* = L_t/2 − 1; χ_L = V⟨φ_L²⟩; G_L(p_min); φ_R = Σ_f φ_f; φ_T_R; first min(100, n/4) = 75
measurements cut; 20-block errors; r_L = [m_L(h)/m_L(0)]/[m_L^free(h)/m_L^free(0)]. Gates `gate1`–`gate4`.

**Pre-registration and deviations.** The gates and the reader were fixed before the data were read (notebook commit
b60c1d1) and are applied as written, including their failures. G3 was written for a heavy half that responds as in the
free theory; it is ill-posed at h = 0.5 (below the resolution; the free theory fails it on 4³×8) and at y = 3.0, where
the elementary channel is screened (K4.2). Stage 0 is not a verdict.

**Controls.** An injected free-like SYM series passes G2 (max |pull| 1.47σ); an injected 1/(1 + 3h) fall of m_L at
y = 2.0 is flagged by G2 at every h (pulls −48 … −178σ); a mis-signed h (mirrored m_R shift, negative φ_R) is flagged by G3
on every y = 2.0/2.41 row with h ≥ 1; errors inflated ×20 are flagged by G1; the G4 ordering fails with the y labels
2.0 ↔ 3.0 swapped.

**Caveats.** L = 4 only; the t* log-ratio is a cosh-contaminated gap proxy; G3/G4 at y = 3.0 rest on few resolved rows.

**Data.** `data/derived/n1stage/S0/*/*.npz`; `data/derived/free/free_L4_aaaa.json`, `free_L4x8_aaaa.json`,
`free_L4_pppa.json`. Raw archive: `results/xi_scan/K_N1_S0/{L4,L4x8,L4pppa}/*.npz`.

**Check.** `python claims/K5.3/check.py` (≈ 2 s): recomputes every gate from the derived chains, prints the numbers
above, runs the five controls.

**Provenance.** Notebook claim C190 (unhiggsed_notebook @ 3c4a02c).
