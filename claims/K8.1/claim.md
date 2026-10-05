# K8.1 — tier 1: a partner Majorana mass can only lower a light gap (lemma); at P_c the explicit M_R on half of the generation suppresses the ε/σ channel to the size of its SMG value by h = 2 and raises the light pair-channel effective mass, with no SSB

**Statement.**
1. Lemma (partner algebra). For the two-state matrix [[0, Δ], [Δ, M_p]] — a fermion Dirac-paired with strength Δ to a
   partner that carries a Majorana mass M_p — the light eigen-gap is λ = √(Δ² + M_p²/4) − |M_p|/2 ≤ Δ, strictly decreasing
   in |M_p| for every Δ > 0. A Majorana mass on the composite partner can only lower the light gap (the seesaw direction);
   a rising light gap is not that mechanism.
2. Stored stage-1 data at P_c (y = 2.41), model N1 with h C_χ on the heavy doublet, all-antiperiodic 8⁴, 6⁴ and 6³×12:
   (i) the ε/σ channel collapses: S(π) falls monotonically with h on all three lattices (no step rises beyond 1σ, the
   h = 1 → 2 step > 3σ, the total h = 0 → 2 fall 9–12σ) and at h = 2 equals the SMG (y = 3.0, h = 0) value of the same
   lattice within 15 %; |Σ_stag| likewise; at y = 3.0 neither moves with h (within 15 %);
   (ii) the light pair-channel gap rises: the midpoint cosh mass of the light correlator CL_t rises monotonically with h
   on all three lattices (8⁴ h = 1 → 2 by 17σ), while at y = 3.0 it is h-stable within 10 %;
   (iii) no SSB of the light doublet at P_c: e − e_free on (6⁴, 8⁴) is −0.09 … +0.05 at every h (SSB would need ≈ +0.5), and
   χ_L(p_min)/χ_L(0) equals its free value within 17 % at every h and lattice;
   (iv) quench: on 4 stored h = 0 configurations of the 6⁴ P_c chain, putting h = 2 into the operator at fixed σ raises the
   light midpoint cosh mass on every configuration — part of the rise is present at fixed σ, the rest comes with the
   reorganised ensemble of (i).

**Numbers.** (K8.1 check output; cut at trajectory 100, 10 blocks, delete-one-block jackknife for the cosh mass)

| lattice | S(π) at P_c, h = 0 → 0.5 → 1 → 2 | S(π) SMG (h = 0) | \|Σ_stag\| P_c h = 0 → 2 vs SMG | light midpoint cosh mass, P_c, h = 0 / 0.5 / 1 / 2 | y = 3.0, h = 0 → 2 |
|---|---|---|---|---|---|
| 8⁴ (t = 3) | 25.99(1.91) → 14.51 → 6.39 → 2.91(15) | 3.02(10) | 0.0738 → 0.0244 vs 0.0251 | 0.197(9) / 0.211(13) / 0.332(16) / 0.614(5) | 0.716 → 0.712 (−0.5 %) |
| 6⁴ (t = 2) | 10.20(83) → 8.90 → 5.53 → 2.87(15) | 3.23(9) | 0.0824 → 0.0433 vs 0.0459 | 0.392(4) / 0.407(5) / 0.453(4) / 0.569(1) | 0.763 → 0.712 (−6.7 %) |
| 6³×12 (t = 5) | 12.12(82) → 8.65 → 5.15 → 2.75(7) | 2.95(13) | 0.0631 → 0.0299 vs 0.0310 | 0.327(22) / 0.359(11) / 0.416(4) / 0.475(1) | 0.423 → 0.458 (+8.2 %) |

- Error model of this table: plain 10-block errors without autocorrelation inflation (the tier-1 estimator as the check
  computes it). The programme's error model C0 (σ = max{σ_block(20), σ_naive √(2τ_eff)}, used from stage 1b on: K8.2,
  `results/K8_gap_vs_h.csv`) gives larger errors on the same central values — S(π) SMG 3.02(12) (8⁴), 3.23(14) (6⁴),
  2.95(14) (6³×12); 8⁴ P_c h = 0 25.99(2.34), h = 2 2.91(12) — and these are the errors to quote. With C0 errors the
  S(π) falls h = 0 → 2 are 9.9σ / 8.7σ / 11.0σ (8⁴ / 6⁴ / 6³×12), the h = 1 → 2 steps 7.6σ / 6.4σ / 7.5σ; no clause changes.
- S(π) step pulls (falls) h = 0→0.5, 0.5→1, 1→2: 8⁴ 4.6, 4.9, 7.4 (total 12σ); 6⁴ 1.4, 6.1, 7.8 (9σ); 6³×12 3.0, 4.0, 9.4 (11σ).
- e − e_free at h = 0, 0.5, 1, 2: −0.086, −0.049, +0.021, +0.053; max |χ_L(p_min)/χ_L(0) ÷ free − 1| = 17.0 %.
- Quench: per-configuration light midpoint cosh mass h = 0: 0.456, 0.449, 0.398, 0.422; h = 2: 0.520, 0.597, 0.534, 0.523;
  paired Δm = +0.112 ± 0.019 (5.9σ), all four positive; re-measurement at h = 0 reproduces the stored CL_t to ≤ 1e-16.

**Method.** Lemma: closed form against the eigenvalues of 2000 random (Δ, M_p) pairs and the sign of dλ/dM_p.
Stored data: `masspairing.analysis.stage1.sigma_channel_rows` (per-trajectory S(π), |Σ_stag| and the light pair-channel
correlator after the cut). Quench: dense N1 measure (`masspairing.measure.N1Measure`, exact) of 4 stored configurations
(every 56th after the cut) with h = 0 and h = 2 in the operator.

**Pre-registration and deviations.** The h-scaling of the rise and of the σ channel was pre-registered separately (K8.2);
it is not claimed here. Thresholds of (i)/(ii) were set after a first look at the data: "every S(π) step > 3σ" became the
three-part clause above (the 6⁴ first step is 1.4σ: 6⁴ is the least critical box at h = 0), and the y = 3.0 h-stability
bound is 10 % (6³×12 moves +8.2 %).

**Controls.** Injected seesaw rows m(h) = m(0)/(1 + 3h) fail the monotone-rise clause; the partner-mass series of the lemma
(Δ = 0.4, M_p = 0.6h: 0.400, 0.277, 0.200, 0.121) falls and is rejected by the rise; with the y labels swapped the "P_c"
S(π) series has no > 3σ fall at every step (pulls 0.1, −1.6, 2.4).

**Caveats.** CL_t = Σ G_L² is a positive zero-momentum two-fermion (pair) channel; its midpoint cosh mass is an effective
mass at the stated t with no plateau at L_t ≤ 12 (K5.7 E8), not a single-fermion mass. Three lattices, h ≤ 2 here
(through h = 3 in K8.2). The quench uses 4 configurations on 6⁴; at 8⁴ the fixed-σ fraction of the rise is smaller (K8.2 E3). The 6³×12
light readings carry a free-sized shape-induced light one-point function (K5.7, E9). The lemma's premise (a Dirac mixing Δ
with a Majorana-massive partner) is absent in N1 at tree level (the light one-link masses are symmetry zeros, K5.2): it
excludes one mechanism and says nothing about the measured rise, which is the light half approaching free as the critical
point is detuned (K8.4, K8.5). At h = 3 the ε/σ channel falls 27–31 % below the SMG value (K8.2 T3).

**Data.** `data/derived/n1stage/S1/{L8,L6x12,L6}/*.npz`, free baselines `data/derived/free/free_L8_aaaa.json`,
`free_L6x12_aaaa.json`, `free_baselines.json`; `data/configs/n1stage_quench_L6_y2.41_h0.npz`. Raw archive:
`results/xi_scan/K_N1_S1/*/*.npz` (four chains from their pre-extension copies `results/xi_scan/K_N1_S1b/pre/`).

**Check.** `python claims/K8.1/check.py` (≈ 8 min single-threaded; the dense quench dominates).

**Provenance.** Notebook claim C199 items 1 and 3 (unhiggsed_notebook @ 3c4a02c).
