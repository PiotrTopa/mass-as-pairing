# K5.5 — stage 1 by the pre-registered letter: no verdict at L ≤ 8 for the gap channel, the SSB clause flagged by one row at the 2σ edge, the (a)/(c) clauses with every number

**Statement.** Applying the stage-1 criteria literally to the 26 stage-1 chains:
1. The gap-floor gate fails on the pre-registered proxy at both y: the primary estimator is the t* log-ratio of the light
   pair-channel correlator (the 6³×12 cosh fit does not exist at every h of either y), and m_L(0) on 6³×12 is below 0.3
   → "NO VERDICT at L ≤ 8" for the gap channel. Recorded beside it: the cosh effective mass and the fit read above 0.3;
   the t* log-ratio of a cosh is ≈ m²/2, so the floor as written tests the proxy, not the gap; at y = 3.0 the light
   correlator is not a single cosh on [3, 6].
2. The SSB clause (b) is flagged at y = 3.0 by one row (h = 2) at the 2σ edge, against a free exponent 0.1 below the
   threshold; on that row χ_L is 1.2 % of free and there is no p = 0 peak. Not flagged at y = 2.41 (free-like exponents).
3. (a)/(c): r_L ≥ 0.7 − 2σ everywhere, (c) nowhere; the momentum clause g ≤ 0.8 + 2σ holds at y = 3.0 (g ≈ 0.003 at 8⁴,
   falling with V: a Luttinger-zero-like light block) and fails at y = 2.41 at h = 2 only (by 0.017); e + 2σ < 0.5 holds at
   y = 2.41 and fails at y = 3.0 (σ_e = 0.13–0.45 at 45 measurements). Class readings: y = 3.0 → (b) at L = 8 and 6;
   y = 2.41 → UNDECIDED at both volumes; V-stable in both cases.
4. The controls of the verdict logic have power (below).

**Numbers.** (K5.5 check output; estimator named with every r_L)

1. Gap floor on 6³×12, h = 0: t* log-ratio m_L(0) = 0.066(3) (P_c) / 0.086(1) (SMG) < 0.3; cosh effective mass at t = 5
   0.33(2) / 0.423(3); cosh fit [3, 6] 0.34(2) / 0.523(2). Fit χ²/dof at h = 0, 0.5, 1, 2: y = 2.41 0.001, 0.002, 0.11,
   2.95 (exists at h ≤ 1 only); y = 3.0 9.6–13.4. y = 3.0, h = 0: cosh effective mass 0.489(2) at t = 4 → 0.423(3) at t = 5
   (17σ).
2. e = d ln χ_L/d ln V on (6⁴, 8⁴): y = 3.0: e(2) = 0.786(129), e − 2σ = 0.527 > 0.5; e(0.5) = 0.43(45), e(1) = 0.16(33),
   e(0) = 0.60(44); y = 2.41: e = 0.39, 0.45, 0.44 at h = 0.5, 1, 2 (free 0.44, 0.43, 0.39); free massless e_free = 0.39–0.45.
   Flagged row: χ_L(p_min)/χ_L(0) = 1.16 (8⁴) / 0.72 (6⁴); χ_L/free = 0.0121 (8⁴) / 0.0076 (6⁴).
3. r_L = [m_L(h)/m_L(0)]/[free ratio], t* log-ratio (the primary estimator on every lattice):

| y | lattice | r_L at h = 0.5 / 1 / 2 | g = G_L(p_min)/free |
|---|---|---|---|
| 2.41 | 8⁴ | 1.15(17) / 2.30(23) / 5.84(52) | 0.730(34) (h = 0), 0.770(29), 0.784(15), 0.821(2) |
| 2.41 | 6⁴ | 1.04(4) / 1.24(4) / 1.84(6) | 0.843 at h = 2 |
| 2.41 | 6³×12 | 1.15(6) / 1.48(7) / 2.94(13) (raw t* ratio 1.12, 1.31, 1.64) | — |
| 3.0 | 8⁴ | 0.989(5) / 0.973(4) / 0.879(4) | 0.0028–0.0033 |
| 3.0 | 6⁴ | 0.998(7) / 0.976(6) / 0.884(5) | 0.013–0.015 |
| 3.0 | 6³×12 | 1.03(3) / 1.17(3) / 2.10(5) (raw 1.00, 1.04, 1.17) | 0.0059–0.0070 |

   e + 2σ at y = 2.41: 0.44, 0.47, 0.46.

**Method.** `masspairing.analysis.stage1.verdict`, `class_reading`; criteria of the stage-1 order: r_L with the primary
estimator (the 6³×12 cosh fit over t ∈ [3, 6] if it exists — χ²/dof ≤ 2, m > 2σ — at every h of that y, else the t*
log-ratio); e(h) on (6⁴, 8⁴); g(h) at 8⁴; (a) r_L ≥ 0.7 − 2σ ∀h at 8⁴ and 6³×12, e + 2σ < 0.5 ∀h, g ≤ 0.8 + 2σ;
(b) e − 2σ > 0.5 at some h; (c) r_L + 2σ < 0.3 with g ≥ 0.8 − 2σ; gap-floor gate m_L(0) on 6³×12 ≥ 0.3; V-stability
between 6⁴ and 8⁴.

**Pre-registration and deviations.** The criteria were fixed before the data were read (notebook commits 42417f8 for
the order as edited after stage 0, eaa3692 for the implementation). They are applied as written. Recorded defects of the
letter: the floor tests the t* proxy (≈ m²/2), not the gap; the (b) threshold sits 0.1 above the free exponent (the clause
was written as if the free χ_L were V-independent); the free-ratio normalisation of r_L is void on 6³×12 at t* (no free
decaying midpoint, K5.4), so r_L there is quoted both as written and raw.

**Controls.** S2 (injected SSB: χ_L(8⁴, h > 0.3) × V₈/V₆) → (b) at both y. S1 as written (m_L → m_L/(1 + 3h)) → (c) at 6⁴
P_c only; at 8⁴ the measured rise exceeds the injected fall (r_L(2) → 0.83), so S1 is miscalibrated against these data.
S1″ (r_L := 1/(1 + 3h), g := free, both y) → (c) at both volumes at P_c. Free-null injection (every light number at its
free value) → never (a), never (b). S4: the stage-0 y = 2.0 rows pass max(2σ, 3 %) at every h (|r_L − 1| ≤ 0.020;
χ_L/free 0.90–0.93; G_L/free 0.94–0.96); the free instrument gate holds on every stage-1 lattice (|G_L^free − 1| ≤ 1.8e-15
on 6⁴/8⁴ aaaa); the reader's synthetic self-test flags (a), (c) with the M trigger, and (b) correctly.

**Caveats.** "No verdict" is a statement about the pre-registered proxy and thresholds at L ≤ 8, recorded as written; the
stage-1b order replaced the proxy (K5.8). χ_L at y = 3.0 is 5e-5 … 6e-4 with 10–50 % errors at 45 measurements: e there is
not readable. The 6³×12 light readings carry a free-sized shape-induced light one-point function (K5.7, E9).

**Data.** `data/derived/n1stage/S1/*/*.npz`, `data/derived/n1stage/S0/L4/L4_y2_*.npz` (S4 control); free baselines
`data/derived/free/free_L8_aaaa.json`, `free_L6x12_aaaa.json`, `free_baselines.json`, `free_L4_aaaa.json`. Raw archive:
`results/xi_scan/K_N1_S1/*/*.npz` (four chains from `results/xi_scan/K_N1_S1b/pre/`), `results/xi_scan/K_N1_S0/L4/`.

**Check.** `python claims/K5.5/check.py` (≈ 2 s): recomputes the verdict and every control live.

**Provenance.** Notebook claim C194 items 1–3, 5 (unhiggsed_notebook @ 3c4a02c).
