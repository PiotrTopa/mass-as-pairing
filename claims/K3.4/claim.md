# K3.4 — no spontaneous breaking under the same-parity pairing source, L ≤ 8

**Statement.** On the sign-free link-field model at (y, κ, λ) = (2.41, −0.01, 1), g₁₀ = 0.05, at g₆ = +0.05 and g₆ = 0,
the flavour-blind same-parity pairing-channel source h O_h (K → K − hC₀, C₀ the nodal same-parity pairing field —
not a Majorana mass; det D_c > 0, I.7) at h = 0.005, 0.01, 0.02 and L = 4, 6, 8 gives a source-direction one-point
function ⟨Φ_src⟩_h that is linear in h with an L-independent slope S = ⟨Φ_src⟩/h and a vanishing h → 0 intercept. By
the criterion fixed before the verdict was computed there is no spontaneous breaking in the pairing channel at either
point: the susceptibility in the source direction is finite (≈ 1.3 in lattice units), volume-independent from L = 4 to
8, and 0.3–0.4 × its free-theory value — the suppression K3.2 finds for χ₁₀.

**Numbers.**

| point | L | S̄ = ⟨Φ_src⟩/h (weighted over h) | S(0.005)/S(0.02) | m₀ intercept (fit / 2-pt) | exact dΦ/dh at h = 0.005 | S̄/S_free |
|---|---|---|---|---|---|---|
| (2.41, 0.05, +0.05) | 4 | 1.058(64) | 0.49(26) | −0.0031(18) / −0.0048(32) | 1.130(22) | 0.30 |
| | 6 | 1.389(22) | 0.98(8) | −0.0001(6) / −0.0005(11) | 1.346(14) | 0.42 |
| | 8 | 1.304(80) | 0.67(27) | −0.0010(21) / −0.0069(39) | – | 0.41 |
| (2.41, 0.05, 0) | 4 | 1.190(54) | 1.05(26) | −0.0006(17) / +0.0018(33) | 1.163(31) | 0.34 |
| | 6 | 1.286(20) | 1.00(7) | −0.0001(6) / +0.0004(11) | 1.295(10) | 0.39 |
| | 8 | 1.242(65) | 1.40(26) | +0.0026(18) / +0.0045(33) | – | 0.39 |

- d ln S̄/d ln V on (6,8): −0.055(55) and −0.030(48).
- Free theory (y = 0, s = 0, dense, h = 0): dΦ_src/dh = 3.5146 / 3.3305 / 3.1941 at L = 4 / 6 / 8 (8⁴ stored).

**Method.** 18 chains (L = 4: exact propagators, 400 trajectories after 60; L = 6: 1000 after 100, exact propagators
every 4th; L = 8: 1500 after 100, 8 noise vectors every 2nd), trajectory-length jitter 0.3, no cut beyond
thermalisation. Per chain a blocked jackknife (`masspairing.analysis.k3.ensemble`); m₀(L) = intercept of the weighted
linear fit of ⟨Φ_src⟩ over the three h (two-point extrapolation 2⟨Φ⟩(0.005) − ⟨Φ⟩(0.01) as a cross-check); S̄ the
inverse-variance mean of ⟨Φ_src⟩/h; ρ = S(0.005)/S(0.02); the exact h-derivative is the Wick-connected part plus
V var(Φ_src) (exact propagators only, L = 4, 6). Verdict: `masspairing.analysis.k3.source_point`.

**Pre-registration and deviations.** Criterion ("⟨Φ⟩/h independent of L and h → no SSB; ⟨Φ⟩(h → 0) growing with L → SSB",
notebook commit 270b3ab), made operational before the verdict was computed (notebook commit a7c3a71): SSB ⇔ [m₀(8) > 2σ
and m₀(8) − m₀(6) > 2σ] or e(6,8) − 2σ > 0; no SSB ⇔ |m₀(L)| < 2σ at every L, e(6,8) ≤ 2σ, e(6,8) + 2σ < 0.5 and
ρ(L) ≤ 1 + 2σ at L = 6, 8; undecided otherwise.

**Controls.** (1) Link-field equilibration ⟨s²⟩/2g 1.03–1.11 over the first 50 trajectories of all 18 chains.
(2) Linear-response consistency: the exact h-derivative equals S(L, 0.005) within 3σ at L = 4, 6 (pulls +0.4, −0.0,
−2.0, +0.1σ). (3) Power: replacing the 8⁴ one-point functions by the broken-phase pattern S(8, h) = S(6, h) V₈/V₆ gives
"SSB" at both points; the critical pattern S(8, h) = S(6, h)(V₈/V₆)^0.5 gives "SSB" too (never "no SSB").
(4) S̄(P_c) + 2σ < 0.6 × S_free at every L and point (largest ratio 0.42).

**Caveats.** h ≥ 0.005 only; the 8⁴ one-point functions come from the stochastic estimator (errors 0.0015–0.0019,
3–4× the 6⁴ ones), their h-dependence is noisy (χ²/2 up to 1.8 across the three h; ρ(8) = 1.40(26) at g₆ = 0 is a
1.5σ excess); the exact derivative is never read at 8⁴. The source is flavour-blind: it probes the SU(4)-singlet
direction of the pairing channel, the only sign-free direction, not a flavour-selective term. Two volume pairs.

**Data.** `data/derived/k3/F8_P2_src/` (18 chains, archive `results/xi_scan/F8_P2_src/`).

**Check.** `python claims/K3.4/check.py` (≈ 20 s; free 4⁴/6⁴ dense, 8⁴ stored unless `K3_FREE_L8=1`).

**Provenance.** Notebook claim C106 (unhiggsed_notebook @ 3c4a02c).
