# K6.3 — the α_L ensemble readout on the stage-1 configurations: a saturated Luttinger zero of the light corner block in SMG at every h; at P_c a pole-side block moving toward free with h

**Statement.** The odd-part frequency exponent α of the ensemble-averaged corner block, projected on the light (L) and
heavy (R) taste doublets, on every stored stage-1 configuration after the cut:
1. Free references: α_L^free = −1 exactly on the 6⁴ pppa box (p⃗ = 0: the pole value) and +0.19 / +0.10 / +0.58 on
   6⁴ / 8⁴ / 6³×12 aaaa (the spatial momentum enters the p-odd part as an effective gap), so every reading is taken
   against its free value at the same L, L_t, bc, h; α_L^free falls with h (the O(h p²) light term), α_R^free rises.
2. SMG (y = 3.0): α_L = 0.928–0.977 at every h and lattice, +0.74 … +0.91 above free on 6⁴/8⁴ and above the α of a free
   Dirac pole of mass 2 (the bandwidth) by ≥ 0.36 — a saturated zero, not a heavy pole; h-stable, |α_L(h) − α_L(0)| ≤ 0.021;
   the p-odd norm is 5.4 % (6⁴) → 3.1 % (8⁴) of free, falling with L by 1.71× and h-independent within 7 %; α_R = α_L
   within 1.8σ (screening).
3. P_c (y = 2.41): α_L(0) lies above free (gapped side) and the excess shrinks with h while the p-odd norm rises toward
   free (0.64 → 0.88 at 8⁴): the lowest-momentum light block moves toward the free form; the clause "α_L within 2σ of
   free" is not met (8⁴ h = 2: 9σ). On the pppa box (p⃗ = 0) α_L = −0.17 / −0.26, +0.8 above the pole, norm 1/3 of free.
4. The label-swap control (P_L ↔ P_R moves α by > 2σ at h ≥ 0.5) passes at P_c for h ≥ 1 and fails as written where the
   SMG background screens the chirality (every y = 3.0 row) and at (2.41, 0.5) on 8⁴ and 6³×12.
5. Resolution: the free Dirac mass that moves α_L by twice the chain's error is 0.2–0.5 on every aaaa chain; the loop
   resolution 2 sin(π/L_t) is 1.00 / 0.77 / 0.52 at L_t = 6 / 8 / 12.

**Numbers.** (K6.3 check output; α with 10-block jackknife errors inflated by √(2τ_eff); [free])

| lattice | y | h | α_L [free] | α_R [free] | ‖O_L(p₁)‖·2 sin p₁ / free | swap σ |
|---|---|---|---|---|---|---|
| 8⁴ | 3.0 | 0 / 0.5 / 1 / 2 | 0.957(6) / 0.944(9) / 0.936(7) / 0.957(11) [0.10 … 0.05] | 0.959 / 0.951 / 0.944 / 0.934 [0.10, 0.17, 0.33, 0.74] | 0.031 / 0.031 / 0.033 / 0.031 | 0.2 / 0.7 / 0.9 / 1.8 |
| 6⁴ | 3.0 | 0 / 0.5 / 1 / 2 | 0.931(12) / 0.943(9) / 0.931(11) / 0.928(7) [0.19 … 0.08] | 0.934 / 0.958 / 0.938 / 0.931 | 0.054 / 0.052 / 0.054 / 0.055 | 0.3 / 0.8 / 0.5 / 0.3 |
| 6³×12 | 3.0 | 0 / 0.5 / 1 / 2 | 0.977(3) / 0.965(3) / 0.968(4) / 0.968(5) [0.58] | 0.980 / 0.966 / 0.974 / 0.979 | 0.043 / 0.043 / 0.045 / 0.046 | 0.5 / 0.1 / 0.8 / 1.4 |
| 8⁴ | 2.41 | 0 / 0.5 / 1 / 2 | 0.255(22) / 0.211(11) / 0.138(8) / 0.070(2) [0.10, 0.10, 0.09, 0.05] | 0.241 / 0.213 / 0.306 / 0.660 [0.10, 0.17, 0.33, 0.74] | 0.64 / 0.70 / 0.80 / 0.88 | 0.5 / 0.1 / 15.5 / 169 |
| 6⁴ | 2.41 | 0 / 0.5 / 1 / 2 | 0.248(8) / 0.245(9) / 0.214(3) / 0.115(3) [0.19, 0.19, 0.16, 0.08] | 0.249 / 0.277 / 0.344 / 0.653 [0.19, 0.24, 0.35, 0.73] | 0.76 / 0.77 / 0.81 / 0.88 | 0.1 / 3.2 / 22.7 / 113 |
| 6³×12 | 2.41 | 0 / 0.5 / 1 / 2 | 0.642(10) / 0.632(5) / 0.615(6) / 0.587(3) [0.58] | 0.632 / 0.643 / 0.670 / 0.804 [0.58, 0.60, 0.66, 0.83] | 0.71 / 0.73 / 0.78 / 0.87 | 0.8 / 1.1 / 7.3 / 54 |
| 6⁴ pppa | 2.41 | 0 / 0.5 | −0.171(21) / −0.259(25) [−1.000, −1.021] | −0.168 / −0.217 [−1, −0.74] | 0.33 / 0.37 | 0.1 / 1.0 |

- SMG h-stability: within 2σ of h = 0 on 7 of 9 rows, max 2.7σ; at 8⁴: 1.1σ, 2.2σ, 0.0σ at h = 0.5, 1, 2.
- P_c excess over free at h = 0: +0.153 (8⁴), +0.055 (6⁴), +0.065 (6³×12); at h = 2: +0.021, +0.034, +0.012.
- Free α_L shift at Dirac mass m = 0.1 / 0.2 / 0.3 / 0.5 on 8⁴: +0.003 / +0.010 / +0.023 / +0.061.
- f_M on 6⁴/8⁴ chains 0.745–0.763, free 0.750: kinematic on aaaa boxes, recorded, not used.
- Projectors P_L,B, P_R,B orthogonal of rank 16 to 1.4e-14; corner-block operator identity ≤ 1.1e-14; two configurations
  re-scanned live reproduce the frozen block norms to 5.6e-17.

**Method.** Per stored σ after the cut (8⁴: 45, 6³×12: 56, 6⁴: every second, 113) the doublet operator D_c(σ) − hC_χ ⊗ 1₂
(`masspairing.corner.doublet_operator`), one sparse LU, corner blocks G_B(p) at the base momentum (π/L)(1,1,1) (aaaa) or
0 (pppa) and p₀ = ±π/L_t, ±3π/L_t; block means over 10 consecutive blocks; ensemble average first, then the L / R
projection (the taste projector at the base momentum); O(p) = [⟨G(p₀)⟩ − ⟨G(−p₀)⟩]/2; α = ln(‖O(p₃)‖/‖O(p₁)‖)/ln(sin p₃/sin p₁);
free references on σ = 0 and a Dirac-mass scan K + m at the same L, L_t, bc, h (`masspairing.analysis.stage1.alpha_file`).

**Pre-registration and deviations.** The readout, its projectors, the free references and the floor were fixed before
the stage-1 data were read (notebook commit eaa3692), including the reading "relative to free" on aaaa boxes; the order
text's "α_L → −1 (pole)" is the p⃗ = 0 value and applies on the pppa box only. The label-swap clause fails as written on
the rows listed in item 4.

**Controls.** The label swap (item 4); free references (item 1); the Dirac-mass floor (item 5); the pppa box reproducing
the pole value −1 exactly.

**Caveats.** α cannot tell a pole from a zero of the same gap on aaaa boxes (the P_c light gap ≈ 0.6 is below the loop
resolution 0.77 at L_t = 8), so the P_c reading is "pole-side and moving toward free", not a mass. Corner blocks of
stored configurations; 45–113 configurations per chain.

**Data.** `data/derived/n1stage/alpha/S1/*.npz` (block means compressed to the L / R corner subspaces, per-configuration
block norms), `data/derived/n1stage/alpha/ref_*.npz` (free and Dirac-mass references); `data/configs/n1stage_alpha_L6_y3_h2.npz`
(two configurations for the live re-scan). Raw archive: `results/laneK5S1A/alpha/*.npz` (frozen corner-block scans of the
stage-1 configurations).

**Check.** `python claims/K6.3/check.py` (≈ 12 s): recomputes every α from the frozen block means and re-scans two
configurations live. `tests/equivalence/test_n1stage.py` compares the readout with the notebook's frozen
`alpha_summary.json` (max relative deviation 3.4e-13).

**Provenance.** Notebook claim C196 (unhiggsed_notebook @ 3c4a02c).
