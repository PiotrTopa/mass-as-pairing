# I.9 — Pfaffian sign of the flavour-selective source: three routes agree, the pencil gives the flip points

**Statement.** For the flavour-selective source, M(h) = M(0) − h C₀ ⊗ diag(0, 0, 0, 1) in the real 4V basis
(M(0) = Re[W D_c W†], Pf M(0) = det D_c > 0), the sign and the log-magnitude of Pf M(h) are given identically by three
independent routes: pivoted Parlett–Reid elimination (`pfaffian.pfaffian_logpr`), LAPACK Householder
tridiagonalisation of the dense 4V × 4V matrix (`pfaffian_householder`), and the Schur route through the sourced-flavour
block of the h = 0 propagator, Pf M(h)/Pf M(0) = Pf(G_F − h G_F C_F G_F)/Pf(G_F) (`pf_sign_config`, one dense inverse of
D_c per configuration for all h). The sign changes exactly at h = 1/λ for the real positive eigenvalues λ of G_F C_F
(`sign_flips`; each λ is doubly degenerate, hence a simple root of the Pfaffian). The sourced flavour is immaterial:
the source on flavour 3 gives the same Pfaffian as on flavour 4 on every configuration (SU(2)_R commutes with D_c and
acts transitively on the flavour directions).

**Numbers.**

| test | result |
|---|---|
| 24 random antisymmetric matrices, n = 8 … 78: Parlett–Reid vs Householder (sign, log\|Pf\|) | 1.4e-13 |
| 2 log\|Pf\| = log\|det\| | 1.6e-13 |
| congruence Pf(BABᵀ) = det B · Pf A | 3.1e-13 |
| negative Pfaffians among the 24 | 9 |
| physical M(h), 3 configurations × h ∈ {0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1}: three routes | 1.2e-12 |
| flavour-blind mask: sign +1, log ratio = log det D_c(h)/det D_c(0) | 1.0e-12 |
| first negative Pfaffian on the grid | hot 4⁴ configuration at h = 0.2 (all routes −1) |
| sign(h) on h = 0.1 … 6 vs (−1)^#(pencil flips below h), three configurations | identical |
| pencil flip points: hot configuration / stored P_c wedge configuration / stored P_c ε configuration | 0.129, 1.067 / 1.862 / none below 6 |
| source on flavour 3 vs flavour 4 (h = 0.3) | 4.5e-13 |

**Method.** Configurations: two stored equilibrium configurations of the 4⁴ production box (periodic space,
antiperiodic time) at P_c (y, κ) = (2.41, −0.01) — the ε model and the wedge model at g₁₀ = 0.05 — and one hot
configuration of the wedge model at g₁₀ = 0.1. The direct routes act on the dense 4V × 4V M(h); the Schur route and the pencil
on the V × V sourced-flavour block G_F of the dense D_c⁻¹. The scan h = 0.1 … 6 uses the Schur route; the pencil is the
eigenvalue problem of G_F C_F.

**Controls.** Parlett–Reid without the sign of the pivot permutation disagrees in sign on 12 of the 24 random
matrices. The Schur route with C₀ built without the fermionic boundary sign differs from the direct route by 1.28 in
log|Pf| at h = 0.3. A negative Pfaffian of the physical operator is exhibited, so the sign test is not vacuous.

**Caveats.** This is the method part: the routes, the pencil and the flavour symmetry. Sign statistics of ensembles are
not claimed here. The stored P_c configurations are positive on the whole production range h ≤ 0.2; flip points
exist at larger h (and on hot configurations at smaller h), so a |Pf|-sampled ensemble is reweighted by the sign
(I.8).

**Data.** `data/configs/n1inst/L4_y2.41_eps_pppa.npz`, `data/configs/n1inst/L4_y2.41_g0.05_pppa.npz` (from the
archive files `results/xi_scan/F8_P3_eps/L4_y2.41_k-0.01_g0_g60.npz`, `results/xi_scan/F8_P1_wedge/L4_y2.41_k-0.01_g0.05_g60.npz`;
`scripts/derive_n1inst.py configs`).

**Check.** `python claims/I.9/check.py` (≈ 1.5 min): prints and asserts every number above.

**Provenance.** Notebook claim C141 (method part) (unhiggsed_notebook @ 3c4a02c).
