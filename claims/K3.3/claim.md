# K3.3 — the response of χ₁₀ to the pure 10-channel term is flat at P_c: R(8)/R(6) = 1.02(5) and 0.97(2)

**Statement.** At P_c = (2.41, −0.01), λ = 1, the finite-g response R(L) = [χ₁₀(wedge, g) − χ₁₀(completion, g)]/g of the
complete 10-channel correlator (p = 0, the largest corner) to the pure plaquette term g T_Q — the wedge action is the
per-link completion action minus g T_Q exactly (I.4) — does not grow with the volume from L = 4 to 8 at g = 0.01 and
0.02. By the criterion fixed before the dense 8⁴ numbers were read, the pure (10,3,1) ⊕ two-site-6 term is irrelevant
at both couplings up to L = 8, while the ε-channel response grows on the same chains.

**Numbers.** (8⁴: dense measurements; L = 4, 6: exact-propagator chains)

| g | R(4) | R(6) | R(8) | R(8)/R(6) | d ln R/d ln V (6,8) | R(6)/R(4) | verdict |
|---|---|---|---|---|---|---|---|
| 0.01 | 15.27(1.05) | 16.06(0.51) | 16.42(0.51) | **1.022(45)** | +0.019(38) | 1.05(8) | irrelevant |
| 0.02 | 10.83(42) | 10.01(17) | 9.74(16) | **0.973(23)** | −0.024(21) | 0.92(4) | irrelevant |

- Dense 8⁴ χ₁₀(p = 0): completion 0.3431(42) / 0.2806(22), wedge 0.5073(27) / 0.4754(23) at g = 0.01 / 0.02
  (0.30–0.54 × the free 0.9365, K3.2); τ_int ≈ 0.5 at the 10-trajectory spacing; o/r stream pulls +1.19, −0.62,
  −1.53, +1.04σ.
- q + 2σ_q = 1.11 and 1.02 (< 1.78, the power bound).
- ε-channel control V⟨|φ_stag|²⟩: R_ε = −44(15), −248(66), −460(91) at g = 0.01 and −43(7), −133(28), −288(71) at
  g = 0.02 (L = 4, 6, 8); |R_ε(6)| − |R_ε(4)| = 3.0σ and 3.1σ; R_ε(8)/R_ε(6) = 1.85(62), 2.16(70).

**Method.** Two sign-free models at P_c: ε + per-link completion (`quads="links"`) and ε + wedge (all squares,
g₆ = 0), g = 0.01, 0.02. L = 4, 6: chains with exact propagators (6⁴ wedge g = 0.01 kept from trajectory 450). L = 8:
each chain continued from a copy of its final configuration with dense fermion measurements every 10th trajectory, an
"o" stream (original seed, 300 trajectories past the copy point) and an "r" stream (re-seeded, 250 past the copy point,
first 50 dropped): 30 + 20 dense measurements per chain. Errors: per stream Madras–Sokal τ_int and blocked jackknife;
streams combined by concatenation and by inverse-variance weighting, the larger error used; R errors in quadrature,
ratio errors by linear propagation (`masspairing.analysis.k3.combine_streams`, `response_verdict`).

**Pre-registration and deviations.** Criterion (q = R(8)/R(6)): relevant ⇔ R(8) > 2σ and q − 1 > 2σ_q; irrelevant ⇔
q ≤ 1 + 2σ_q and q + 2σ_q < (V₈/V₆)^0.5 = 1.78 (power against a response growing with the K3.1 threshold exponent
0.5); undecided otherwise; both g must agree. Formulated before the data were read (notebook commit 270b3ab) and
fixed in the check before the dense 8⁴ numbers were read (notebook commit a7c3a71). The ε-control growth on (4,6) is
asserted here as the growth of |R_ε| in units of its error (the ratio R_ε(6)/R_ε(4) = 5.69(2.48) at g = 0.01 is 1.9σ
from 1).

**Controls.** (1) Link-field equilibration ⟨s²⟩/2g (per-link ⟨s²⟩/12g) 1.01–1.05 over the first 50 kept trajectories
of every 8⁴ stream; acceptance 0.60–0.86; o/r streams consistent (< 3σ). (2) Power: injecting R(8) → 2 R(6) into the
dense wedge χ₁₀ (shift +0.157 / +0.206) gives "relevant" at both g; R(8) → 1.78 R(6) (exponent 0.5) gives "relevant"
too. (3) The dense 8⁴ χ₁₀ agrees with the stochastic estimate (8 noise vectors) on the same chains before the copy
point: pulls −0.14, +1.98, +0.26, +1.08σ, the dense errors 19–66× smaller. (4) Blocks of 2 and 5 dense measurements
give ratio errors 0.039–0.044 / 0.021–0.022 and the same verdicts.

**Caveats.** R is a finite-g response, not the g → 0 derivative (R(6) = 16.1 at g = 0.01 vs 10.0 at g = 0.02); the
verdict is "the finite-g response does not grow with L at either coupling". Two volume pairs, L ≤ 8; ξ₁₀ ≈ 0.5 lattice
spacings (K3.2): a statement about IR-irrelevance on this lattice model at P_c, not a scaling dimension. The (6,8)
growth of the ε control is 1.4–1.7σ with 50 dense measurements; the certified control is its (4,6) growth plus the
injected-growth test.

**Data.** `data/derived/k3/F8_P3_compl/`, `F8_P3_wedge/` (L = 4, 6), `F8_P3x_compl*/`, `F8_P3x_wedge*/` (8⁴ streams),
`F8_P3_compl_rA/`, `_rB/` (stochastic replicas), from the archive chains `results/xi_scan/F8_P3*/`. Table
`results/K3_linear_response.csv` (`scripts/table_k3_response.py`), figure `figures/k3_chi10_response.pdf`.

**Check.** `python claims/K3.3/check.py` (≈ 2 s): items (1)–(5) of its docstring.

**Provenance.** Notebook claim C107 (unhiggsed_notebook @ 3c4a02c).
