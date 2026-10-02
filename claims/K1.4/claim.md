# K1.4 — ℤ₄ is the unique anomaly-free remnant of U(1)_ψ for one generation (η invariants; instanton vertex ψ⁴)

**Statement.** For ν Weyl fermions of charge 1 (one generation is ν = 16):
(a) Dai–Freed: with ξ = η/2 of the twisted Dirac operator on the lens spaces S⁵/ℤ_n carrying a Spin-ℤ_{2m} structure,
the anomaly phase is exp(−2πi ν ξ). Spin-ℤ₄ is anomaly-free for ν = 16 (ℝP⁵ gives ξ = 1/16, ν ≡ 0 mod 16);
Spin-ℤ₈ is anomalous for ν = 16 with phase −1 (S⁵/ℤ₄, ξ = 5/32 and 3/32; ν ≡ 0 mod 32: ν = 32 is free, ν = 48 is not);
Spin-ℤ₁₆ is anomalous for ν = 16, 32, 48 (S⁵/ℤ₈, ξ = 21/64, …; ν ≡ 0 mod 64).
(b) Instantons: T(16) = 2T(10), and under a minimal SU(2) ⊂ Spin(4) ⊂ Spin(10) the 16 contains exactly 4 doublets
(the (4,2,1) of Pati–Salam), so the unit-instanton 't Hooft vertex is ψ⁴ (U(1)_ψ charge 4) and ℤ_n ⊂ U(1)_ψ is free
of the mixed ℤ_n–Spin(10)² anomaly iff n | 4. Both routes leave ℤ₄ as the largest anomaly-free subgroup for one
generation.

**Numbers.**

| quantity | value |
|---|---|
| η_g(S⁵): ζ-regularised spectrum vs fixed-point formula 2L(g), 8 rotations (n, l) | max deviation 3.6 × 10⁻⁶ (fit-limited) |
| control: Spin-ℤ₄, ℝP⁵, h = 1 | ξ = 1/16 |
| control: Spin × ℤ₃, Spin × ℤ₅, charge 1 | order of ξ: 9, 5 |
| Spin-ℤ₈, S⁵/ℤ₄ (h = 1, 3) | ξ = 5/32, 3/32; phase of ν = 16: −1 |
| Spin-ℤ₁₆, S⁵/ℤ₈ (h = 1) | ξ = 21/64; phases of ν = 16, 32, 48: −i, −1, i |
| anomaly-free (ν = 16, 32, 48) | Spin-ℤ₄: yes, yes, yes; Spin-ℤ₈: no, yes, no; Spin-ℤ₁₆: no, no, no |
| minimal anomaly-free ν, all odd charges, m = 2, 4, 8 | 16, 32, 64; equal to Hsieh's conditions (1.3) in 14/14 cases |
| Tr J² on the 16 / on the 10 (one rotation generator) | 4 / 2 = 2 |
| zero modes of a minimal SU(2)₁ instanton: 16, 10 | 4, 2 |

The full table (every lens space S⁵/ℤ_n, n ≤ 4m, and structure h) is `results/algebra_anomaly.csv`.

**Method.** Equivariant APS on the ball B⁶ with a positive-scalar-curvature metric gives η_g(S⁵) = 2L(g),
L(g) = Π_j i/(2 sin(θ_j/2)); η(S⁵/ℤ_n, χ) = (1/n) Σ_{l≠0} χ(l) η_{g^l}; a Spin-ℤ_{2m} structure is a pair [(g̃, h)]
with nh ≡ m mod 2m (`masspairing.algebra.eta`). The fixed-point values are cross-checked against the Dirac spectrum of
the round S⁵ (eigenvalues ±(k + 5/2), equivariant traces from H_k ⊗ S^±, ζ-regularised after an exact quasi-polynomial
fit). Instanton count: SU(2)₁ content of the exact D5 → D3 × A1 × A1 branching, summing the SU(2) index
(2j)(2j+1)(2j+2)/6.

**Caveats.** That lens spaces detect the full bordism group Ω₅^{Spin-ℤ_{2m}} is cited (Hsieh, arXiv:1808.02881), not
certified; the certified content is the η values and the ν-conditions, which coincide with the published ones.

**Data.** None (exact arithmetic). Table: `results/algebra_anomaly.csv` (`scripts/table_algebra_anomaly.py`).

**Check.** `python claims/K1.4/check.py` (≈ 1 s): spectral control, controls, the anomaly table and verdicts, the
comparison with Hsieh's conditions, the Dynkin-index ratio and the zero-mode count.

**Provenance.** Notebook claims C007, C010 (unhiggsed_notebook @ 3c4a02c).
