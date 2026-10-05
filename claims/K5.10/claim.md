# K5.10 — 0+1D exact instance of "no light mass term without symmetry breaking": with the residual symmetry exact every light mass-type Green's-function part vanishes, one symmetry-breaking term creates one, and a symmetric light gap can still scale like 1/M (pre-registered verdict VOID)

**Statement.** Sixteen Majoranas in 0+1 dimensions (256 states): 8 light (a) and 8 heavy (b), each half with a
Fidkowski–Kitaev (Cayley 4-form) quartic W, a light–heavy quartic λX, a heavy Majorana mass M on the b's and, as the
sabotage, a Higgs-like light–heavy bilinear Φ:

  H = U_a W(a) + U_b W(b) + λ X(a, b) + M Σ_j i b_{2j−1} b_{2j} + Φ Σ_k i a_k b_k.

M breaks the antiunitary T (every Majorana even) and the reflection R_b (b_even → −b_even) and keeps T' = R_b T; every
light bilinear i a_k a_l is T'-odd; Φ breaks T'. This is the general proposition behind K5 (if no light mass bilinear is
invariant under the symmetry left by the explicit mass and that symmetry is not broken spontaneously, every mass-type
part of the light two-point function vanishes) in an interacting model where it can be checked exactly:
1. With T' exact (Φ = 0), at every M ∈ [0, 100] and λ ∈ [−1, 1], the light bilinears and the mass-type (ω-even,
   chirality-flipping) part of the light Green's function vanish to rounding (≤ 5·10⁻¹⁵); the same holds for 12 random
   T'-invariant Hamiltonians, and one T'-odd bilinear makes the part ≥ 0.16.
2. With Φ = 0.3 the mass-type part is nonzero (≥ 7.8·10⁻⁵, a ratio 1.5·10¹⁰ to the zeros), and without light quartic
   the light gap is the seesaw Φ²/M (d ln Δ_L/d ln M = −0.88).
3. Without a light quartic (U_a = 0) the light half is gapped only through virtual heavy excitations: its gap falls like
   1/M (Δ_L = 0.717 → 0.0078 at λ = 0.5, 3.07 → 0.031 at λ = 1 for M = 0 → 100; slopes −1.09, −1.13 over M = 10 … 100) while
   its mass-type part stays zero. A seesaw-like scaling of a light gap therefore does not by itself signal a Majorana
   mass; the discriminator is the mass-type part of the light propagator (pole versus zero at ω → 0).
4. With the light quartic on (U_a = 1) the light symmetric gap moves by at most 6 %, downward, for either sign of λ.

The pre-registered verdict is **VOID** by its own letter: the zero clauses Z/Z′ count only if the sabotage controls S1
and S2 fire, and S1 did not reach its absolute threshold (10⁻³; the threshold assumed an O(U) light gap, the FK gap is
14 U). Items 1–4 are the facts behind the verdict, labelled post hoc where they were not pre-registered.

**Numbers.** (K5.10 check output)

| clause / reading | value |
|---|---|
| FK quartic on 8 Majoranas | unique ground state, gap 14 |
| Z (U_a = 1, Φ = 0): min split; max light bilinear; max mass-type part; min Δ_L(100)/Δ_L(10) | 12.94; 1.2·10⁻¹⁴; 5.1·10⁻¹⁵; 0.985 |
| Z′ (U_a = 0): max mass-type part (threshold 10⁻¹⁰) | 6.3·10⁻⁸ — fails; even/odd ratio 3.3·10⁻¹⁰ at that point (rounding) |
| S1 (Φ = 0.3): min mass-type part (threshold 10⁻³) | 7.8·10⁻⁵ — does not fire |
| S2 (Φ = 0.3, U_a = λ = 0): d ln Δ_L / d ln M | −0.881 |
| verdict as written | VOID |
| Z′ light gap Δ_L(M) at λ = 0.5 / 1, M = 0 → 100; slopes over 10 … 100 | 0.717 → 0.0078 / 3.07 → 0.031; −1.09, −1.13 |
| Z: min Δ_L(M)/Δ_L(0) | 0.940 |
| random T'-invariant H (12/12 unique): max light bilinear / mass-type part; + one T'-odd term: min | 1.6·10⁻¹³; 0.16 |

**Method.** `masspairing.algebra.ed01`: Jordan–Wigner Majoranas on 2⁸ states, dense diagonalisation; the light Green's
function from the Lehmann sum, G_kl(iω) = Σ_n [M_kl/(iω − E_n) + M_lk/(iω + E_n)], M_kl = ⟨0|a_k|n⟩⟨n|a_l|0⟩, whose ω-even
part Σ_n E_n (M_lk − M_kl)/(ω² + E_n²) is the mass-type part (ω = 0.01, 0.1, 1); the light gap is the lowest excitation
with nonzero light weight. Symmetries are checked as operator identities.

**Pre-registration and deviations.** The scans and the clauses Z, Z′, S1, S2 were fixed before the code existed
(notebook lane TH, pre-registration P2 and its amendment 1). The verdict is reported as VOID; items 2 (ratio), 3, 4
and the random-Hamiltonian test are post hoc and labelled.

**Controls.** S1 and S2 as written; the T'-odd bilinear added to each random T'-invariant Hamiltonian.

**Caveats.** 0+1 dimensions has no critical point: the behaviour at P_c (K8) is not tested here. The model checks the
symmetry statement and shows that it fixes nothing about the size or M-dependence of a symmetric light gap.

**Data.** None (pure computation).

**Check.** `python claims/K5.10/check.py` (≈ 15 s).

**Provenance.** Notebook claim C207 (unhiggsed_notebook @ ea0ce47).
