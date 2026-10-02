# K2.2 — the pure Sym² term has no sign-free formulation found; the completed term is sign-free

**Statement.** *A. Hamiltonian decouplings (2+1d model of K1.6).* After a Hubbard–Stratonovich decoupling every factor
is exp(c O) with O = (i/4)γᵀAγ hermitian and c real or imaginary. By the Majorana-time-reversal criterion (Li–Jiang–Yao,
PRL 117, 267002 (2016); Wei–Wu–Li–Yao, PRL 116, 250601 (2016)) the weight is non-negative for every field if the ensemble
shares two anticommuting MTR symmetries {T⁺, T⁻} (Majorana class) or {T⁻, T⁻} (Kramers class). The ε-term decoupling
(He–Wu–You–Xu–Meng–Lu 2016), for V > 0 and V < 0, is in the Majorana class on the 2-site and 2 × 2 clusters, with
positive exact weights. Every natural decoupling of the Sym² term — scheme A (complex pairing field on Φ^{ab}, with the
commutator factor of the product decoupling), scheme B (Fierz form −g|Φ|² = −2g[nn′ − K†K + n′]: density plus exchange)
and scheme C (density plus SU(4) spin channel) — has no MTR symmetry at all (L = 0, also with the ε term added), and
its exact weights are negative or complex. Control: the ε ensemble plus a flavour Zeeman term h(n₁ − n₂) loses the
sign-free class and produces negative weights. *B. Fermion bags (Euclidean).* The interaction-expansion weights
⟨Π_Q T_Q^{k_Q}/k_Q!⟩₀ of the Sym² plaquette term are positive wherever tested (2³ to all orders; 4², 4³ to Σk ≤ 3), but
this is a property of the specific combination T_Q = 2[K̃_νK̃_ν′ + K̃_μK̃_μ′], not of the oriented link monomials, which go
negative at 4 and 5 links: it is not a theorem, and the vertex weight is a flavour-summed quantity whose cost is
exponential in the number of vertices. The two-site Λ² term has negative weights already at second order. *C. Real
HS and the Yukawa route.* The only SU(4) × U(1)_ε-invariant bilinears on a quad are the four link K's; a positive real
flavour-blind decoupling e^{gT_Q} = Σ_φ w(φ)e^{Σ a_l K̃_l} would need E[a_l²] = 0 (the two-flavour same-link monomial has
coefficient 2 in K_l² and 0 in e^{gT_Q}), so none exists (moment obstruction); the 10-plet Yukawa Pfaffian is
indefinite for real, imaginary and complex auxiliary fields. *D. The completed term is sign-free.* On every quad
T_Q + Σ_{l∈Q} K_l² = (K̃_ν + K̃_ν′)² + (K̃_μ + K̃_μ′)², and more generally g₁₀T_Q + g₆T₆ + (g₁₀ − g₆)(K_ν² + K_ν′²) +
(g₁₀ + g₆)(K_μ² + K_μ′²) = (g₁₀ − g₆)(K̃_ν + K̃_ν′)² + (g₁₀ + g₆)(K̃_μ + K̃_μ′)², a sum of squares for |g₆| ≤ g₁₀. The real
Gaussian HS e^{gB²} = E_s[e^{sB}] is exact, and the fermion weight is Pf((M − A(s)) ⊗ 1₄) = Pf(M − A(s))⁴ =
det(M − A(s))² ≥ 0 with M − A(s) real antisymmetric: no rooting, HMC-able. The added term is the nearest-neighbour
6-modulus |Φ₆|² (by K1.5 a 50/50 mixture of the 10 and 126 (6,1,1) moduli); the pure 10-channel line is not reachable
sign-free in these routes.

**Numbers.**

| quantity | value |
|---|---|
| MTR, ε term V ≷ 0, 2-site and 2×2: (dim L_sym, dim L_anti, class) | (10, 6, Majorana) |
| MTR, Sym² schemes A, B, C and ε + A, both clusters | L = 0, no T⁻, no class |
| MTR, ε + Zeeman control | (5, 1), T⁺ only, no class |
| exact weights, 2-site cluster, 6 slices, 120 draws: ε (V = ±1, V = 3) | real (\|Im\|/\|w\| ≤ 1 × 10⁻¹¹), 0 % negative |
| ε (V = 3) + Zeeman h = 2 (control) | 22 % negative |
| Sym² scheme A, A + ε, B, C (g = 1) | 47 %, 43 %, 17 %, 45 % negative; \|Im\|/\|w\| up to 1 |
| pairing HS with commutator factor vs e^{λΦ†Φ} at λ = 0.02, 0.04 | 1.7 × 10⁻⁷, 1.6 × 10⁻⁷ (quadrature floor) |
| 2³ plaquette term, all orders | 552 configurations, all positive, min 4.68, max 2.06 × 10⁴ |
| 2³ Λ² term T₆, Σk ≤ 3 | 19/57 negative, first at order 2, config (0,0,0,1,0,1), w = −6.52 |
| 4² (Σk ≤ 3); collinear geometry with orientation; 4³ 2×2×4 block (Σk ≤ 3) | 164 (min 20); 164 (min 20); 285 (min 8.89), all positive |
| oriented K-monomials on ≤ 3 links, 4² | 164 992 nonzero, all positive (min 2.4 × 10⁻⁴) |
| negative 4-link monomial; 5-link monomial (m = 2 each) | −21/256 = −0.0820; −1.857 |
| 10-plet Yukawa Pfaffian on 4², 40 draws: real / imaginary / complex fields | 42 % / 55 % negative (real Pf); \|Im Pf\|/\|Pf\| up to 1, 33 % Re < 0 |
| sum-of-squares identities, every quad of 4² and 2³; g₆ = ±0.5, ±1 | exact (0) |
| Pf(M − A)⁴ = det(M − A)² > 0, 4² and 4³, link fields of amplitude 0.5, 2, 8 | max rel. deviation 1.1 × 10⁻¹² |

**Method.** Dense Jordan–Wigner operators (256 states on 2 sites), BdG → Majorana conversion (verified against the
dense operators), MTR solution spaces as null spaces of the intertwining equations and polar decomposition
(`masspairing.algebra.hamiltonian`); exact weights blockwise in the two fermion-parity sectors. Grassmann algebra and
Wick's theorem factorised over flavours (`masspairing.algebra.staggered`, `.bags`); link monomials from the generating
function pf(a)⁴, pf(a) = Pf(M − Σa_lA_l)/Pf(M), cross-checked by Wick. Pfaffians by pivoted Parlett–Reid.

**Controls.** ε-term monomer weights Pf(G[S])⁴ ≥ 0 on all 255 subsets of 2³; the flavour Zeeman sabotage of the ε
ensemble (class lost, 22 % negative weights); the two-site Λ² term in the same bag expansion (negative at order 2); the
naive orientation of the collinear quads (negative single insertions); g < 0 flips every odd order. Wick conventions are
checked against brute-force Berezin integration.

**Caveats.** "No sign-free formulation found" covers the decouplings and representations listed, not every conceivable
one; the bag positivity is empirical on small lattices. Exploration (not certified): with V = 1, h = 0.7 the 2-site
Zeeman-control weights happen to stay positive (a sign-problematic class does not make every weight negative).

**Data.** None. Table: `results/algebra_mtr.csv` (`scripts/table_algebra_mtr.py`).

**Check.** `python claims/K2.2/check.py` (≈ 5 min, dominated by the exact Hamiltonian weights).

**Provenance.** Notebook claims C021, C022, C023 (unhiggsed_notebook @ 3c4a02c).
