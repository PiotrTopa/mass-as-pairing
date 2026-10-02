# K1.6 — the four-flavour lattice model: channel algebra, Fierz identities, symmetries; U(1)_ε ≠ U(1)_ψ

**Statement.** *Model.* Four flavours of reduced staggered fermions χ^a(x), one real Grassmann variable per site and
flavour, S = ½ Σ χ^a M χ^a − U Σ_x χ¹χ²χ³χ⁴(x) − g Σ_Q Σ_ab Φ̄^{ab}(y,y′)Φ^{ab}(x,x′), with even sites carrying the 4 of
SU(4) and odd sites the 4̄ (χ_e → Uχ_e, χ_o → U*χ_o) and U(1)_ε: χ(x) → e^{iαε(x)}χ(x). The Sym²(4) = 10 bilinear
Φ^{ab}(x,x′) = χ^a(x)χ^b(x′) + χ^b(x)χ^a(x′) needs two sites of the same parity; on the plaquette quad
Q = (x, x′ = x+μ+ν; y = x+μ, y′ = x+ν) the Fierz forms with K_uv = Σ_a χ^a(u)χ^a(v) are
Σ_ab Φ̄Φ = 2[K_{xy′}K_{x′y} − K_{xy}K_{x′y′}] and Σ_ab Λ̄Λ = −2[K_{xy′}K_{x′y} + K_{xy}K_{x′y′}].
*Channel algebra.* 4 ⊗ 4 = 6 ⊕ 10 has no singlet, 4 ⊗ 4̄ one (the kinetic term); 6 ⊗ 6 has exactly one singlet, ε_abcd
(6 is self-conjugate); 10 ⊗ 10 has none and 10 ⊗ 10̄ one — there is no holomorphic SU(4)-invariant quartic built from
the 10 (the candidate Σ_ab Φ^{ab}Φ^{ab} is SO(4)- but not SU(4)-invariant). On-site χ^aχ^b is purely Λ²(4) = 6.
*Symmetries.* Kinetic, ε and Sym² terms are SU(4)-invariant; kinetic and Sym² terms are U(1)_ε-invariant; the ε term
breaks U(1)_ε exactly to ℤ₄. *Hamiltonian form* (four complex fermions ξ^a(r) per site, H = −tΣ(ξ†ξ + h.c.) +
VΣ(ξ¹ξ²ξ³ξ⁴ + h.c.) − gΣΦ†Φ): Σ_ab Φ†Φ = 2(n_r n_r′ − K†K + n_r′), D_r² = 2ξ¹ξ²ξ³ξ⁴ (D_r = ξ¹ξ² + ξ³ξ⁴);
[N, H_band] = [N, H₁₀] = 0, [N, H_ε] ≠ 0, [e^{iπN/2}, H_ε] = 0; all three terms commute with the 15 SU(4) charges.
*U(1)_ε is not U(1)_ψ.* On the 16 = (4,2,1) ⊕ (4̄,1,2), U(1)_ε acts as Q_ε = +1 on (4,2,1), −1 on (4̄,1,2): it commutes
with the whole Pati–Salam algebra and is orthogonal to spin(10) ⊕ u(1)_ψ; Tr Q_ε = Tr Q_ε³ = Tr Q_ε T₄T₄ = 0 but
Tr Q_ε T_L T_L = Tr Q_ε T_R T_R = 2 (a mixed anomaly with the taste SU(2)s, which are not exact on the lattice);
Tr Q_ψ = Tr Q_ψ³ = 16. The lattice ℤ₄ is the U(1)_ψ ℤ₄ times a Spin(10) element: e^{iπQ_ε/2} = e^{−iπQ_ψ/2} e^{2πi J_L3},
with e^{2πi J_L3} = −1 on (4,2,1) and +1 on (4̄,1,2).

**Numbers.**

| quantity | value |
|---|---|
| singlets in 4⊗4, 4⊗4̄, 6⊗6, 6⊗6̄, 10⊗10, 10⊗10̄ | 0, 1, 1, 1, 0, 1 |
| 6⊗6 singlet vs ε_abcd (relative deviation) | 1.6 × 10⁻¹⁶ |
| Fierz forms on a quad (max coefficient error) | 0, 0 |
| invariance under random SU(4) (3 draws), U(1)_ε (max change) | ≤ 3.6 × 10⁻¹⁵, ≤ 8.9 × 10⁻¹⁶ |
| change of the ε term under U(1)_ε at random angles (min) / at π/2 | 1.75 / 2.4 × 10⁻¹⁶ |
| Σ_ab Φ^{ab}Φ^{ab}: change under SU(4) / under SO(4) | 4.20 / 2.7 × 10⁻¹⁵ |
| Hamiltonian identities and commutators (2-site cluster) | 0 (exact) |
| residual of Q_ε outside spin(10) ⊕ u(1)_ψ (Q_ψ: 9.2 × 10⁻¹⁶) | 1.0000 |
| Tr Q_ε, Tr Q_ε³, Tr Q_εT₄T₄; Tr Q_εT_LT_L, Tr Q_εT_RT_R; Tr Q_ψ, Tr Q_ψ³ | 0, 0, 0; 2, 2; 16, 16 |
| ℤ₄ identity: max difference (with the other sign of Q_ψ) | 2.5 × 10⁻¹⁶ (2.0) |

**Method.** SU(4) representations on 4 ⊗ 4 and on Sym²/Λ² with numerical joint null spaces of the generators; exact
Grassmann polynomials of the four-flavour fields on one quad (`masspairing.algebra.staggered.FieldAlgebra`) and linear
substitutions χ → Vχ; dense Jordan–Wigner operators on a 2-site cluster (`masspairing.algebra.hamiltonian`); 16 × 16
linear algebra on the Pati–Salam blocks of K1.5.

**Caveats.** U(1)_ψ is not a symmetry of the real reduced staggered fields; on Spin(10)-invariant physics the two ℤ₄'s
coincide, on the Pati–Salam-reduced lattice model up to a Pati–Salam-central element.

**Data.** None (exact algebra).

**Check.** `python claims/K1.6/check.py` (≈ 3 s).

**Provenance.** Notebook claims C020, C101 (unhiggsed_notebook @ 3c4a02c).
