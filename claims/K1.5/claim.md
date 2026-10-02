# K1.5 — Pati–Salam embedding and the channel content of the lattice ε and Sym² bilinears

**Statement.** Blocking the ten gammas as 6 + 4 gives Spin(6) × Spin(4) = SU(4) × SU(2)₁ × SU(2)₂ ⊂ Spin(10). On the
Weyl 16 the two Spin(6)-chirality blocks are each an so(6) spinor 4-plet tensored with a doublet, one block a doublet of
SU(2)₁ only and the other of SU(2)₂ only, and the two 4-plets are conjugate: 16 = (4,2,1) ⊕ (4̄,1,2). Exact branching:
10 = (6,1,1) ⊕ (1,2,2); 126 = (6,1,1) ⊕ (10,3,1) ⊕ (10̄,1,3) ⊕ (15,2,2);
120 = (1,2,2) ⊕ (10,1,1) ⊕ (10̄,1,1) ⊕ (6,3,1) ⊕ (6,1,3) ⊕ (15,2,2); 45 = (15,1,1) ⊕ (1,3,1) ⊕ (1,1,3) ⊕ (6,2,2).
Channel content: the Λ²(4) = (6,1,1) bilinear of one block (the on-site ε channel of the lattice model) is exactly half
in the 10 and half in the 126 (mixing angle 45°: the 10's (6,1,1) is LL + RR, the 126's is Γᵢ Γ₆₇₈₉, i.e. LL − RR); the
Sym²(4) = (10,3,1) bilinears of one block lie entirely in the 126; the (1,2,2) of the 10 has no LL or RR component.

**Numbers.**

| quantity | value |
|---|---|
| so(6) Casimir on each chirality block; commutant dimension | 15/4; 4 (= End ℂ²) |
| (SU(2)₁, SU(2)₂) Casimirs of the two blocks | (0, 3/4) and (3/4, 0) |
| LL and RR (6,1,1) bilinears: fraction in the 10, in the 126 | 0.500000, 0.500000 (deviation < 10⁻⁹; angle 45.00°) |
| LL and RR Sym²(4) bilinears (30 dimensions): fraction in the 10, in the 126 | ≤ 6 × 10⁻³¹, 1.000000 |
| (1,2,2) of the 10 restricted to LL / RR | 0 |
| fraction of Γᵢ Γ₆₇₈₉ (i < 6) in the 126 | 1 (to 10⁻¹²) |

**Method.** c₆ = iΓ₀…Γ₅ and c₄ = Γ₆…Γ₉ split the 16 into two 8-dimensional blocks; so(6) Casimir, commutant (null
space of the intertwining equations) and the Casimirs of the two su(2) triplets J₁, J₂ of so(4) on each block
(`masspairing.algebra.spin10.pati_salam`). Branching by exact weight restriction D5 → D3 × A1 × A1 and the Weyl-group
multiplicity formula. Channel fractions by Frobenius projection onto the orthonormal, mutually orthogonal 10- and
126-spans inside Sym²(16).

**Caveats.** None beyond the stated conventions (which su(2) is called 1 is fixed by the basis; the physics labels L, R
of K1.6 follow the doublet content).

**Data.** None (exact algebra). Table: `results/algebra_branching.csv` (`scripts/table_algebra_branching.py`).

**Check.** `python claims/K1.5/check.py` (≈ 1 s).

**Provenance.** Notebook claim C008 (unhiggsed_notebook @ 3c4a02c).
