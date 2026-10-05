# N.1 — the ℤ₄ of one generation is X = 5(B−L) − 4Y mod 4; while it is exact there is no Majorana neutrino mass and no |ΔL| = 2 amplitude, ν^c alone cannot be gapped symmetrically, and a ℤ₄-preserving heavy ν^c mass leaves one exactly massless neutral fermion per generation

**Statement.** In Standard-Model language, on left-handed Weyl fields:
1. X = 5(B−L) − 4Y is ≡ 1 mod 4 on every field of the 16 (Q, u^c, d^c, L, e^c, ν^c: 1, 1, −3, −3, 1, 5) and ≡ 2 on the Higgs
   (Y = 1/2): the generator acts as i on the 16 and −1 on the 10 — the centre of Spin(10), the ℤ₄ of K1 on matter. Every
   fermion bilinear has X ≡ 2 (every Majorana mass is ℤ₄-odd); every Yukawa ψψH has X ≡ 0.
2. Anomaly count ν = n(X ≡ 1) − n(X ≡ 3) mod 16 (K1.4): one generation 16 ≡ 0; without ν^c 15; ν^c alone 1 — a ℤ₄-symmetric
   gap for ν^c by itself is excluded by anomaly matching.
3. After electroweak breaking the neutral Higgs component has X' = X + 4T₃ ≡ 0, so X' ≡ 5(B−L) − 4Q mod 4 is exact; every
   fermion keeps an odd X' (ν_L, e_L, u_L, d_L ≡ 3; ν^c, e^c, u^c, d^c ≡ 1).
4. With ℤ₄' exact: ν_L ν_L (including the Weinberg operator) and ν^c ν^c have X' ≡ 2 and are forbidden; ν_L ν^c (Dirac)
   is allowed; every |ΔL| = 2 amplitude with ΔB = ΔQ = 0 (neutrinoless double-beta decay included) has ΔX' ≡ 2 and
   vanishes; sphalerons are compatible. A condensate that permits M_R must carry B−L = ±2 with Q = 0 (a 126-like Δ_R).
5. Neutral mass terms pair X' ≡ 1 with X' ≡ 3 only (a bipartite mass matrix). A ℤ₄-preserving heavy ν^c mass is a Dirac
   pairing with an X' ≡ 3 partner S (a sterile fermion or a composite of the symmetric-mass-generation type); with ν_L, S
   and ν^c the matrix has rank ≤ 1: one exactly massless neutral fermion per generation (mostly ν_L), not a
   seesaw-suppressed one (m_D = 0.1, M = 100: eigenvalues 7·10⁻¹⁸, 100, 100). Control: the type-I matrix with the ℤ₄'-odd
   entry M ν^c ν^c gives the seesaw m_D²/M.

**Numbers.** (N.1 check output) As listed above; all charge arithmetic is exact (rationals).

**Method.** Exact fractions for the charges; numpy eigenvalues for the two mass matrices.

**Caveats.** The formula X = 5(B−L) − 4Y is García-Etxebarria–Montero's (Dai–Freed anomalies in particle physics); the
absence of a Majorana neutrino mass with exact ℤ₄ is known (Wang); the bipartite counting of item 5 is the small
addition. Assumes ℤ₄ exact (no explicit breaking) and counts only the neutral fermions named: additional X' ≡ 1 neutral
states allow further Dirac masses, including small Dirac-seesaw-like singular values, but still no Majorana mass and no
|ΔL| = 2. An observed |ΔL| = 2 signal would exclude an exact ℤ₄ in the neutrino sector; a null does not establish it.

**Data.** None.

**Check.** `python claims/N.1/check.py` (< 1 s).

**Provenance.** Notebook claim C209 (unhiggsed_notebook @ ea0ce47).
