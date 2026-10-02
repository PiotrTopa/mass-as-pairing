# K6.1 — No frequency winding in the sign-free class; the loop's quantisation scale lies above the bandwidth

**Statement.**
1. **No winding.** Let G_B(p) = W(p)† D⁻¹ W(p) be the corner block of the propagator (the 16 corner shifts p + πA times
   the internal index; `corner.CornerBlocks`) and N_det(p⃗) = (1/2π) Σ_n Arg[det G_B(p₀^{n+1}, p⃗)/det G_B(p₀^n, p⃗)] the
   winding around the antiperiodic time loop. For every operator of the τ₂K / anti-hermitian class — the doublet
   operator D_c(σ) = K ⊗ 1₂ + i y σ·τ on any configuration, and K − hC for any real antisymmetric C — the corner block is
   anti-hermitian (a compression of an anti-hermitian matrix by an isometry), its determinant is real (even dimension),
   every step of the loop has Arg ∈ {0, π}, and N_det is at most a count of sign changes of a real number (half turns
   where the block is singular between two loop points), never a winding. For the Dirac-massive free operator K + m the
   determinant is real by the ±iλ pairing of the spectrum of K. The loop is closed through the time doubler at p₀ = π;
   a relativistic (charge-conjugation-symmetric) fermion is at half filling in every phase.
2. **What the loop carries.** In the τ₂K class G_B(−p) = τ₂ G_B(p)* τ₂ exactly, so H(p) = iG_B(p) has at −p the negated
   spectrum: every eigenvalue branch changes sign across p₀ = 0 on every configuration. The per-configuration content is
   the parity split E = [G_B(p) + G_B(−p)]/2 (p-even, the Majorana / ℤ₄-odd mass part) and O = [G_B(p) − G_B(−p)]/2 (p-odd,
   kinetic), read as the Majorana fraction f_M = ‖E‖²/(‖E‖² + ‖O‖²), the branch-resolved winding
   N_odd = [n − tr(sgn H(p) sgn H(−p))]/2 (n = 32 for the doublet) and the odd-part exponent α (K6.2). For a coherent
   Majorana mass m whose corner block is 2i sin p₀ Γ₀ + i m M (M a hermitian involution anticommuting with Γ₀; the AFM
   background σ = s ε(x) n̂ has m = y s):
   f_M = m²/(m² + 4 sin²p₁), N_odd = n · 4 sin²p₁/(m² + 4 sin²p₁), p₁ = π/L_t;
   the free massless operator has f_M = 0, N_odd = n, α = −1.
3. **Quantisation scale.** Quantisation in the pre-registered sense (N_odd within 1 of 0 or n and f_M within 0.03 of 0 or
   1) needs m ≥ m* = √31 · 2 sin(π/L_t) = 7.87 / 5.57 / 4.26 at L_t = 4 / 6 / 8 — above the staggered bandwidth
   |λ(K)| ≤ 4 — or the trivial side m ≲ 0.4 sin(π/L_t), where N_odd = n whether the fermion is gapless or a Luttinger
   zero. No gap of this model (AFM m = y|Σ_stag| ≈ 0.2–0.5, SMG gap ≲ 1) can quantise the loop at L_t ≤ 8.

**Numbers.**
- No winding (item 1), each case: every loop step within 4.1e-14 of {0, π} (7.4e-13 on the 6⁴ white-noise background),
  |Im det|/|det| ≤ 6.5e-13, N_det = 0. Cases: free massless, free Dirac mass m = 1, coherent AFM σ = 0.4 ε(x) n̂ at
  y = 2.45, white-noise σ at y = 2.45 (L = 4, 6; 32 × 32 blocks); K − hC_χ (h = 0.5, 1, 3) and K − C₀ (L = 4, 6, 8;
  16 × 16); two stored 6⁴ equilibrium configurations at y = 2.028 and 2.792 (κ = −0.01). The 6⁴ white-noise block has
  4 half turns (sign changes of a real determinant) and N_det = 0; all other cases have none.
- τ₂K relation G_B(−p) = τ₂G_B(p)*τ₂ on white-noise σ: 3.1e-14 (L = 4), 6.4e-14 (L = 6); spectrum of H(−p) = minus that
  of H(p).
- Formulas of item 2 on the AFM background, m ∈ {0.37, 0.98, 2, 4, 8}, L = 4, 6, 8: f_M to < 1e-8, N_odd to < 1e-6; e.g.
  L = 8: m = 0.37 f_M = 0.18943, N_odd = 25.94; m = 2 f_M = 0.87226, N_odd = 4.09; m = 8 f_M = 0.99093, N_odd = 0.29. Free
  massless: f_M < 1e-30, N_odd = 32, α = −1.000 (L = 6, 8; α undefined at L_t = 4).
- Quantisation table: no m with 0.37 ≤ m ≤ 4 quantises at any L_t ≤ 8; m = 8 quantises at L_t = 6 and 8, not at 4; the
  massless operator is (trivially) quantised at every L_t.

**Method.** One sparse LU of D per configuration; corner blocks at every p₀ of the antiperiodic loop at p⃗ = 0 by
solving against the 16 n_int plane waves; determinants, parity split and sign functions of the dense blocks.

**Controls.** A synthetic block e^{ip₀} ⊕ 1 is reported as N_det = 1.000000 (no half turns) and a synthetic real block
cos p₀ ⊕ 1 as two half turns: the code measures windings and jumps when they exist.

**Caveats.** The statements are per configuration and exact; the loop is the one of the finite box at p⃗ = 0. Item 3 is
a statement about the coherent mass form; incoherent backgrounds are in K6.2.

**Data.** `data/configs/n1inst/L6_y2.028_eps_pppa.npz`, `data/configs/n1inst/L6_y2.792_eps_pppa.npz` (from the archive
files `results/xi_scan/F_L6_k-0.01/L6_y2.028_k-0.01.npz`, `…/L6_y2.792_k-0.01.npz`; `scripts/derive_n1inst.py configs`).

**Check.** `python claims/K6.1/check.py` (≈ 5 min): prints and asserts every case above.

**Provenance.** Notebook claims C180, C181 (unhiggsed_notebook @ 3c4a02c).
