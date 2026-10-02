# K2.6 — exact flavour-factorised bag expansion on 2³×4; the 2+2 split is positive in bag and conjugate-pair form

**Statement.** Setting: Z(U, h) = ∫Πdχ exp(−½Σ_a χ^aM^aχ^a) Π_x(1 + Uχ¹χ²χ³χ⁴(x)), M^a = K − h_aC with C = C_χ the
taste-chiral mass (K2.5), on 2³×4 (32 sites, 128 Grassmann variables; the κ = 0 model, σ integrated out exactly).
Expanding the product, Z = Σ_B U^{|B|} Π_a Pf(M^a_{B̄}) over site sets B, and every bilinear expectation value is a
derivative of this sum. All sets with |B| ≤ 4 (36 457) are enumerated exactly, plus 200 random larger sets
(|B| = 6 … 32). (1) *Factorisation sign.* Pf(G_full[B]) in the vertex ordering equals +Π_a Pf(G^a[B]) on every set for
the flavour assignments 1+3, 2+2 and flavour-blind: the bag weight is the product of the four flavour Pfaffians with no
extra sign. (2) *1+3 (source on one flavour): the light amplitude is a symmetry zero.* For C′ = C_χ and for C′ = C₀, the
numerator of Σ_{a≤3}⟨½χ^aC′χ^a⟩ is exactly 0 at orders U⁰, U², U⁴ and on the random large sets (K is bipartite and C′
same-parity, so tr[(K_{B̄})⁻¹C′_{B̄}] = 0 whenever the restriction is regular), while the heavy amplitude is O(h) ≠ 0 for
C_χ and ≡ 0 for the nodal C₀. These are the per-flavour U(1)s of the κ = 0 model; at κ ≠ 0 only their diagonal survives.
(3) *2+2 (equal Majorana mass on two flavours): every bag weight is a perfect square,*
Π_a Pf(M^a_{B̄}) = [Pf(K_{B̄}) Pf((K − hC)_{B̄})]² ≥ 0: the 2+2 split is sign-free in the fermion-bag representation at
every h and volume. (4) *Complex-conjugate-pair HS.* With the vertex written as e^{z(χ¹χ³) + z̄(χ²χ⁴)} averaged over a
complex field z with ⟨z⟩ = ⟨z²⟩ = 0, ⟨|z|²⟩ = U (exact, since the on-site squares vanish), the fermion operator splits into
M₁₃ = [[K, Z], [−Z, K − hC]] and M₂₄ = [[K, Z̄], [−Z̄, K − hC]], and Pf(M₂₄) = conj Pf(M₁₃): the weight is
|Pf M₁₃|² = det(M₁₃†M₁₃)^{1/2} ≥ 0 — an RHMC with one complex pseudofermion, no rooting, no sign, for a Majorana mass on
half the flavours.

**Numbers.** h = 0.5.

| quantity | value |
|---|---|
| factorisation sign, 36 656 sets, 1+3 / 2+2 / blind: worst relative deviation | 1.2 × 10⁻¹⁵ / 1.2 × 10⁻¹⁵ / 3.3 × 10⁻¹⁵ |
| 1+3, C′ = C_χ, orders U⁰, U², U⁴: light numerator | 0, 0, 0 |
| heavy numerator; Z_k | 8.479 × 10⁹, 1.428 × 10¹⁰, 1.341 × 10¹⁰; 6.152 × 10⁹, 1.177 × 10¹⁰, 1.270 × 10¹⁰ |
| 1+3, C′ = C₀: light and heavy numerators, all orders and the large sets | 0 |
| 2+2: negative weights; max relative deviation from the perfect square, 36 657 sets | 0; 0 |
| Pf(M₂₄) vs conj Pf(M₁₃), 20 random z: worst relative deviation; minimal weight | 0; 3.26 × 10⁶ > 0 |

**Method.** Jacobi form of the bag weights, Π_a Pf(G^a[B]) with G^a = (M^a)⁻¹ (the constant Π_a Pf M^a dropped),
order-by-order coefficients N_k = Σ_{|B|=k} ∂_{h′}Π_a Pf(M^a_{B̄} − h′C′_{B̄}) with the derivative −½ p_a tr[(M^a_{B̄})⁻¹C′]
on regular restrictions and a complex step on singular ones (`masspairing.algebra.bags`); Pfaffians by Householder
tridiagonalisation (sign and modulus) and pivoted Parlett–Reid.

**Caveats.** On 2³×4 the kinetic matrix has only time hopping (x + μ̂ = x − μ̂ for L = 2 cancels the two spatial terms,
2 nonzeros per row) and the C_χ entries are 0.25 / 0.5 instead of 0.125: the lattice is eight time-chains coupled through
the source pattern, so the statements here are algebraic identities of this setting (items 1–4 hold for the stated
structure — bipartite K, same-parity C — independently of that). The 2+2 positivity holds in the bag and conjugate-pair
representations; in the σ-triplet representation of the existing RHMC the 2+2 chiral mass has a sign (not part of this
claim). The 1-of-4 split has no positive representation found.

**Data.** None.

**Check.** `python claims/K2.6/check.py` (≈ 3.5 min).

**Provenance.** Notebook claim C148 items 1, 2, 4, 5; the 2³×4 caveat is C151 item 5 (unhiggsed_notebook @ 3c4a02c).
