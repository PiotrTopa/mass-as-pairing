# K8.6 — the large-N sign: the explicit mass removes the heavy doublet from the fermion loop that drives the staggered-σ instability, so the large-N critical coupling rises with h; the sign holds for the chain runner's own operator and beyond the Gaussian order; the shift is second order at small h and not a power of h over [0.5, 3]

**Statement.** Model N1 (reduced staggered doublet operator D = (K − hC_χ)⊗1 + i y σ·τ, all-antiperiodic boxes) in a
staggered background σ(x) = ε(x) σ₀ ê₃. Because ε anticommutes with K and commutes with the same-parity C_χ,
D†D = A + y²σ₀² + 2ihyσ₀ (C_χ ε)⊗τ₃ with A = −(K − hC_χ)², and the Gaussian (large-N) curvature of the staggered mode is
V[1 − y²Π(h)] with

  Π(h) = (2/V)[Tr A⁻¹ + 2h² Tr(A⁻¹ C_χε A⁻¹ C_χε)],

the second trace ≤ 0 (κ and the σ quartic do not enter).
1. **Closed form = exact determinant.** On 4⁴ (h = 0.7, y = 1.3) the second σ₀-difference of ½ log det(D†D) equals
   V y² Π(h) to 2·10⁻⁷ relative.
2. **Sign and shape.** Π(h) decreases strictly on 4⁴, 6⁴, 8⁴ and 6³×12 for h ∈ [0, 3], so the large-N critical coupling
   y_MF(h) = Π(h)^{−1/2} rises: on 8⁴ by ×1.016 / 1.056 / 1.106 / 1.159 / 1.271 at h = 0.5 / 1 / 1.5 / 2 / 3. The relative
   distance δ_L(h) = 1 − Π_L(h)/Π_L(0) is second order at small h (δ(0.25)/δ(0.5) = 0.25–0.27: the action is even in h and
   the box is gapped) and its local exponent falls from 1.7–1.8 to 1.0–1.1 between h = 0.5 and 3 on the data lattices: the
   shift is a crossover, not a power of h, in the measured window.
3. **The chain runner's operator.** The staggered curvature computed from the production operator (the dense
   DoubletOperator of the chain runner, no closed form) gives the same y_MF ratios to ≤ 5·10⁻⁵ on 4⁴ and 6⁴, and the three
   σ directions give the same curvature.
4. **Beyond the Gaussian.** The full mean-field potential v(σ₀) = ½σ₀² + ¼σ₀⁴ − (1/V) log|det D| keeps the sign at every
   order in σ₀: the staggered transition stays continuous and moves up, y* = 2.02 → 2.06 → 2.18 → 2.36 at h = 0, 1, 2, 3 on
   4⁴ (recorded on 6⁴: 1.88 → 2.34 at h = 0 → 3); the first-order uniform channel moves up too (4.92 → 5.24).
5. **Stored P_c data (post hoc).** The P_c structure factor is fitted by 1/S(π) = s₀ + b δ_L(h) on three lattices
   (χ²/dof 0.55 / 1.11 / 1.42, 4 dof), where a single power of h (χ²/dof 6.4 / 5.2 / 3.2) and h² (34 / 17 / 25) fail. The
   fitted amplitude is about twice the bare large-N value. On 6⁴ the light pair-channel effective mass follows the
   Gaussian σ-mode form m² = m₀² + a δ_L, and the three independent stage-1c 6⁴ points (in no fit) lie within 0.6σ of that
   curve; these points lie between fitted points at h = 0 and 1, and K8.8 shows that a generic saturating curve fits the
   σ channel as well — the fit carries the sign and the exclusion of a single power, not a selected shape.

The sign is the standard one for an attractive channel driven by the fermion loop: removing loop weight weakens the
instability, so the critical coupling moves to larger y. Read on the data side (K8.5), the light half at P_c moves
toward the symmetric side. The large-N size of the shift is too large (K8.5, P3).

**Numbers.** (K8.6 check output)

| box | Π(0) | y_MF(h)/y_MF(0), h = 0.25 / 0.5 / 1 / 1.5 / 2 / 3 | local exponent of δ, h = 0.5 … 3 |
|---|---|---|---|
| 4⁴ | 0.25000 | 1.0015 / 1.0058 / 1.0230 / 1.0507 / 1.0874 / 1.1794 | 1.94 → 1.48 (not a data lattice) |
| 6⁴ | 0.28764 | 1.0031 / 1.0121 / 1.0448 / 1.0908 / 1.1431 / 1.2538 | 1.83 → 1.08 |
| 8⁴ | 0.29911 | 1.0042 / 1.0162 / 1.0559 / 1.1056 / 1.1587 / 1.2705 | 1.71 → 0.99 |
| 6³×12 | 0.29048 | 1.0033 / 1.0130 / 1.0476 / 1.0947 / 1.1473 / 1.2586 | 1.80 → 1.06 |

- RPA b = 1.163(32) / 1.021(41) / 1.072(35) on 8⁴ / 6⁴ / 6³×12 (1.8–2.0 × the bare y²Π(0)/3).
- Stage-1c points on the 6⁴ Gaussian curve: h = 0.25 0.3921(72) vs 0.3882 (+0.54σ), h = 0.5 0.4041(52) vs 0.4046
  (−0.11σ), h = 0.75 0.4270(57) vs 0.4286 (−0.29σ).
- Control: bosonic mass ½σ² → ½·1.21σ² raises y*(0) by 1.099 (expected 1.1).

**Method.** `masspairing.analysis.largen`: dense inverses (V ≤ 4096) for Π; the production path builds the model with
`masspairing.action.build_model` (κ = −0.01, λ = 1, pattern chiral) and takes log|det D| of `DoubletOperator.dense`;
y* is the smallest y on a 0.02 grid where min_{σ₀>0} v(σ₀) < v(0), with F(u) = −(1/V) log|det D(yσ₀ = u)| tabulated on
481 points of [0, 12]. Fits: weighted least squares on the stage-1b P_c reference rows (stored h ∈ {0, 0.5, 1, 2}, new
h ∈ {1.5, 3}).

**Controls.** δ_L with permuted h labels fails (χ²/dof 198 / 114 / 147); the y* detector follows a known shift of the
bosonic mass. Recorded limit: the shape of a mass on both doublets also fits (χ²/dof 3.06 / 1.02 / 0.54).

**Caveats.** Large N is a Gaussian approximation (the bare y_MF(0) ≈ 1.83 against P_c's 2.41; the fitted amplitude is off
by ≈ 2). Mean field locates the staggered-σ (symmetry-breaking) instability; that the line between the symmetric phase
and SMG follows it for h > 0 is an assumption, supported by the direction of the data (K8.5, K8.7), not derived. The fit
of item 5 was formed after the data were known. The 6⁴ y* values are recorded from the stored scan (16 min of CPU);
its 4⁴ rows are checked live.

**Data.** `data/derived/n1stage/{S1,S1b}/` (the P_c rows), `data/derived/largen/mf_prod.json` (the recorded y* scan; raw
archive `results/laneTHP/mf_prod.json`, bundle `n1-k8`).

**Check.** `python claims/K8.6/check.py` (≈ 6–7 min on one thread).

**Provenance.** Notebook claims C206 and C211 (unhiggsed_notebook @ ea0ce47).
