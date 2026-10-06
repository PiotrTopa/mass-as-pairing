# K8.5 — direction of the shift: at P_c the explicit heavy mass moves the light half toward the symmetric gapless side, not into SMG (pre-registered P1 MIXED as written; P3 deep SMG — the large-N size of the shift is falsified, its direction is seen on the SMG side too)

**Statement.**
1. **P1 (pre-registered, 6⁴).** With f_O = (O_Pc − O_SYM)/(O_SMG − O_SYM) the position of P_c between the symmetric point
   (y = 2.0) and SMG (y = 3.0, h = 0), the light single-fermion readout g = G_L(p_min)/free and pair susceptibility
   c = χ_L/free both move toward the symmetric side from h = 0 to 3: f_g 0.198 → 0.070 (15σ), f_c 0.363 → 0.074 (29σ). The
   third clause, S(π)_Pc(3) closer to the symmetric value S_SYM(3) than to S_SMG(0) beyond 2σ, holds at 1.9σ only, so the
   verdict as written is **mixed / undecided**. No "toward SMG" clause holds. The S clause is direction-blind by
   construction: S(π) peaks at the critical point and falls on both sides.
2. **P3 (pre-registered, 6⁴).** At (y, h) = (3.0, 3), where the large-N estimate (K8.6) put the moved critical line,
   g = 0.0204 and c = 0.0164 < 0.03: **deep SMG**. The large-N size of the shift is too large. From h = 2 to 3 at y = 3.0,
   g and c move toward free (recorded).
3. **Stored P_c series (post hoc).** On 8⁴, 6⁴ and 6³×12 the ε-vertex density O4 falls monotonically with h (8⁴
   0.0355 → 0.0104), g rises monotonically toward 1 and c rises (8⁴ 0.517 → 0.920). On 4⁴ (stage 0, where y = 2.0 rows
   exist) f_g 0.131 → 0.094 and f_c 0.220 → 0.148 at h = 2.
4. **The SMG side (y = 3.0, post hoc).** g and c rise monotonically with h on 8⁴, 6⁴ and 6³×12 (every step non-decreasing
   within 2σ, overall rise > 3σ; 8⁴ g 0.0028 → 0.0033, c 0.0022 → 0.0121), and every one of the four independent stage-1c
   8⁴ replicas has g above the h = 0 value: inside SMG the explicit mass makes the light half less gapped relative to π/L.
   The light pair-channel cosh mass at y = 3.0 falls with h on 6⁴ only (0.763 → 0.712 → 0.610 at h = 0, 2, 3); it is flat
   on 8⁴ (within 0.008) and rises on 6³×12 (0.423 → 0.458), so this direction is carried by g and c.

No light observable moves toward its SMG value at any h on any lattice. The direction agrees with the large-N sign
(K8.6: the critical coupling rises with h).

**Numbers.** (K8.5 check output)

| 6⁴ row | y | h | n | g | c | S(π) | O4 | m |
|---|---|---|---|---|---|---|---|---|
| P_c | 2.41 | 0 | 225 | 0.7351(77) | 0.5684(81) | 10.20(83) | 0.0302 | 0.3919(84) |
| P_c | 2.41 | 3 | 225 | 0.8868(14) | 0.8803(33) | 2.233(82) | 0.0106 | 0.6449(20) |
| SYM | 2.0 | 0 | 88 | 0.9137(15) | 0.8919(17) | 2.27(12) | 0.0114 | 0.7906(16) |
| SYM | 2.0 | 3 | 88 | 0.9524(11) | 0.9507(18) | 1.675(92) | 0.0061 | 0.8241(10) |
| SMG | 3.0 | 0 | 225 | 0.0125(1) | 0.0019(3) | 3.23(14) | 0.0866 | 0.7633(22) |
| SMG | 3.0 | 2 | 225 | 0.0146(2) | 0.0076(6) | 3.07(17) | 0.0858 | 0.7125(18) |
| (3.0, 3) | 3.0 | 3 | 88 | 0.0204(7) | 0.0164(14) | 2.71(25) | 0.0856 | 0.6096(45) |

- P1: Δf_g = −0.128(9), Δf_c = −0.289(10); dS = 0.44(23) (1.9σ); verdict mixed / undecided; classifier controls pass.
- P3: deep SMG; h = 2 → 3 at y = 3.0: Δg = +0.0057(7), Δc = +0.0088(15).

**Method.** `masspairing.analysis.direction`: rows by the stage-1b reader (cut at trajectory 100, error model C0); P1 by a
Gaussian Monte Carlo over the independent chains (200 000 draws); the stored P_c reference series of stage 1b (stored
h ∈ {0, 0.5, 1, 2}, new h ∈ {1.5, 3}); the stage-0 4⁴ chains at y = 2.0, 2.41, 3.0; the four stage-1c 8⁴ replicas.

**Pre-registration and deviations.** P1 and P3 and their clauses were committed before their three 6⁴ chains started
(notebook lane TH, pre-registrations P1 and P3). The chains ran without stored configurations; their trajectory index
per measurement is reconstructed exactly from the fermion-measurement flags (scripts/derive_th.py). Items 3 and 4 are
post hoc, labelled. The (3.0, 3) chain has one 50-trajectory window with acceptance 0.48 (disclosed).

**Controls.** P1 classifier: a synthetic row halfway to SMG with the SMG value of S returns "toward SMG", a row at the
symmetric point returns "toward SYM". SMG side: a synthetic 8⁴ y = 3.0 h = 2 row with g at 80 % of its h = 0 value is
flagged.

**Caveats.** P1 and P3 are on one volume (6⁴), with 88 measurements per new chain. "Toward the symmetric side" is a
finite-volume statement about g and c at p_min: whether the light fermion at (2.41, h > 0) is gapless or has a small
symmetric gap in infinite volume is not measured. The data show the direction of the light observables; they do not
locate a critical line for h > 0. The 8⁴ repeat of P1 reads the same (MIXED as written, clause (i) toward the
symmetric side; K8.9).

**Data.** `data/derived/n1stage/{S0/L4,S1,S1b,S1c}/`, `data/derived/n1stage/TH/{sym6,y3h3}/*.npz`, free baselines in
`data/derived/free/`. Raw archive: `results/xi_scan/K_N1_S0/L4/`, `results/xi_scan/K_N1_S1{,b,c}/`,
`results/laneTH/{sym6,y3h3}/*.npz` (bundle `n1-k8`).

**Check.** `python claims/K8.5/check.py` (≈ 20 s).

**Provenance.** Notebook claims C208 and C214 (unhiggsed_notebook @ ea0ce47).
