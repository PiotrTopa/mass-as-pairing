# K8.7 — the ε-vertex density at P_c against h-matched references: P_c moves toward the symmetric side after h-matching; the heavy mass alone removes a large part of the vertex outside SMG; the light-only part carries no phase information

**Statement.** O4 is the expectation of the on-site ε vertex (all four flavours, both taste doublets); O4L its
light-doublet restriction (n^L = ¼ tr Γ G_L(x, x)).
1. O4L is 0.4–0.5 % of O4 at P_c on 6⁴, 0.7–0.9 % on 6³×12 and 9–10 % on 8⁴, flat in h; on 6⁴ (h = 0) it is 3.0·10⁻⁵ in the
   symmetric phase (y = 2.0) and 3.9·10⁻⁵ in SMG (y = 3.0) while O4 is 0.0114 vs 0.0866. The stored observables cannot split
   the vertex into light and heavy parts.
2. With h-matched references at both ends, the position f_O4 = (O4_Pc − O4_SYM(h)) / (O4_SMG(h) − O4_SYM(h)) on 6⁴ falls
   from 0.250(5) at h = 0 to 0.056(1) at h = 3: toward the symmetric side beyond 2σ. Against the h = 0 symmetric value
   O4_Pc(3) is 0.93 of it, against the h-matched one 1.74 times it: "O4 falls to the symmetric-point value" holds only
   without h-matching.
3. At fixed y on 6⁴, h = 0 → 3 removes 46 % of O4 at y = 2.0, 1 % at y = 3.0 and 65 % at P_c: the heavy doublet drops out of
   the vertex outside SMG, so the comparison must be h-matched; after h-matching the direction toward the symmetric side
   survives (K8.5).

**Numbers.** (K8.7 check output) As above.

**Method.** `masspairing.analysis.direction.o4_rows`: O4 and O4L series of each chain with the stage-1b reader's
selection (cut at trajectory 100) and the error model C0; f_O4 errors by a Gaussian Monte Carlo (100 000 draws). The 6⁴
references at y = 2.0 (h = 0, 3) and (3.0, 3) are the chains of K8.5; y = 3.0, h = 0 is the stored stage-1 chain.

**Controls.** A synthetic P_c(h = 3) row at the h-matched SMG value does not read "toward SYM".

**Caveats.** Post hoc. One volume (6⁴) for the h-matched position; the y = 2.0 and (3.0, 3) chains have 88 measurements.

**Data.** `data/derived/n1stage/{S1,S1b,S1c}/`, `data/derived/n1stage/TH/{sym6,y3h3}/*.npz` (their own ts_O4L),
`data/derived/n1stage/o4l_series.npz` (ts_O4L of the stage-1/1b/1c chains, one array per derived chain). Raw archive:
`results/xi_scan/K_N1_S1{,b,c}/`, `results/laneTH/{sym6,y3h3}/*.npz` (bundle `n1-k8`).

**Check.** `python claims/K8.7/check.py` (≈ 10 s).

**Provenance.** Notebook claim C212 (unhiggsed_notebook @ ea0ce47).
