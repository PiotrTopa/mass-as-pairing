# K5.6 — control S3: the phase-flipped pattern gaps neither doublet in the free theory; on SMG configurations the light timeslice readout is pattern-blind and only the composite response distinguishes the patterns

**Statement.** The phase-flipped pattern (the un-phased plane pair (0,3) + (1,2), no ζ phase, no ε sign; sign-free) is not
a corner mass: in the free theory K − 0.5 C_flip on 6⁴ aaaa the light and heavy doublets are exactly degenerate and both
stay near massless, so the order's expectation for control S3 ("both doublets heavy") presumed a mass the pattern does not
carry; the control is recorded as written. A 6⁴ chain at (y = 3.0, h = 0.5) with the flipped pattern has the same light
t* log-ratio as the chiral chain (the SMG gap, far below the free value), light = heavy (no taste selectivity on SMG
configurations), pattern-blind light-sector readouts, and a composite response that differs from the chiral one by 53σ.

**Numbers.** (K5.6 check output)

1. Free 6⁴ aaaa, h = 0.5: flipped pattern m_L(t*) = m_R(t*) = 0.4093 (free massless 0.4055), G_L(p_min) = G_R(p_min) =
   0.9885; chiral pattern: light 0.4055 / heavy 0.4286, G_L = 1.0000 / G_R = 0.9660. Free χ_L: flipped 0.0877 vs chiral
   0.0278 (3.16×).
2. Flipped chain: 400 trajectories, 75 measurements after the cut, acceptance ≥ 0.94 per 50 trajectories;
   m_L(t*) = 0.2686(20), 72σ below the free flipped value (chiral chain at the same point: 0.2674(13), 0.5σ apart);
   |m_L − m_R| = 0.1σ; χ_L/free 0.0008 (chiral 0.0021); G_L(p_min)/free 0.0128 (chiral 0.0126); A = +0.000(5),
   φ_R = −0.7(6.0)·10⁻⁵.
3. Composite: φ_T_R = −0.000490(7) (flipped) vs −0.000972(6) (chiral), 53σ; φ_R differs by 0.2σ.

**Method.** The flipped pattern injected into the production action (same integrator, Hasenbusch split, start, exact
measurement every 4 trajectories, 100 thermalisation + 400 trajectories); read with the stage-1 reader
(`masspairing.analysis.stage1.read_chain`) next to the free operator with the same pattern and the chiral chain.

**Pre-registration and deviations.** The control was pre-registered in the stage-1 order (notebook commit 42417f8); its
expectation rested on a property the pattern does not have (item 1), so it is recorded, not a verdict input.

**Caveats.** One chain, one point; the real pattern controls of K5 are the free mass test of the taste-chiral pattern and,
in the SMG phase, the composite response.

**Data.** `data/derived/n1stage/S3/L6_y3_h0.5_flipped.npz`, `data/derived/n1stage/S3/free_flipped_L6_aaaa_h{0.0,0.5}.json`,
`data/derived/n1stage/S1/L6/L6_y3_k-0.01_g0_g60_h0.5_chiral_bcaaaa.npz`. Raw archive: `results/laneK5S1A/s3_flipped/`.

**Check.** `python claims/K5.6/check.py` (≈ 1 s).

**Provenance.** Notebook claim C198 (unhiggsed_notebook @ 3c4a02c).
