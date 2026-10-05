# K5.11 — stage 1c at y = 3.0 (SMG), on two independent 8⁴ replicas per h: verdict (a) at h = 1 and h = 2; at h = 2 the light pair susceptibility grows faster with the volume than free without an SSB-scale exponent; (b) does not fire

**Statement.** With the explicit taste-chiral Majorana mass h C_χ on the heavy doublet, in the SMG phase (y = 3.0), the
pre-registered K5 clauses (as in K5.8, thresholds unchanged) applied to two independent 8⁴ chains per h (seeds 1101,
1202 at h = 2; 1303, 1404 at h = 1; 90 measurements each after the cut), each replica alone and pooled, with the stored
h = 0 chains as denominators and the stage-1b 6⁴ rows as the 6⁴ side, read class **(a)** on every set: pooled
r_L^cosh(8⁴) = 0.977(2) at h = 1 and 0.921(2) at h = 2, G_L(p_min)/free = 0.0030 / 0.0034. At h = 2,
e − e_free = 0.600(104) (5.8σ): the recorded partial outcome "growth relative to free, no SSB-scale exponent" is
CONFIRMED — the 2σ lower edge (0.39) is above 0 and below the SSB threshold 0.5 of (b1), R_peak(8⁴) = 0.880(83) shows no
p = 0 peak sharper than free, and χ_L/free = 0.0167(16) (1.7 % of free); (b) does not fire. At h = 1 the growth is not
resolved, e − e_free = 0.296(167). Pooled with the stage-1b extensions (labelled): (a), e − e_free = 0.551(93).

**Numbers.** (K5.11 check output)

| set | r_L^cosh(8⁴) h = 1 / 2 | e − e_free h = 1 / 2 | class |
|---|---|---|---|
| replica A | 0.978(2) / 0.922(2) | 0.358(210) / 0.692(117) | (a) |
| replica B | 0.976(2) / 0.921(2) | 0.229(230) / 0.497(148) | (a) |
| pooled | 0.977(2) / 0.921(2) | 0.296(167) / 0.600(104) | (a) |
| pooled with stage 1b | | — / 0.551(93) | (a) |

Pooled, h = 2: G_L(p_min)/free 0.0034, R_peak(8⁴) 0.880(83), χ_L/free 0.0167(16).

**Scope.** The h = 2 growth e − e_free = 0.600(104) is a (6⁴, 8⁴) ratio between a box on which the light/heavy
taste split exists on 19.8 % of the momenta (6⁴, the infrared-most fifth of the modes) and a box on which it is
complete (8⁴), so it is not a volume exponent of one observable. On the complete pair (4⁴, 8⁴) the same susceptibility
relative to free falls with the volume (e − e_free = −0.32(4) at h = 2, −0.44(7) at h = 1), and at h = 2 it falls
monotonically over the complete boxes 4⁴, 4³×8, 8⁴ with the 6⁴ value below all three (K5.13). R_peak rises from 0.61(5)
on 6⁴ to 0.88(8) on 8⁴ (2.8σ) and stays below 1. Spontaneous breaking at (3.0, h = 2) is not resolved at L ≤ 8: neither
established nor excluded. The clause (b) here is the stage-1b re-specification relative to free, written after the
stage-1 row at the same (y, h) had flagged (b) by its letter (e = 0.786(129), K5.5).

**Method.** `masspairing.analysis.stage1c` (the stage-1b reader per chain; pooled values with each member's
delete-one-block jackknife × its own error-model inflation, conservative by construction); the clauses of
`stage1b.verdict_y3`.

**Pre-registration and deviations.** Chains, seeds, clauses and the confirmation rule were fixed before the chains
started (notebook order K-N1-S1c, commit 56a8e32). The replicas use the same clauses as stage 1b unchanged.

**Controls.** S2′ (injected SSB) → (b); S3′ (injected free return) → (c); S6′ (free null) → not (a)/(b), all on the
pooled rows; sabotage — the pooled χ_L(8⁴, h = 2) set to its free-scaling value gives e − e_free = 0.000(104), "NOT
REPRODUCED".

**Caveats.** L ≤ 8; e − e_free is a (6⁴, 8⁴) finite-size ratio and its central value exceeds the SSB threshold of (b1)
(the 2σ edge does not): incipient symmetry breaking is not excluded by two volumes. The pair channel has no plateau at
L_t ≤ 12 (K5.7). Integrity and replica consistency: K5.9. The same numbers are asserted beside the stage-1b verdict in
K5.8.

**Data.** `data/derived/n1stage/S1c/L8_h{1,2}_r{A,B}/*.npz`, `data/derived/n1stage/{S1,S1b}/`, free baselines in
`data/derived/free/`. Raw archive: `results/xi_scan/K_N1_S1c/L8_*/` (bundle `mass-as-pairing-data_n1-stage1c.tar.gz`).

**Check.** `python claims/K5.11/check.py` (≈ 10 s).

**Provenance.** Notebook claim C204 (unhiggsed_notebook @ fd0c934).
