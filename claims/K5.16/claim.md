# K5.16 — Every inequivalent light mass-type channel at h = 2: on (4⁴, 8⁴) no light mass channel grows faster than free at the SSB scale, at y = 3.0 or at P_c; the (6⁴, 8⁴) excess at y = 3.0 is the 6⁴ mode set

**Statement.** The symmetry-breaking criteria of K5.5, K5.8 and K5.11 read one channel, the anti-self-dual (0,3) light
Majorana channel. Here the pair susceptibility χ = V⟨φ²⟩ of the light part of every one of the sixteen corner mass
strings is measured at h = 2 on 4⁴, 6⁴ and 8⁴, at y = 3.0 (SMG) and at P_c. On 4⁴ aaaa the sixteen strings fall into six
classes under the symmetry group (the same under the point-group stabiliser and with the shifts). On (4⁴, 8⁴) no
channel grows faster than free at the SSB scale (e − e_free ≈ +1): at y = 3.0 the three anti-self-dual Majorana channels
fall (−0.325(48), −0.504(70), −0.382(69)), the one-link LR channels — the light–heavy Dirac mixing a seesaw would need —
are free-like (−0.055(15)), and the sextet channels fall or stay (−0.016 … −0.221); at P_c every channel lies within
+0.001 … +0.079 of free scaling, the largest being the on-site (3,1) sextet, the AFM channel. On (6⁴, 8⁴) at y = 3.0
several channels read SSB-sized values (up to +1.755), but in every one of them the 6⁴ value lies below both the 4⁴ and
the 8⁴ value: the 6⁴ light sector is the infrared-most fifth of its modes (K5.13, K5.15), so that pair carries no
symmetry-breaking information in any channel.

**Numbers.** (K5.16 check output; χ/free, orbit classes averaged)

y = 3.0, h = 2 (4⁴ n = 225, 6⁴ n = 250, 8⁴ n = 139):

| light channel | 4⁴ | 6⁴ | 8⁴ | e − e_free (4⁴, 8⁴) | e − e_free (6⁴, 8⁴) |
|---|---|---|---|---|---|
| asd(0,3) (the measured χ_L) | 0.0365 | 0.00839 | 0.0148 | −0.325(48) | +0.493(108) |
| asd(0,1) | 0.0214 | 0.00406 | 0.00529 | −0.504(70) | +0.231(166) |
| asd(0,2) | 0.0166 | 0.00453 | 0.00575 | −0.382(69) | +0.208(191) |
| one-link LR (4 strings) | 0.3903 | 0.0970 | 0.3353 | −0.055(15) | +1.077(36) |
| on-site sextet (3,1) | 0.9721 | 0.1235 | 0.9301 | −0.016(30) | +1.755(62) |
| four-link sextet (3,1) | 0.1410 | 0.0756 | 0.0763 | −0.221(26) | +0.009(55) |
| three-link LR sextet (3,1) (4 strings) | 0.0294 | 0.00913 | 0.0192 | −0.155(11) | +0.644(20) |
| three-link LR sextet (1,3) (4 strings) | 0.0172 | 0.00485 | 0.0129 | −0.101(33) | +0.853(45) |
| four-link sextet (1,3) | −0.0133 | −0.0123 | −0.00519 | — | — |
| on-site sextet (1,3) | −0.1660 | −0.0222 | −0.1564 | — | — |

P_c, h = 2 (4⁴ n = 225, 6⁴ n = 225, 8⁴ n = 45):

| light channel | 4⁴ | 6⁴ | 8⁴ | e − e_free (4⁴, 8⁴) | e − e_free (6⁴, 8⁴) |
|---|---|---|---|---|---|
| asd(0,3) | 0.7942 | 0.8006 | 0.8508 | +0.025(4) | +0.053(10) |
| asd(0,1) | 0.7808 | 0.7821 | 0.8184 | +0.017(6) | +0.039(13) |
| asd(0,2) | 0.7815 | 0.7794 | 0.7995 | +0.008(4) | +0.022(9) |
| one-link LR | 1.1260 | 1.0229 | 1.2023 | +0.024(7) | +0.140(16) |
| on-site sextet (3,1) | 1.9584 | 1.3019 | 2.4346 | +0.079(26) | +0.544(59) |
| four-link sextet (3,1) | 0.9322 | 1.0885 | 1.0564 | +0.045(9) | −0.026(23) |
| three-link LR sextet (3,1) | 0.8069 | 0.8527 | 0.8408 | +0.015(1) | −0.012(2) |
| three-link LR sextet (1,3) | 0.8000 | 0.8329 | 0.8264 | +0.012(1) | −0.007(1) |
| four-link sextet (1,3) | 0.7453 | 0.7242 | 0.7623 | +0.008(2) | +0.045(4) |
| on-site sextet (1,3) | 0.5823 | 0.6922 | 0.5831 | +0.001(2) | −0.149(3) |

Self-dual controls on 8⁴ (sd(0,1), sd(0,2), sd(0,3) = the direction of C_χ): 0.008, 0.008, 0.000 of free at y = 3.0;
0.803, 0.802, 0.666 at P_c (the source's own light block, K5.15); zero light part on 4⁴ and 6⁴.

**Method.** `masspairing.analysis.light_channels`. Channels: the anti-self-dual planes asd(0,j), the self-dual planes
sd(0,j), the four one-link LR strings, the on-site and four-link sextets and the four three-link LR sextets, each sextet
string in its (3,1) and (1,3) flavour parts; light part P_L O P_L (taste-diagonal) or P_L O P_R + h.c. (LR). Samples
(aaaa): y = 3.0 — 4⁴ stage 0 (trajectories ≥ 150), 6⁴ stage-1b new trajectories (≥ 1000), 8⁴ the stage-1c replicas
rA, rB (every second stored configuration ≥ 100) and the stage-1 chain with its extension (every third ≥ 100); P_c —
4⁴ stage 0, 6⁴ and 8⁴ stage 1 (≥ 100). Errors as K5.13 (10 blocks per member, 20 pooled, raised to σ_naive √(2τ_int));
free values of the same channel on the same box. Orbit rows: the mean of the members and their errors added in
quadrature over the number of members (the members share configurations; the combination is the recorded convention).
The per-configuration values were computed in the research notebook from the stored σ configurations of the raw
archive; the 8⁴ ones on one rented A100 GPU in complex128, the 4⁴ and 6⁴ ones on CPU.

**Controls.** The (0,3) channel equals the stored χ_L on every configuration (≤ 4·10⁻¹³), and every stored χ_L equals the
derived chain at the same trajectory. The GPU results agree with the CPU code over all 22 channels to 6.3·10⁻¹⁵ (free
8⁴) and 1.5·10⁻¹⁴ (one stored 8⁴ configuration). The asd(0,3) row reproduces the K5.13 reading of the same pairs within
errors (−0.32(4), +0.51(9)).

**Caveats.** h = 2 only; two volumes per pair; the 8⁴ P_c set has 45 configurations. 4⁴ is about 3ξ across in the SMG
phase, and (4⁴, 8⁴) compares a source-free light doublet with a partly dressed one (K5.15). The (1,3) on-site and
four-link sextets are not positive at y = 3.0 (their bilinear is anti-Hermitian; the interacting part exceeds the free
value with the opposite sign): reported, not read. Exponents are finite-size ratios at L ≤ 8.

**Data.** `data/derived/k5channels/channels_h2.npz` (per configuration: trajectory, stored χ_L, 22 channels; one array
per sample), `data/derived/k5channels/free_and_gates_h2.json` (free values on 4⁴, 6⁴, 8⁴; the GPU/CPU gate values);
chains as in K5.13 for the trajectory match.

**Check.** `python claims/K5.16/check.py` (≈ 15 s).

**Provenance.** Notebook claim C220 (unhiggsed_notebook @ 1494763).
