# M.1 — the moonshot trigger is not met at L ≤ 8: at P_c the light pair-channel effective mass rises with the partner's Majorana mass

**Statement.** The pre-registered moonshot trigger M — at P_c (y = 2.41) the free-normalised light ratio r_L(h) falling
monotonically to r_L(2) + 2σ < 0.7 on both 6⁴ and 8⁴ with a volume-stable log-slope, and absent at y = 3.0 — is not
triggered: r_L rises monotonically in 2σ steps at both volumes, the light pair-channel midpoint cosh mass grows with h on
8⁴, 6⁴ and 6³×12, and the light momentum readout rises toward free. The trigger logic has power (an injected fall
triggers it; the free null does not). In stage 1b the order's fall alert (a fall from h = 1 to 1.5 at > 3σ on 6³×12 and
6⁴) stays silent, and an injected seesaw fires it. There is no critical seesaw of the light half at L ≤ 8 (the rise is the
anti-seesaw, K8.1, K8.2).

**Numbers.** (M.1 check output)

- r_L (t* log-ratio over the free ratio) at h = 0.5, 1, 2: 6⁴ 1.04(4), 1.24(4), 1.84(6); 8⁴ 1.15(17), 2.30(23), 5.84(52);
  6³×12 as written 1.15, 1.48, 2.94 (raw t* ratio 1.12, 1.31, 1.64; the free normalisation is void there, K5.4).
- r_L(2) + 2σ < 0.7 fails on both volumes; log-slope on the last two h: e_r = +0.57(7) (6⁴) vs +1.34(20) (8⁴), not V-stable;
  the absence clause at y = 3.0 holds: r_L(2, 8⁴) = 0.879(4).
- Light pair-channel midpoint cosh mass at P_c, h = 0 → 2: 8⁴ 0.197(9) → 0.614(5), 6⁴ 0.392(8) → 0.569(2), 6³×12 (t = 5)
  0.327(22) → 0.475(1); at h = 2 8⁴/6⁴ = 1.08 while the h = 0 value halves (ratio 0.50).
- G_L(p_min)/free at 8⁴: 0.730, 0.770, 0.784, 0.821 at h = 0, 0.5, 1, 2.
- Stage-1b fall alert: fall significance from h = 1 to 1.5: −5.2σ (6³×12), −10.6σ (6⁴), i.e. rises; silent.

**Method.** `masspairing.analysis.stage1.m_trigger` on the stage-1 rows; `masspairing.analysis.stage1b.T1` (fall alert)
on the stage-1b rows.

**Pre-registration and deviations.** The trigger was fixed before the stage-1 data were read (notebook commits 42417f8,
eaa3692), the fall alert before any stage-1b trajectory (notebook commit 27ea0ee). The trigger presupposed that K7 is
not first order at L ≤ 8 (K7.2, K7.3).

**Controls.** S1_M (r_L := 1/(1 + 3h) at y = 2.41 only, g := free) → M triggered. Free-null injection → M not triggered.
S1′ (injected seesaw on the stage-1b P_c rows) → the fall alert fires.

**Caveats.** L ≤ 8 and h ≥ 0.5; CL_t is a two-fermion (pair) channel and its gap is an effective mass at the stated t
(no plateau at L_t ≤ 12), not a fermion mass.

**Data.** `data/derived/n1stage/S1/*/*.npz`, `data/derived/n1stage/S1b/*/*.npz`, free baselines in `data/derived/free/`.
Raw archive: `results/xi_scan/K_N1_S1/*/*.npz`, `results/xi_scan/K_N1_S1b/*/*.npz`.

**Check.** `python claims/M.1/check.py` (≈ 1.5 min).

**Provenance.** Notebook claims C194 item 4, C202 (fall alert C4) (unhiggsed_notebook @ 3c4a02c).
