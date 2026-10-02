"""I.6: the complete SU(4)-invariant 10/6-channel estimator (dense = stochastic; fermionic boundary sign).

    sum_ab <Phibar^{ab}(y, y') Phi^{ab}(x, x')>       = disc10 - 2 K1K2 - 2 tr(G1G2) + 2 tr(G3G4) + 2 K3K4,
    sum_ab <Lambdabar^{ab}(y, y') Lambda^{ab}(x, x')> = disc6  - 2 K1K2 + 2 tr(G1G2) + 2 tr(G3G4) - 2 K3K4,

as class-averaged structure factors at the 16 corner momenta (and corner + p_min), local energies <T_Q>, <T_6>, the
source-direction one-point function and its Wick-connected h-derivative (masspairing.measure.ChannelMeasure).
(1) On stored 4^4 and 6^4 configurations the dense estimator equals an independent real-flavour Wick evaluation
    (masspairing.analysis.k3_exact.channel_reference) at all 16 corners of both channels and for both local
    energies; S(p + pi(1,1,1,1)) = -S(p) (8 independent corners); the two controls (K3K4 omitted; tr(G3G4) replaced
    by sum_ab G3^{ab} G4^{ab}) move the dense chi10 by more than the tolerance.
(2) Boundary sign: under a translation of a hot configuration by two sites in time the structure factors are
    invariant with the fermionic sign of the pair operators and change without it; the free pair correlator on
    4^3 x 8 is uniform over the time slices only with the sign.
(3) Stochastic estimator (8 Z2 doublet-diluted noise vectors, U-statistic over ordered pairs of distinct vectors) vs
    dense: 34 quantities per configuration within 5 sigma, rms pull in [0.6, 1.5]; (4) the stochastic chi10 with
    K3K4 omitted is off by > 5 sigma (rejected).
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.action import build_model
from masspairing.analysis import k3_exact as kx
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice, shift_table
from masspairing.measure import ChannelMeasure

CONFIGS = ("c072_L4_y2.41_g0.1_g60", "c072_L4_y2.41_g0.05_g60.05", "c073_L6_y2.41_g0.1_g60.1")
NSEED = {4: 48, 6: 32}
c = Check("I.6")

# ---------------------------------------------------------------- (1) dense = independent evaluation
for name in CONFIGS:
    F, model = kx.load_config(name)
    S = model.D.dense_S(F)
    o = ChannelMeasure(model, exact=True).bilinears(F)
    ms = ChannelMeasure(model, exact=True)
    ref = kx.channel_reference(ms, S)
    e1 = max(
        np.abs(o["chi10_corners"] - ref["chi10_corners"]).max(),
        np.abs(o["chi6_corners"] - ref["chi6_corners"]).max(),
        abs(o["E10"] - ref["E10"]),
        abs(o["E6"] - ref["E6"]),
    )
    e3 = max(
        np.abs(o["chi10_corners"] + o["chi10_corners"][::-1]).max(), np.abs(o["chi6_pmin"] + o["chi6_pmin"][::-1]).max()
    )
    dA = abs(kx.channel_reference(ms, S, "old_contraction")["chi10"] - o["chi10"])
    dB = abs(kx.channel_reference(ms, S, "no_K3K4")["chi10"] - o["chi10"])
    c.item(
        f"(1) {name}: dense vs independent Wick evaluation (32 corners, 2 energies); S(p+pi) = -S(p); controls "
        "A (old contraction), B (no K3K4) move chi10 by",
        (e1, e3, dA, dB),
        e1 < 1e-10 and e3 < 1e-10 and dA > 1e-8 and dB > 1e-8,
        fmt="{0[0]:.1e}; {0[1]:.1e}; {0[2]:.1e}, {0[3]:.1e}",
    )

# ---------------------------------------------------------------- (2) boundary signs
lat = Lattice((4, 4, 4, 4))
model = build_model(lat, 2.41, -0.01, 1.0)
rng = np.random.default_rng(110)
F = np.asarray(model.start(rng, "hot", 0.8))
sig = model.split(F)[0]
Ft = model.pack(np.roll(sig, 2, axis=4), np.zeros(0))
res = {}
for sgn in (True, False):
    a = ChannelMeasure(model, exact=True, bc_signs=sgn).bilinears(F)
    b = ChannelMeasure(model, exact=True, bc_signs=sgn).bilinears(Ft)
    res[sgn] = max(np.abs(a[k] - b[k]).max() for k in ("chi10_corners", "chi6_corners", "chi10_pmin", "chi6_pmin"))
    res[sgn, "loc"] = max(abs(a[k] - b[k]) for k in ("E10", "E6", "phi_src_dh"))
c.item(
    "(2) hot 4^4 configuration translated by 2 sites in time: change of the structure factors with / without the "
    "boundary sign; local energies and source derivative",
    (res[True], res[False], res[True, "loc"]),
    res[True] < 1e-10 and res[True, "loc"] < 1e-10 and res[False] > 1e-3,
    fmt="{0[0]:.1e} / {0[1]:.1e}; {0[2]:.1e}",
)
lat8 = Lattice((4, 4, 4, 8))
m8 = build_model(lat8, 0.0, -0.01, 1.0, g10=0.1, g6=0.0)
V = lat8.V
Gf = kx.flavour_propagator(m8.D.dense_S(np.zeros(m8.nfields)))
ms = ChannelMeasure(m8, exact=True)
t = ms.coords[3]
xe = ms.even
mu, nu, sg, de, do = [p for p in ms.planes if p[:3] == (0, 3, 1)][0]
xd, we = shift_table(lat8, de)
jo, wo = shift_table(lat8, do)
y = shift_table(lat8, (1, 0, 0, 2))[0][xe]
pc = kx.pair_pieces(Gf, y, jo[y], xe, xd[xe])["phi"]
raw = np.array([pc[t[xe] == k].mean() for k in range(8)])
cov = np.array([(pc * we[xe] * wo[y])[t[xe] == k].mean() for k in range(8)])
c.item(
    "(2) free 4^3 x 8, plane (0,3), pairs 2 t apart: spread of sum_ab <Phibar Phi> over the time slices without / "
    "with the sign",
    (np.ptp(raw), np.ptp(cov)),
    np.ptp(cov) < 1e-10 and np.ptp(raw) > 0.1,
    fmt="{0[0]:.4f} / {0[1]:.1e}",
)
print("      slices without sign:", np.round(raw, 4), " with sign:", np.round(cov, 4))

# ---------------------------------------------------------------- (3) stochastic vs dense, (4) control B, noise
SC = ("chi10", "chi6", "E10", "E6", "phi_src", "phi_src_dh")
for name in CONFIGS:
    F, model = kx.load_config(name)
    L = model.lat.shape[0]
    ex = ChannelMeasure(model, exact=True).bilinears(F)
    runs, sab = [], []
    t1 = time.time()
    for seed in range(NSEED[L]):
        ms = ChannelMeasure(model, n_noise=8, seed=1100 + seed, cg_tol=1e-10)
        samples = ms._solve_samples(F, model.D.prepare(F))
        o = ms.full_channels(None, samples)
        runs.append(o)
        sab.append(o["chi10"] - kx.stochastic_k3k4(ms, samples))
    pulls, noise = [], {}
    for k in SC:
        v = np.asarray([r[k] for r in runs], float)
        se = v.std(ddof=1) / np.sqrt(len(v))
        pulls.append((v.mean() - ex[k]) / se)
        noise[k] = v.std(ddof=1)
        c.record(
            f"    {name} {k}: dense, stochastic, pull, per-configuration noise",
            f"{ex[k]:+.5f} {v.mean():+.5f}({se:.5f}) {pulls[-1]:+.2f} {noise[k]:.5f}",
        )
    for k in ("chi10_corners", "chi6_corners", "chi10_pmin", "chi6_pmin"):
        v = np.asarray([r[k][1:8] for r in runs], float)
        pulls += list((v.mean(0) - ex[k][1:8]) / (v.std(0, ddof=1) / np.sqrt(len(v))))
    p = np.asarray(pulls)
    sv = np.asarray(sab)
    zs = (sv.mean() - ex["chi10"]) / (sv.std(ddof=1) / np.sqrt(len(sv)))
    c.item(
        f"(3) {name}, n_noise 8 x {NSEED[L]} seeds: {len(p)} quantities, max |pull|, rms pull",
        (np.abs(p).max(), np.sqrt(np.mean(p**2))),
        np.abs(p).max() < 5 and 0.6 < np.sqrt(np.mean(p**2)) < 1.5,
        fmt="{0[0]:.2f}, {0[1]:.2f}",
    )
    c.item(
        f"(4) {name}: stochastic chi10 with K3K4 omitted vs dense, pull (rejected if > 5)",
        (sv.mean(), ex["chi10"], zs),
        abs(zs) > 5,
        fmt="{0[0]:+.4f} vs {0[1]:+.4f}: {0[2]:+.1f}",
    )
    c.record(
        f"    {name}: noise per configuration of chi10 at n_noise 8 (absolute, relative)",
        f"{noise['chi10']:.4f}, {noise['chi10'] / abs(ex['chi10']):.2f}  ({(time.time() - t1) / NSEED[L]:.1f} s/seed)",
    )
c.done()
