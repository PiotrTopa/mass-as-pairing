#!/usr/bin/env python3
"""K6.3 -- the alpha_L ensemble readout on the stage-1 configurations: a saturated Luttinger zero of the light half's
corner block in the SMG phase at every h, a pole-side block moving toward free with h at P_c; free references, the
label-swap control and the resolution floor.

Reads the frozen per-block corner-block means data/derived/n1stage/alpha/S1 (26 chains) and re-scans two stored
configurations (data/configs/n1stage_alpha_L6_y3_h2.npz) live, comparing their corner-block norms with the frozen ones
(about 1 min)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis.n1stage import stored_configs
from masspairing.claimcheck import Check
from masspairing.corner import doublet_operator
from masspairing.lattice import Lattice

c = Check("K6.3")
recs = stage1.alpha_records("S1")
c.item("chains", len(recs), len(recs) == 26)
c.item(
    "projector error (P_L,B, P_R,B orthogonal, rank 16), max; operator identity of the corner-block hook, max",
    (max(r["proj_err"] for r in recs), max(r["hook_identity"] for r in recs)),
    max(r["proj_err"] for r in recs) < 1e-13 and max(r["hook_identity"] for r in recs) < 2e-14,
    "{0[0]:.1e}; {0[1]:.1e}",
)
# live re-scan of two stored configurations of the (6^4, 3.0, h = 2) chain
z, ch = stored_configs("n1stage_alpha_L6_y3_h2")
fr = next(r for r in recs if r["L"] == 6 and r["Lt"] == 6 and r["y"] == 3.0 and r["h"] == 2.0)
shape, bc = tuple(ch["shape"]), tuple(ch["bc"])
lat = Lattice(shape, bc=bc)
worst = 0.0
for j, f in enumerate(z["fields"]):
    sig = np.asarray(f, float)[: 3 * lat.V].reshape((3,) + shape)
    D = doublet_operator(
        lat, ch["y"], sig, h=ch["h10"], pattern="chiral", comp=tuple(ch["h10_comp"]), dual_sign=ch["h10_dual_sign"]
    )
    nrm = np.linalg.norm(stage1.corner_blocks4(D, shape, bc), axis=(1, 2))
    k = int(np.where(fr["sel_traj"] == z["cfg_traj"][j])[0][0])
    worst = max(worst, float(np.abs(nrm - fr["all_norms"][k]).max()))
c.item("two configurations re-scanned live: corner-block norms = frozen, max deviation", worst, worst < 1e-10, "{:.1e}")


def get(L, Lt, y, h, pppa=False):
    return next(
        r
        for r in recs
        if r["L"] == L
        and r["Lt"] == Lt
        and abs(r["y"] - y) < 1e-9
        and abs(r["h"] - h) < 1e-9
        and (r["base"][0] == 0) == pppa
    )


A = lambda r: r["rd_L"]["alpha"]
E = lambda r: r["rd_L"]["alpha_err"]
F = lambda r: r["free"]["rd_L"]["alpha"]
N = lambda r: r["rd_L"]["normO1_2sinp1"] / (r["free"]["rd_L"]["normO1"] * 2 * np.sin(r["p1"]))
lats = ((8, 8), (6, 6), (6, 12))
for y in (3.0, 2.41):
    for L, Lt in lats:
        for h in (0.0, 0.5, 1.0, 2.0):
            r = get(L, Lt, y, h)
            c.record(
                f"{L}^3x{Lt} y={y} h={h}",
                f"alpha_L {A(r):+.3f}({E(r):.3f}) [free {F(r):+.3f}]  alpha_R "
                f"{r['rd_R']['alpha']:+.3f}({r['rd_R']['alpha_err']:.3f}) "
                f"[free {r['free']['rd_R']['alpha']:+.3f}]  |O_L(p1)| 2 sin p1 / free {N(r):.3f}  swap "
                f"{r['swap_sigma']:.1f} sigma",
            )
for h in (0.0, 0.5):
    r = get(6, 6, 2.41, h, True)
    c.record(
        f"6^4 pppa y=2.41 h={h}",
        f"alpha_L {A(r):+.3f}({E(r):.3f}) [free {F(r):+.3f}]  alpha_R {r['rd_R']['alpha']:+.3f} [free "
        f"{r['free']['rd_R']['alpha']:+.3f}]  norm/free {N(r):.3f}  swap {r['swap_sigma']:.1f}",
    )

# 1. free references
pp = [get(6, 6, 2.41, h, True) for h in (0.0, 0.5)]
c.item(
    "1. pppa (p = 0): alpha_L^free at h = 0 (pole value -1), h = 0.5",
    (F(pp[0]), F(pp[1])),
    abs(F(pp[0]) + 1) < 1e-9 and abs(F(pp[1]) + 1) < 0.03,
    "{0[0]:+.4f}, {0[1]:+.3f}",
)
fr0 = {(L, Lt): (F(get(L, Lt, 2.41, 0.0)), F(get(L, Lt, 2.41, 2.0))) for L, Lt in lats}
c.item(
    "1. aaaa: alpha_L^free on 8^4 / 6^4 / 6^3x12 at h = 0 and h = 2",
    tuple(v for k in lats for v in fr0[k]),
    abs(fr0[(6, 6)][0] - 0.193) < 0.005
    and abs(fr0[(8, 8)][0] - 0.102) < 0.005
    and abs(fr0[(6, 12)][0] - 0.577) < 0.005
    and fr0[(6, 6)][1] < 0.09
    and fr0[(8, 8)][1] < 0.06,
    "{0[0]:+.3f}->{0[1]:+.3f} / {0[2]:+.3f}->{0[3]:+.3f} / {0[4]:+.3f}->{0[5]:+.3f}",
)
aR = [get(8, 8, 2.41, h)["free"]["rd_R"]["alpha"] for h in (0.5, 1.0, 2.0)]
c.item("1. alpha_R^free rises with h on 8^4 (h = 0.5, 1, 2)", aR, aR[0] < aR[1] < aR[2] and aR[2] > 0.7, "{}")
c.item(
    "1. free massless |O_L(p1)| 2 sin p1 = 1 on 6^4 / 8^4 aaaa",
    max(
        abs(get(L, L, 2.41, 0.0)["free"]["rd_L"]["normO1"] * 2 * np.sin(get(L, L, 2.41, 0.0)["p1"]) - 1) for L in (6, 8)
    ),
    True,
    "{:.1e}",
)
fm = [get(L, Lt, y, h)["rd_L"]["f_M"] for L, Lt in ((6, 6), (8, 8)) for y in (2.41, 3.0) for h in (0.0, 2.0)]
c.item(
    "1. f_M on 6^4/8^4 chains (kinematic, free value 0.75; not used)",
    (min(fm), max(fm), get(8, 8, 2.41, 0.0)["free"]["rd_L"]["f_M"]),
    all(abs(v - 0.75) < 0.02 for v in fm) and abs(get(8, 8, 2.41, 0.0)["free"]["rd_L"]["f_M"] - 0.75) < 1e-6,
    "{0[0]:.3f}-{0[1]:.3f} (free {0[2]:.3f})",
)

# 2. SMG
smg = [get(L, Lt, 3.0, h) for L, Lt in lats for h in (0.0, 0.5, 1.0, 2.0)]
c.item(
    "2. SMG: alpha_L range over every h and lattice",
    (min(A(r) for r in smg), max(A(r) for r in smg)),
    all(0.92 < A(r) < 0.98 for r in smg) and all(A(r) - F(r) > 0.38 for r in smg),
    "{0[0]:.3f}-{0[1]:.3f}",
)
c.item(
    "2. SMG: alpha_L - alpha_L^free on 6^4 / 8^4, range",
    (min(A(r) - F(r) for r in smg if r["Lt"] != 12), max(A(r) - F(r) for r in smg if r["Lt"] != 12)),
    True,
    "+{0[0]:.2f} ... +{0[1]:.2f}",
)
c.item(
    "2. SMG 6^4/8^4: alpha_L above the free Dirac pole of mass 2 (the bandwidth) by > 0.2, min margin",
    min(A(r) - r["massed_L"][-1] for r in smg if r["Lt"] != 12),
    all(A(r) - F(r) > r["massed_L"][-1] - F(r) + 0.2 for r in smg if r["Lt"] != 12),
    "{:.2f}",
)
dev = [
    (
        r,
        abs(A(r) - A(get(r["L"], r["Lt"], 3.0, 0.0))),
        abs(A(r) - A(get(r["L"], r["Lt"], 3.0, 0.0))) / np.hypot(E(r), E(get(r["L"], r["Lt"], 3.0, 0.0))),
    )
    for r in smg
    if r["h"] > 0
]
c.item(
    "2. SMG h-stability: max |alpha_L(h) - alpha_L(0)|; rows within 2 sigma of 9; max distance",
    (max(d for _, d, _ in dev), sum(s <= 2 for _, _, s in dev), max(s for _, _, s in dev)),
    max(d for _, d, _ in dev) < 0.025 and sum(s <= 2 for _, _, s in dev) >= 7 and max(s for _, _, s in dev) < 3,
    "{0[0]:.3f}; {0[1]}; {0[2]:.1f} sigma",
)
r8 = [s for r, _, s in dev if r["L"] == 8]
c.item(
    "2. (a) confirmation at 8^4: distance from h = 0 at h = 0.5, 1, 2 (sigma)",
    r8,
    r8[0] <= 2 and r8[2] <= 2 and 2 < r8[1] < 2.5,
    "{}",
)
nn = {k: [N(get(*k, 3.0, h)) for h in (0.0, 0.5, 1.0, 2.0)] for k in lats}
c.item(
    "2. SMG |O_L(p1)| 2 sin p1 / free: mean on 6^4, 8^4, 6^3x12; fall 6^4 -> 8^4; h-spread (max/min) per lattice",
    (
        np.mean(nn[(6, 6)]),
        np.mean(nn[(8, 8)]),
        np.mean(nn[(6, 12)]),
        np.mean(nn[(6, 6)]) / np.mean(nn[(8, 8)]),
        max(max(v) / min(v) for v in nn.values()),
    ),
    1.6 < np.mean(nn[(6, 6)]) / np.mean(nn[(8, 8)]) < 1.9
    and max(nn[(6, 6)]) / min(nn[(6, 6)]) < 1.07
    and max(nn[(8, 8)]) / min(nn[(8, 8)]) < 1.08
    and max(nn[(6, 12)]) / min(nn[(6, 12)]) < 1.08,
    "{0[0]:.3f}, {0[1]:.3f}, {0[2]:.3f}; {0[3]:.2f}x; {0[4]:.3f}",
)
c.item(
    "2. SMG: alpha_R = alpha_L within 2.5 sigma at every row (screening), max distance",
    max(abs(A(r) - r["rd_R"]["alpha"]) / np.hypot(E(r), r["rd_R"]["alpha_err"]) for r in smg),
    all(abs(A(r) - r["rd_R"]["alpha"]) / np.hypot(E(r), r["rd_R"]["alpha_err"]) < 2.5 for r in smg),
    "{:.1f} sigma",
)

# 3. P_c
pc = [get(L, Lt, 2.41, h) for L, Lt in lats for h in (0.0, 0.5, 1.0, 2.0)]
ex = {(r["L"], r["Lt"], r["h"]): A(r) - F(r) for r in pc}
c.item(
    "3. P_c h = 0: alpha_L - free on 8^4, 6^4, 6^3x12",
    (ex[(8, 8, 0.0)], ex[(6, 6, 0.0)], ex[(6, 12, 0.0)]),
    0.04 < ex[(6, 6, 0.0)] < 0.07 and 0.13 < ex[(8, 8, 0.0)] < 0.17 and 0.05 < ex[(6, 12, 0.0)] < 0.08,
    "+{0[0]:.3f}, +{0[1]:.3f}, +{0[2]:.3f}",
)
c.item(
    "3. P_c h = 2: the excess shrinks (by >= 35 %) on 8^4, 6^4, 6^3x12",
    (ex[(8, 8, 2.0)], ex[(6, 6, 2.0)], ex[(6, 12, 2.0)]),
    all(0 < ex[(L, Lt, 2.0)] < 0.65 * ex[(L, Lt, 0.0)] for L, Lt in lats),
    "+{0[0]:.3f}, +{0[1]:.3f}, +{0[2]:.3f}",
)
nP = {(r["L"], r["Lt"], r["h"]): N(r) for r in pc}
c.item(
    "3. P_c |O_L(p1)| 2 sin p1 / free rises from h = 0 to 2 on 8^4, 6^4, 6^3x12",
    tuple(v for L, Lt in lats for v in (nP[(L, Lt, 0.0)], nP[(L, Lt, 2.0)])),
    all(nP[(L, Lt, 2.0)] > 0.86 and nP[(L, Lt, 0.0)] < 0.77 for L, Lt in lats),
    "{0[0]:.2f}->{0[1]:.2f}, {0[2]:.2f}->{0[3]:.2f}, {0[4]:.2f}->{0[5]:.2f}",
)
r82 = get(8, 8, 2.41, 2.0)
c.item(
    "3. the (c) clause 'alpha_L within 2 sigma of free' is not met: 8^4 h = 2 distance",
    (A(r82), E(r82), F(r82), (A(r82) - F(r82)) / E(r82)),
    (A(r82) - F(r82)) / E(r82) > 5,
    "{0[0]:.3f}({0[1]:.3f}) vs {0[2]:.3f}: {0[3]:.0f} sigma",
)
c.item(
    "3. pppa (p = 0): alpha_L at h = 0, 0.5 and |O_L(p1)| / free",
    (A(pp[0]), E(pp[0]), A(pp[1]), E(pp[1]), N(pp[0]), N(pp[1])),
    all(-0.3 < A(r) < -0.1 for r in pp) and all(0.3 < N(r) < 0.4 for r in pp),
    "{0[0]:+.2f}({0[1]:.2f}), {0[2]:+.2f}({0[3]:.2f}); {0[4]:.2f}, {0[5]:.2f}",
)

# 4. label swap
sw = {(r["L"], r["Lt"], r["y"], r["h"]): r["swap_sigma"] for r in recs if r["h"] > 0 and r["base"][0] > 0}
c.item(
    "4. swap control passes at P_c for h >= 1 on every lattice (min distance) and at h = 0.5 on 6^4",
    (
        min(sw[(L, Lt, 2.41, h)] for L, Lt in lats for h in (1.0, 2.0)),
        max(sw[(L, Lt, 2.41, h)] for L, Lt in lats for h in (1.0, 2.0)),
        sw[(6, 6, 2.41, 0.5)],
    ),
    all(sw[(L, Lt, 2.41, h)] > 2 for L, Lt in lats for h in (1.0, 2.0)) and sw[(6, 6, 2.41, 0.5)] > 2,
    "{0[0]:.0f}-{0[1]:.0f} sigma; {0[2]:.1f} sigma",
)
c.item(
    "4. swap control fails as written at (2.41, 0.5) on 8^4 and 6^3x12 and on every y = 3.0 row (max)",
    (
        sw[(8, 8, 2.41, 0.5)],
        sw[(6, 12, 2.41, 0.5)],
        max(sw[(L, Lt, 3.0, h)] for L, Lt in lats for h in (0.5, 1.0, 2.0)),
    ),
    sw[(8, 8, 2.41, 0.5)] < 2
    and sw[(6, 12, 2.41, 0.5)] < 2
    and all(sw[(L, Lt, 3.0, h)] < 2 for L, Lt in lats for h in (0.5, 1.0, 2.0)),
    "{0[0]:.1f}, {0[1]:.1f}; {0[2]:.1f} sigma",
)

# 5. floor
fl = [r["floor_m"] for r in recs if r["base"][0] > 0]
c.item(
    "5. measured floor (free Dirac mass moving alpha_L by twice the chain's error), range over aaaa chains",
    (min(fl), max(fl)),
    all(f is not None and 0.2 <= f <= 0.5 for f in fl),
    "{0[0]}-{0[1]}",
)
r80 = get(8, 8, 3.0, 0.0)
shifts = [a - F(r80) for a in r80["massed_L"][:4]]
c.item("5. free alpha_L shift at Dirac mass m = 0.1, 0.2, 0.3, 0.5 on 8^4", [round(s, 3) for s in shifts], True)
c.item(
    "5. loop resolution 2 sin(pi/L_t) at L_t = 6, 8, 12", [round(2 * np.sin(np.pi / Lt), 2) for Lt in (6, 8, 12)], True
)
c.done()
