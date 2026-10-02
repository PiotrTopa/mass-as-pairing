"""K3.4: no spontaneous breaking under the same-parity pairing source, L <= 8.

The source h O_h (K -> K - h C0, C0 the nodal same-parity pairing field, flavour-blind; det D_c > 0, I.7) at
(y, kappa, lambda) = (2.41, -0.01, 1), g10 = 0.05, g6 = +0.05 and 0, h = 0.005, 0.01, 0.02, L = 4, 6, 8 (18 chains).
Observables: <Phi_src>_h and S = <Phi_src>/h; the exact h-derivative (exact propagators, L = 4, 6).
Criterion (fixed before the verdict was computed): see masspairing.analysis.k3.source_point.
Items: (1) completeness and link-field equilibration; (2) verdict per point; (3) the exact derivative equals
S(h = 0.005) within 3 sigma at L = 4, 6; (4) power: a broken-phase pattern S(8) = S(6) V8/V6 is flagged SSB and a
critical pattern S(8) = S(6) (V8/V6)^0.5 is never called "no SSB"; (5) S(P_c) + 2 sigma < 0.6 x the free-theory
response (dense, L = 4, 6 live; L = 8 stored, K3_FREE_L8=1 recomputes it).
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.action import build_model
from masspairing.analysis import k3
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice
from masspairing.measure import ChannelMeasure

LS = (4, 6, 8)
HS = (0.005, 0.01, 0.02)
PTS = (0.05, 0.0)
NEED = {4: 400, 6: 1000, 8: 1500}
FREE_STORED = {4: 3.51456, 6: 3.33051, 8: 3.19412}  # free d<Phi_src>/dh at h = 0 (dense)


def free_response(L):
    lat = Lattice((L,) * 4)
    model = build_model(lat, 0.0, -0.01, 1.0, g10=0.1, g6=0.0)
    return float(ChannelMeasure(model, exact=True).bilinears(np.zeros(model.nfields))["phi_src_dh"])


c = Check("K3.4")
R, FZ = {}, {}
for g6 in PTS:
    for L in LS:
        for h in HS:
            ((s, k0),), metas = k3.streams([(k3.p2_rel(L, g6, h), 0)])
            R[(g6, L, h)] = k3.ensemble(((s, k0),), metas[0])
            R[(g6, L, h)]["n_stored"] = metas[0]["n_stored"]
            FZ[(g6, L, h)] = k3.freeze_ratio(s, metas[0])
full = all(r["n_stored"] >= NEED[k[1]] for k, r in R.items())
c.item("18 chains complete (400 / 1000 / 1500 trajectories)", full, full)
c.item(
    "link-field equilibration <s_j^2>/2g_j, first 50 trajectories (min, max) >= 0.8",
    (min(FZ.values()), max(FZ.values())),
    min(FZ.values()) >= 0.8,
    fmt="{0[0]:.2f} .. {0[1]:.2f}",
)
M = {k: (r["phi_src"]["mean"], r["phi_src"]["err"]) for k, r in R.items()}
for (g6, L, h), r in sorted(R.items()):
    m, e = M[(g6, L, h)]
    dh = "{:.3f}({:.3f})".format(r["dphi_src_dh"]["mean"], r["dphi_src_dh"]["err"]) if L <= 6 else "-"
    c.record(
        f"g6={g6:g} L={L} h={h:g}: acc, <Phi_src>, S = <Phi_src>/h, exact dPhi/dh, chi10(p=0)",
        f"{r['acceptance']:.2f} {m:.5f}({e:.5f}) {m / h:.3f}({e / h:.3f}) {dh} "
        f"{r['chi10']['mean']:.4f}({r['chi10']['err']:.4f})",
    )


def point(g6, M):
    return k3.source_point({(L, h): M[(g6, L, h)] for L in LS for h in HS}, HS, LS)


OUT = {}
for g6 in PTS:
    o = OUT[g6] = point(g6, M)
    for L in LS:
        m0, sm0, sl, ssl = o["m0"][L]
        Sb, eSb, chi2 = o["S"][L]
        c.record(
            f"g6={g6:g} L={L}: m0 (fit), m0 (2-pt), S, chi2/2, rho = S(0.005)/S(0.02)",
            f"{m0:+.5f}({sm0:.5f}) {o['m0_2pt'][L][0]:+.5f}({o['m0_2pt'][L][1]:.5f}) {Sb:.3f}({eSb:.3f}) "
            f"{chi2 / 2:.2f} {o['rho'][L][0]:.2f}({o['rho'][L][1]:.2f})",
        )
    c.item(
        f"(2) point (2.41, 0.05, {g6:+g}): e(6,8) = d ln S/d ln V -> verdict",
        f"{o['e68'][0]:+.3f}({o['e68'][1]:.3f}) -> {o['verdict']}",
        o["verdict"] == "no SSB",
    )
for (g6, L, h), r in sorted(R.items()):
    if L <= 6 and h == HS[0]:
        a, ea = M[(g6, L, h)][0] / h, M[(g6, L, h)][1] / h
        b, eb = r["dphi_src_dh"]["mean"], r["dphi_src_dh"]["err"]
        pull = (a - b) / np.hypot(ea, eb)
        c.item(
            f"(3) g6={g6:g} L={L}: S(0.005) vs the exact derivative, pull",
            f"{a:.3f}({ea:.3f}) vs {b:.3f}({eb:.3f}): {pull:+.1f}",
            abs(pull) < 3,
        )
for fac, need, name in (
    ((8 / 6) ** 4, ("SSB",), "broken phase, S ~ V"),
    ((8 / 6) ** 2, ("SSB", "undecided"), "critical, exponent 0.5"),
):
    for g6 in PTS:
        Mx = dict(M)
        for h in HS:
            Mx[(g6, 8, h)] = (M[(g6, 6, h)][0] * fac, M[(g6, 8, h)][1])
        v = point(g6, Mx)["verdict"]
        c.item(f"(4) injected {name} at L = 8, g6 = {g6:g}: verdict (required {' or '.join(need)})", v, v in need)
FREE = {L: free_response(L) if (L < 8 or os.environ.get("K3_FREE_L8")) else FREE_STORED[L] for L in LS}
c.item(
    "free-theory dPhi_src/dh at h = 0, L = 4, 6, 8 (8: stored unless K3_FREE_L8=1)",
    [FREE[L] for L in LS],
    all(abs(FREE[L] - FREE_STORED[L]) < 2e-3 for L in LS),
    fmt="{0[0]:.4f}, {0[1]:.4f}, {0[2]:.4f}",
)
ratios = {(g6, L): OUT[g6]["S"][L][0] / FREE[L] for g6 in PTS for L in LS}
ok5 = all(OUT[g6]["S"][L][0] + 2 * OUT[g6]["S"][L][1] < 0.6 * FREE[L] for g6 in PTS for L in LS)
c.item("(5) S(P_c)/S_free at every point and L (all + 2 sigma < 0.6)", ratios, ok5, fmt="{}")
c.done()
