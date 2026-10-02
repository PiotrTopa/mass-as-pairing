"""K3.5: the U(1)_eps-exact family exits through the bond crystal or a gapless fermion, never the (10,3,1) channel.

Six points (y, g10, g6) at kappa = -0.01, lambda = 1, L = 4, 6, 8 (18 chains): the g6 = 0 line and the -edge at y = 0
(U(1)_eps x SU(4) x one-site shift exact; a trivial symmetric gap is forbidden by the Lieb-Schultz-Mattis argument)
and the two -edge points at y = 2.41. Criteria (fixed before the data were read): KEEP (the (10,3,1) condensate
exists) iff the chi10 exponent - 2 sigma >= 0.5 on (6,8) with xi10/L rising; null iff exponent + 2 sigma < 0.5 with
xi10/L falling, chi10 + 2 sigma < 0.6 x free and xi10/L <= free + 2 sigma at every L; bond crystal iff the dimer
exponent - 2 sigma >= 1 on (6,8) with the link-field dimer length xi/L rising and D_perp flat (exponent + 2 sigma
< 0.5). Gaplessness: rho = G_f(middle odd slices)/G_f(edge odd slices) of the zero-momentum fermion correlator
(free: 1).
"""

import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.action import build_model
from masspairing.analysis import k3
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice
from masspairing.measure import ChannelMeasure

P = k3.X_POINTS
CRYSTAL = [(0.0, 0.1, -0.1), (2.41, 0.1, -0.1)]  # g_j = 2 g10 = 0.2 on the -edge
GAPLESS = [(0.0, 0.05, 0.0), (0.0, 0.1, 0.0), (0.0, 0.05, -0.05)]
LS = (4, 6, 8)
NEED = {4: 400, 6: 1000, 8: 1500}
FREE = k3.FREE_CHI10
FREE_CHI6 = {4: 0.5073, 6: 0.3234, 8: 0.2264}  # free chi6, largest corner (000pi), dense
CH = (
    ("chi10", "chi10_corner_max"),
    ("chi6", "chi6_corner_max"),
    ("dimer", "V_dimer_sq_par"),
    ("Dperp", "V_dimer_sq_perp"),
    ("eps", "V_phi_stag_sq"),
)


def val(r, key):
    if key.startswith("V_"):  # V <X> with the series' own error
        return r["V"] * r[key[2:]]["mean"], r["V"] * r[key[2:]]["err"]
    return k3.value(r, key)


def load(extra):
    R, FZ, RHO, PS0 = {}, {}, {}, {}
    for p in P:
        for L in LS:
            ((s, k0),), metas = k3.streams(k3.x_spec(p, L, extra))
            r = R[(p, L)] = k3.ensemble(((s, k0),), metas[0])
            r["n_stored"] = metas[0]["n_stored"]
            sc = k3.cut(s, k0)
            FZ[(p, L)] = k3.freeze_ratio(sc, metas[0])
            RHO[(p, L)] = k3.rho_estimate(sc, r["blen"])
            PS0[(p, L)] = float(np.max(np.abs(sc["ts_phi_stag_sq"])))
    return R, FZ, RHO, PS0


def exponents(R):
    return {
        (p, n, a, b): k3.exponent(val(R[(p, a)], k), val(R[(p, b)], k), a, b)
        for p in P
        for n, k in CH
        for a, b in ((4, 6), (6, 8))
    }


def verdicts(R, EX, RHO):
    out = {}
    for p in P:
        e, s = EX[(p, "chi10", 6, 8)]
        x6, x8 = k3.value(R[(p, 6)], "xi10_cmaxL"), k3.value(R[(p, 8)], "xi10_cmaxL")
        dx = np.hypot(x6[1], x8[1])
        rising, falling = x8[0] - x6[0] > 2 * dx, x8[0] < x6[0] - 2 * dx
        ratio_ok = all(val(R[(p, L)], "chi10")[0] + 2 * val(R[(p, L)], "chi10")[1] < 0.6 * FREE[L][0] for L in LS)
        xi_ok = all(
            k3.value(R[(p, L)], "xi10_cmaxL")[0] <= FREE[L][1] + 2 * k3.value(R[(p, L)], "xi10_cmaxL")[1] + 0.005
            for L in LS
        )
        ed, sd = EX[(p, "dimer", 6, 8)]
        ep, sp = EX[(p, "Dperp", 6, 8)]
        d6, d8 = k3.value(R[(p, 6)], "xi2_s_dimer_0L"), k3.value(R[(p, 8)], "xi2_s_dimer_0L")
        out[p] = dict(
            keep=e - 2 * s >= 0.5 and rising,
            null=e + 2 * s < 0.5 and falling and ratio_ok and xi_ok,
            crystal=ed - 2 * sd >= 1.0 and d8[0] - d6[0] > 2 * np.hypot(d6[1], d8[1]) and ep + 2 * sp < 0.5,
            nogrow=all(
                not (np.isfinite(EX[(p, n, 6, 8)][0]) and EX[(p, n, 6, 8)][0] - 2 * EX[(p, n, 6, 8)][1] >= 0.5)
                for n, _ in CH
            ),
            gapless=RHO[(p, 8)][0] + 2 * RHO[(p, 8)][1] >= 0.9,
        )
    return out


c = Check("K3.5")
R, FZ, RHO, PS0 = load(0)
full = all(r["n_stored"] >= NEED[k[1]] for k, r in R.items())
acc = [r["acceptance"] for r in R.values()]
emdh = [r["exp_mdH"] for r in R.values()]
c.item(
    "18 chains complete; freeze <s_j^2>/2g_j (min, max) >= 0.8; acceptance >= 0.9; <exp(-dH)> within 3 % of 1",
    (min(FZ.values()), max(FZ.values()), min(acc), max(acc), min(emdh), max(emdh)),
    full and min(FZ.values()) >= 0.8 and min(acc) >= 0.9 and max(abs(np.asarray(emdh) - 1)) < 0.03,
    fmt="freeze {0[0]:.2f}-{0[1]:.2f}, acceptance {0[2]:.2f}-{0[3]:.2f}, <exp(-dH)> {0[4]:.3f}-{0[5]:.3f}",
)
z0 = max(PS0[(p, L)] for p in P for L in LS if p[0] == 0.0)
c.item(
    "on-site bilinear at y = 0: max |phi_stag_sq| over all y = 0 series (exact zero: no readout of the on-site 6)",
    z0,
    z0 == 0.0,
    fmt="{:.1e}",
)
tst = {}
for y, g6 in ((0.0, 0.0), (0.0, -0.1), (2.41, -0.1)):
    lat = Lattice((4,) * 4)
    m = build_model(lat, y, -0.01, 1.0, g10=0.1, g6=g6)
    rng = np.random.default_rng(3)
    f = np.asarray(m.start(rng, "hot+hot" if y else "hot+zero"))
    f = f + 0.3 * rng.standard_normal(f.shape)
    tst[(y, g6)] = ChannelMeasure(m, exact=True).bilinears(f)["phi_stag_abs"]
c.item(
    "dense 4^4, arbitrary link fields: |phi_stag| at y = 0 (g6 = 0, -edge) and y = 2.41",
    list(tst.values()),
    tst[(0.0, 0.0)] == 0.0 and tst[(0.0, -0.1)] == 0.0 and tst[(2.41, -0.1)] > 1e-3,
    fmt="{0[0]:.1e}, {0[1]:.1e}, {0[2]:.3f}",
)
for p in P:
    for L in LS:
        r = R[(p, L)]
        cells = [
            val(r, k)
            for k in (
                "chi10",
                "xi10_cmaxL",
                "chi6_corner_max",
                "V_dimer_sq_par",
                "V_dimer_sq_perp",
                "xi2_s_dimer_0L",
                "V_phi_stag_sq",
            )
        ]
        c.record(
            f"{p} L={L}: chi10, xi10/L, chi6 cmax, V<D_par^2>, V<D_perp^2>, xi_dimer/L, V<|phi_st|^2>, rho(Gf)",
            " ".join(f"{a:.4g}({b:.2g})" for a, b in cells) + f" {RHO[(p, L)][0]:.3f}({RHO[(p, L)][1]:.3f});"
            f" chi10/free {cells[0][0] / FREE[L][0]:.2f}, chi6/free {cells[2][0] / FREE_CHI6[L]:.2f}",
        )
EX = exponents(R)
for p in P:
    c.record(
        f"{p} exponents (4,6) (6,8)",
        "  ".join(
            f"{n} {EX[(p, n, 4, 6)][0]:.2f}({EX[(p, n, 4, 6)][1]:.2f}) "
            f"{EX[(p, n, 6, 8)][0]:.2f}({EX[(p, n, 6, 8)][1]:.2f})"
            for n, _ in CH
        ),
    )
V = verdicts(R, EX, RHO)
c.item(
    "(1)+(2) KEEP fails at every point; chi10 null (exponent + 2 sigma < 0.5, xi10/L falling, < 0.6 x free, "
    "xi10/L <= free + 2 sigma) at every point",
    max(EX[(p, "chi10", 6, 8)][0] for p in P),
    not any(v["keep"] for v in V.values()) and all(v["null"] for v in V.values()),
    fmt="largest (6,8) exponent {:+.2f}",
)
c.item(
    "(3) bond crystal grows ~ V at the two g_j = 0.2 points (dimer (6,8) exponents)",
    [EX[(p, "dimer", 6, 8)] for p in CRYSTAL],
    all(V[p]["crystal"] for p in CRYSTAL),
    fmt="{0[0][0]:.2f}({0[0][1]:.2f}), {0[1][0]:.2f}({0[1][1]:.2f})",
)
c.item(
    "(3) no measured channel with (6,8) exponent - 2 sigma >= 0.5 at the four other points",
    True,
    all(V[p]["nogrow"] for p in P if p not in CRYSTAL),
)
r241 = RHO[((2.41, 0.1, -0.1), 8)]
c.item(
    "(4) fermion correlator flat (rho + 2 sigma >= 0.9) at the three y = 0 points without a crystal; gapped at "
    "(2.41, 0.1, -0.1) (rho + 2 sigma < 0.9)",
    [RHO[(p, 8)] for p in GAPLESS] + [r241],
    all(V[p]["gapless"] for p in GAPLESS) and r241[0] + 2 * r241[1] < 0.9,
    fmt="{0[0][0]:.3f}, {0[1][0]:.3f}, {0[2][0]:.3f}; {0[3][0]:.3f}({0[3][1]:.3f})",
)
for p in CRYSTAL:
    q = {}
    for L, nb in ((8, 4), (6, 10)):
        ((s, k0),), metas = k3.streams(k3.x_spec(p, L))
        x = L**4 * np.asarray(s["ts_dimer_sq_par"], float)
        q[L] = x[: len(x) // nb * nb].reshape(nb, -1).mean(1)
    c.record(
        f"{p}: V<D_par^2> by quarters of the 8^4 chain; 6^4 per-100-trajectory blocks (range)",
        " / ".join(f"{v:.1f}" for v in q[8]) + f"; {q[6].min():.1f}-{q[6].max():.1f}",
    )
# (5) controls
ps = (0.0, 0.05, 0.0)
Rs, EXs = copy.deepcopy(R), dict(EX)
f68 = (8 / 6) ** (4 * 0.8)
for k in ("chi10", "chi10_corner_max"):
    Rs[(ps, 8)][k]["mean"] *= f68
    Rs[(ps, 8)][k]["err"] *= f68
Rs[(ps, 8)]["xi10_cmax"]["mean"] = 1.5 * Rs[(ps, 6)]["xi10_cmax"]["mean"] * 8 / 6
EXs[(ps, "chi10", 6, 8)] = k3.exponent(val(Rs[(ps, 6)], "chi10_corner_max"), val(Rs[(ps, 8)], "chi10_corner_max"), 6, 8)
vs = verdicts(Rs, EXs, RHO)[ps]
Rc, EXc = copy.deepcopy(R), dict(EX)
for p in CRYSTAL:
    Rc[(p, 8)]["dimer_sq_par"]["mean"] = Rc[(p, 6)]["dimer_sq_par"]["mean"] * R[(p, 6)]["V"] / R[(p, 8)]["V"]
    Rc[(p, 8)]["xi2_s_dimer_0"]["mean"] = Rc[(p, 6)]["xi2_s_dimer_0"]["mean"] * 8 / 6
    EXc[(p, "dimer", 6, 8)] = k3.exponent(val(Rc[(p, 6)], "V_dimer_sq_par"), val(Rc[(p, 8)], "V_dimer_sq_par"), 6, 8)
vc = verdicts(Rc, EXc, RHO)
c.item(
    f"(5) control: injected chi10 growth (x{f68:.2f}, xi10/L x 1.5 at 8^4) at {ps} -> KEEP, not null",
    (vs["keep"], vs["null"]),
    vs["keep"] and not vs["null"],
)
c.item(
    "(5) control: 8^4 crystal flattened to its 6^4 value -> crystal clause fails at both points",
    [vc[p]["crystal"] for p in CRYSTAL],
    not any(vc[p]["crystal"] for p in CRYSTAL),
)
R2, _, RHO2, _ = load(100)
EX2 = exponents(R2)
V2 = verdicts(R2, EX2, RHO2)
shifts = {k: abs(EX2[k][0] - EX[k][0]) for k in EX if k[2:] == (6, 8) and np.isfinite(EX[k][0])}
kmax = max(shifts, key=shifts.get)
same = all(V2[p] == V[p] for p in P)
c.item(
    "robustness: 100 more trajectories cut from every chain: largest (6,8) exponent shift (where); verdicts "
    "unchanged",
    (shifts[kmax], kmax[:2]),
    shifts[kmax] < 0.1 and same,
    fmt="{0[0]:.3f} {0[1]}",
)
# free correlator flatness (dense, L = 4, 6)
okf = True
for L in (4, 6):
    lat = Lattice((L,) * 4)
    model = build_model(lat, 0.0, -0.01, 1.0, g10=0.1, g6=0.0)
    G = np.asarray(ChannelMeasure(model, exact=True).correlator(np.zeros(model.nfields), x0=(0,) * 4)["Gf"]).real
    okf &= abs(float(k3.rho_flat(G)) - 1.0) < 2e-3
c.item("free fermion correlator flat: rho = 1 at L = 4, 6 (dense)", okf, okf)
c.done()
