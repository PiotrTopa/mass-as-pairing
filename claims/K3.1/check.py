"""K3.1: the complete 10-channel correlator does not grow with the volume at five link-field points, L <= 8.

Reads the derived series of the 15 (+2 replica) stored chains (data/derived/k3/F8_P1_wedge*). Items: completeness
and link-field equilibration; per point the largest-corner exponent on (6,8) with the kill criterion
"exponent + 2 sigma < 0.5 and xi10/L not rising" (null) vs "exponent - 2 sigma >= 0.8 with xi10/L rising" (alive);
p = 0 is the largest corner; the positive control (columnar dimer susceptibility grows); robustness
with 100 more trajectories cut from every stream.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


from masspairing.analysis import k3
from masspairing.claimcheck import Check

LS = (4, 6, 8)
NEED = {4: 400, 6: 1000, 8: 1500}


def table(extra=0):
    R, FZ = {}, {}
    for p in k3.P1_POINTS:
        for L in LS:
            spec = k3.p1_spec(p, L, extra)
            R[(p, L)] = k3.ensemble_of(spec)
            st, metas = k3.streams(spec)
            s0, k0 = st[0]
            FZ[(p, L)] = k3.freeze_ratio(k3.cut(s0, k0), metas[0])
    return R, FZ


def val(r, key):
    if key in ("dimer_sq_par", "phi_stag_sq"):  # V <X> with the series' own error
        return r["V"] * r[key]["mean"], r["V"] * r[key]["err"]
    return k3.value(r, key)


def exponents(R):
    EX = {}
    for p in k3.P1_POINTS:
        for name, key in (
            ("chi10cmax", "chi10_corner_max"),
            ("chi10", "chi10"),
            ("chi6cmax", "chi6_corner_max"),
            ("dimer", "dimer_sq_par"),
            ("eps", "phi_stag_sq"),
        ):
            for a, b in ((4, 6), (6, 8)):
                EX[(p, name, a, b)] = k3.exponent(val(R[(p, a)], key), val(R[(p, b)], key), a, b)
    return EX


c = Check("K3.1")
R, FZ = table()
# stored-trajectory targets; the split 8^4 chain: original to 1459, replicas to 1420
need = lambda p, L: [1459, 1420, 1420] if (L, p) == (8, (2.41, 0.02, 0.0)) else [NEED[L]]
full = all(s[1] >= n for (p, L), r in R.items() for s, n in zip(r["streams"], need(p, L), strict=True))
c.item("chains complete (400 / 1000 / 1500 stored trajectories)", full, full)
c.record("split 8^4 chain (2.41, 0.02, 0): streams (first kept, stored, kept)", R[((2.41, 0.02, 0.0), 8)]["streams"])
c.item(
    "link-field equilibration <s_j^2>/2g_j over the first 50 kept trajectories (min, max), all >= 0.8",
    (min(FZ.values()), max(FZ.values())),
    min(FZ.values()) >= 0.8,
    fmt="{0[0]:.2f} .. {0[1]:.2f}",
)
EX = exponents(R)
print(
    f"  {'point':>18s} {'L':>2s} {'ntraj':>5s} {'acc':>4s} {'corner':>6s} {'chi10(p=0)':>16s} {'xi10/L':>16s} "
    f"{'chi6 cmax':>16s} {'E10':>16s} {'V<D_par^2>':>14s} {'V<|phi_st|^2>':>14s}"
)
for p in k3.P1_POINTS:
    for L in LS:
        r = R[(p, L)]
        cells = [val(r, k) for k in ("chi10", "xi10_cmaxL", "chi6_corner_max", "E10", "dimer_sq_par", "phi_stag_sq")]
        print(
            f"  {str(p):>18s} {L:2d} {r['ntraj']:5d} {r['acceptance']:4.2f} {r['chi10_corner_max']['label']:>6s} "
            + " ".join(f"{a:>10.4f}({b:.4f})" for a, b in cells)
        )
ok_corner = all(R[(p, L)]["chi10_corner_max"]["corner"] == 0 for p in k3.P1_POINTS for L in LS)
c.item("the largest of the 8 independent corners of chi10 is p = 0 at every point and L", ok_corner, ok_corner)
c.record("E10 (local plaquette energy) at L = 8", [round(R[(p, 8)]["E10"]["mean"], 3) for p in k3.P1_POINTS])
verdicts = {}
for p in k3.P1_POINTS:
    e68 = EX[(p, "chi10cmax", 6, 8)]
    x6, x8 = k3.value(R[(p, 6)], "xi10_cmaxL"), k3.value(R[(p, 8)], "xi10_cmaxL")
    verdicts[p] = k3.scaling_verdict(e68, x6, x8)
    c.record(f"{p}: chi10 (p=0) L = 4 / 6 / 8", [val(R[(p, L)], "chi10") for L in LS], fmt=None)
    c.item(
        f"{p}: largest-corner exponent (4,6) {EX[(p, 'chi10cmax', 4, 6)][0]:+.2f}({EX[(p, 'chi10cmax', 4, 6)][1]:.2f}),"
        f" (6,8) {e68[0]:+.2f}({e68[1]:.2f}); xi10/L {x6[0]:.3f}({x6[1]:.3f}) -> {x8[0]:.3f}({x8[1]:.3f}); verdict",
        verdicts[p],
        verdicts[p] == "null",
    )
c.item(
    "null certified at all five points (no (10,3,1) condensate up to L = 8)",
    verdicts,
    all(v == "null" for v in verdicts.values()),
)
ref = (2.41, 0.1, 0.1)
e1, e2 = EX[(ref, "dimer", 4, 6)], EX[(ref, "dimer", 6, 8)]
c.item(
    "positive control: V<D_par^2> at (2.41, 0.1, +0.1) exponents (4,6), (6,8) >= 0.65",
    (e1, e2),
    e1[0] >= 0.65 and e2[0] >= 0.65,
    fmt="{0[0][0]:.2f}({0[0][1]:.2f}), {0[1][0]:.2f}({0[1][1]:.2f})",
)
for p in k3.P1_POINTS:
    c.record(
        f"{p}: exponents (4,6)/(6,8) chi6 cmax, dimer, eps (not asserted)",
        " ".join(f"{n} {EX[(p, n, 4, 6)][0]:+.2f}/{EX[(p, n, 6, 8)][0]:+.2f}" for n in ("chi6cmax", "dimer", "eps")),
    )
R2, _ = table(extra=100)
EX2 = exponents(R2)
shift = max(abs(EX2[k][0] - EX[k][0]) for k in EX if k[1] == "chi10cmax")
same = all(
    k3.scaling_verdict(
        EX2[(p, "chi10cmax", 6, 8)], k3.value(R2[(p, 6)], "xi10_cmaxL"), k3.value(R2[(p, 8)], "xi10_cmaxL")
    )
    == "null"
    for p in k3.P1_POINTS
)
c.item(
    "robustness: 100 more trajectories cut from every stream: largest chi10 exponent shift, verdicts unchanged",
    shift,
    shift <= 0.03 and same,
    fmt="{:.3f}",
)
c.done()
