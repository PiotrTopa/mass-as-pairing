"""K3.2: the complete 10-channel correlator at P_c is free-like and suppressed (0.2-0.6 x free).

(1) Free theory (y = 0, s = 0, dense propagator): complete chi10 at p = 0, xi10/L and the finite-size exponents at
    L = 4, 6 (live) and 8 (stored value; K3_FREE_L8=1 recomputes it, a dense 8^4 inverse); <T_Q>_0 at 4^4.
(2) At P_c (the K3.1 ensembles and the epsilon model, exact propagators at L = 4, 6): chi10 + 2 sigma < 0.6 x free at
    every point and L, xi10/L <= free + 2 sigma, every (6,8) exponent >= the free one within 2 sigma.
(3) A same-parity flavour-blind source on one sublattice keeps det D_c > 0, but in the free theory the sourced
    sublattice has zero pair amplitude while the other one responds at O(h) (Schur complement).
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.action import build_model
from masspairing.analysis import k3
from masspairing.claimcheck import Check
from masspairing.flavour import real_flavour_block
from masspairing.lattice import Lattice
from masspairing.measure import ChannelMeasure
from masspairing.patterns import source_pattern

c = Check("K3.2")


def xi_over_L(S0, Sp, L):
    return np.sqrt(max(S0 / Sp - 1.0, 0.0)) / (2 * np.sin(np.pi / L)) / L


def free_channels(L):
    lat = Lattice((L,) * 4)
    model = build_model(lat, 0.0, -0.01, 1.0, g10=0.1, g6=0.0)
    b = ChannelMeasure(model, exact=True).bilinears(np.zeros(model.nfields))
    cr, pm = np.asarray(b["chi10_corners"]), np.asarray(b["chi10_pmin"])
    return dict(
        chi10=float(b["chi10"]),
        xi10L=float(xi_over_L(cr[0], pm[0], L)),
        E10=float(b["E10"]),
        cmax=int(np.argmax(np.abs(cr[:8]))),
    )


# ---------------------------------------------------------------- (1) free baseline
free = {L: free_channels(L) for L in ((4, 6, 8) if os.environ.get("K3_FREE_L8") else (4, 6))}
if 8 not in free:
    free[8] = dict(chi10=k3.FREE_CHI10[8][0], xi10L=k3.FREE_CHI10[8][1], E10=np.nan, cmax=0)
for L in (4, 6, 8):
    ok = abs(free[L]["chi10"] - k3.FREE_CHI10[L][0]) < 2e-3 and abs(free[L]["xi10L"] - k3.FREE_CHI10[L][1]) < 2e-3
    c.item(
        f"free L = {L}: chi10(p = 0), xi10/L (p = 0 the largest corner)"
        + (" [stored]" if L == 8 and not os.environ.get("K3_FREE_L8") else ""),
        (free[L]["chi10"], free[L]["xi10L"]),
        ok and free[L]["cmax"] == 0,
        fmt="{0[0]:.4f}, {0[1]:.4f}",
    )
e46 = np.log(free[6]["chi10"] / free[4]["chi10"]) / np.log((6 / 4) ** 4)
e68 = np.log(free[8]["chi10"] / free[6]["chi10"]) / np.log((8 / 6) ** 4)
c.item(
    "free exponents d ln chi10/d ln V on (4,6), (6,8)",
    (e46, e68),
    -0.30 < e46 < -0.24 and -0.16 < e68 < -0.10,
    fmt="{0[0]:+.3f}, {0[1]:+.3f}",
)
c.item(
    "free <T_Q>_0 on 4^4 (= the free plaquette energy 1.3075)",
    free[4]["E10"],
    abs(free[4]["E10"] - 1.3075) < 2e-3,
    fmt="{:.4f}",
)

# ---------------------------------------------------------------- (2) P_c vs free
worst, ok_xi, ok_ex = 0.0, True, True
for p in k3.P1_POINTS:
    R = {L: k3.ensemble_of(k3.p1_spec(p, L)) for L in (4, 6, 8)}
    for L in (4, 6, 8):
        cv, ce = k3.value(R[L], "chi10")
        worst = max(worst, (cv + 2 * ce) / free[L]["chi10"])
    for L in (6, 8):
        x, xe = k3.value(R[L], "xi10_cmaxL")
        ok_xi &= x <= free[L]["xi10L"] + 2 * xe + 0.005
    e, s = k3.exponent(k3.value(R[6], "chi10_corner_max"), k3.value(R[8], "chi10_corner_max"), 6, 8)
    ok_ex &= e + 2 * s >= e68 - 0.02
c.item("K3.1 points, L = 4/6/8: largest (chi10 + 2 sigma)/chi10_free (< 0.6)", worst, worst < 0.6, fmt="{:.2f}")
c.item("K3.1 points, L = 6, 8: xi10/L <= free + 2 sigma", ok_xi, ok_xi)
c.item(f"every K3.1 (6,8) exponent >= the free exponent {e68:+.2f} within 2 sigma", ok_ex, ok_ex)
for L in (4, 6):
    r = k3.ensemble_of([(f"{k3.XI}/F8_P3_eps/{k3.chain_name(L, 2.41, 0, 0)}", 0)])
    cv, ce = k3.value(r, "chi10")
    x, xe = k3.value(r, "xi10L")
    c.item(
        f"epsilon model at P_c (g10 = 0, exact), L = {L}: chi10, ratio to free, xi10/L (free {free[L]['xi10L']:.3f})",
        (cv, ce, cv / free[L]["chi10"], x),
        cv + 2 * ce < 0.6 * free[L]["chi10"] and x <= free[L]["xi10L"] + 2 * xe + 0.005,
        fmt="{0[0]:.3f}({0[1]:.3f}), {0[2]:.2f}, {0[3]:.3f}",
    )

# ---------------------------------------------------------------- (3) one-sublattice source
lat = Lattice((4, 4, 4, 4))
V = lat.V
eps4 = lat.eps_np.reshape(V)
C0 = source_pattern(lat).toarray()
odd, even = eps4 < 0, eps4 > 0
Codd = np.where(odd[:, None] & odd[None, :], C0, 0.0)
Ceven = np.where(even[:, None] & even[None, :], C0, 0.0)
worst = 0.0
for y in (2.41, 0.0):
    model = build_model(lat, y, -0.01, 1.0, g10=0.1, g6=0.0)
    rng = np.random.default_rng(7)
    for Cp in (Codd, Ceven):
        for _trial in range(4):
            Dc = model.D.dense(np.asarray(model.start(rng, "hot+hot")))
            for h in (0.005, 0.05, 0.3):
                s, _ = np.linalg.slogdet(Dc - h * np.kron(Cp, np.eye(2)))
                worst = max(worst, abs(s - 1.0))
c.item(
    "det D_c with an odd-only or even-only source (4^4, y = 2.41, 0; 8 hot configurations x 3 h): max |sign - 1|",
    worst,
    worst < 1e-10,
    fmt="{:.1e}",
)
resp = {}
for L in (4, 6):
    latL = Lattice((L,) * 4)
    VL = latL.V
    e_ = latL.eps_np.reshape(VL)
    C0L = source_pattern(latL).toarray()
    o_, ev_ = e_ < 0, e_ > 0
    Co = np.where(o_[:, None] & o_[None, :], C0L, 0.0)
    Ce = np.where(ev_[:, None] & ev_[None, :], C0L, 0.0)
    model = build_model(latL, 0.0, -0.01, 1.0, g10=0.1, g6=0.0)
    D0 = model.D.dense(np.zeros(model.nfields))
    for h in (0.01, 0.2):
        S = np.linalg.inv(D0 - h * np.kron(Co, np.eye(2))).reshape(VL, 2, VL, 2)
        trS = np.einsum("xzaa->xz", real_flavour_block(S.transpose(0, 2, 1, 3)))  # sum_a G^{aa}(x, z)
        resp[(L, h)] = (0.5 * np.sum(Co * trS) / VL / h, 0.5 * np.sum(Ce * trS) / VL / h)
c.item(
    "free theory, source on odd pairs: <Phi_odd>/h at L = 4, 6 and h = 0.01, 0.2 (identically zero)",
    max(abs(v[0]) for v in resp.values()),
    max(abs(v[0]) for v in resp.values()) < 1e-10,
    fmt="{:.1e}",
)
c.item(
    "induced <Phi_even>/h at L = 4, 6 (h = 0.01; h-independent: h = 0.2 values)",
    [round(resp[(L, h)][1], 3) for L in (4, 6) for h in (0.01, 0.2)],
    all(1.5 < v[1] < 1.9 for v in resp.values())
    and all(abs(resp[(L, 0.01)][1] - resp[(L, 0.2)][1]) < 1e-3 for L in (4, 6)),
)
c.done()
