"""K1.4: Z_4 is the unique anomaly-free remnant of U(1)_psi for one generation: Dai-Freed anomalies from eta invariants
on lens spaces (Spin-Z_4 free, Spin-Z_8 and Spin-Z_16 anomalous for 16 Weyl fermions of charge 1) and the instanton
vertex psi^4 (mixed Z_n - Spin(10)^2 anomaly unless n | 4)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from fractions import Fraction

import numpy as np

from masspairing.algebra.characters import branch_d5_to_ps, decompose, pretty, spinor16, vector10, weyl_dim
from masspairing.algebra.eta import (
    anomaly_table,
    as_fraction,
    eta_spectral,
    hsieh_min_nu,
    lefschetz,
    min_anomaly_free_nu,
    order_mod1,
    xi_spin_z2m,
    xi_spin_zn,
)
from masspairing.algebra.spin10 import gammas, weyl_basis
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers

c = Check("K1.4")

# ---- (0) the fixed-point formula eta_g(S^5) = 2 L(g) against the zeta-regularised Dirac spectrum of S^5
worst = 0.0
for n, l in [(2, 1), (3, 1), (4, 1), (4, 2), (4, 3), (8, 1), (8, 3), (8, 5)]:
    es, res = eta_spectral(n, l)
    ed = 2 * lefschetz([np.pi * l / n] * 3)
    worst = max(worst, abs(es - ed))
    c.record(f"eta_g(S^5) n={n} l={l}: spectral, 2 L(g), fit residual", f"{es:.8f}, {ed:.8f}, {res:.1e}")
c.item("max |spectral - fixed-point| over 8 rotations (fit-limited)", worst, worst < 1e-4, fmt="{:.1e}")

# ---- (1) controls
x = as_fraction(xi_spin_z2m(2, 2, 1, 1))
c.item("control Spin-Z4 on RP^5 (h = 1), charge 1: xi", str(x), x in (Fraction(1, 16), Fraction(-1, 16)))
o3, o5 = order_mod1(xi_spin_zn(3, 1)), order_mod1(xi_spin_zn(5, 1))
c.item("control Spin x Z3 and Spin x Z5, charge 1: order of xi", (o3, o5), (o3, o5) == (9, 5))

# ---- (2) the anomaly table for nu = 16, 32, 48 Weyl fermions of charge 1
verdict = {}
for m in (2, 4, 8):
    rows = anomaly_table(m)
    for n, h, f, per in rows:
        c.record(f"Spin-Z{2 * m} S^5/Z_{n} h={h}: xi = {f}", f"nu xi mod 1 = {per[16]}, {per[32]}, {per[48]}")
    verdict[m] = {nu: all(per[nu] == 0 for _, _, _, per in rows) for nu in (16, 32, 48)}
c.item("Spin-Z4 anomaly-free for nu = 16, 32, 48", verdict[2], verdict[2] == {16: True, 32: True, 48: True})
c.item("Spin-Z8 anomaly-free for nu = 16, 32, 48", verdict[4], verdict[4] == {16: False, 32: True, 48: False})
c.item("Spin-Z16 anomaly-free for nu = 16, 32, 48", verdict[8], verdict[8] == {16: False, 32: False, 48: False})
x4 = [as_fraction(xi_spin_z2m(4, 4, h, 1)) for h in (1, 3)]
ph = [np.exp(-2j * np.pi * 16 * float(f)) for f in x4]
c.item(
    "Spin-Z8 on S^5/Z_4 (h = 1, 3): xi and the phase of nu = 16",
    ([str(f) for f in x4], np.round(ph, 12)),
    x4 == [Fraction(5, 32), Fraction(3, 32)] and np.allclose(ph, -1),
)
x8 = as_fraction(xi_spin_z2m(8, 8, 1, 1))
ph8 = [np.exp(-2j * np.pi * nu * float(x8)) for nu in (16, 32, 48)]
c.item(
    "Spin-Z16 on S^5/Z_8 (h = 1): xi and the phases of nu = 16, 32, 48",
    (str(x8), np.round(ph8, 12)),
    x8 == Fraction(21, 64) and np.allclose(ph8, [-1j, -1, 1j]),
)

# ---- (3) minimal anomaly-free nu for every odd charge against Hsieh's conditions (1.3)
agree = []
for m in (2, 4, 8):
    for q in range(1, 2 * m, 2):
        a, b = min_anomaly_free_nu(m, q), hsieh_min_nu(m, q)
        agree.append(a == b)
        c.record(f"minimal anomaly-free nu, Spin-Z{2 * m}, charge {q}: lens spaces, Hsieh (1.3)", (a, b))
c.item("lens-space minimal nu = Hsieh (1.3) for all odd charges, m = 2, 4, 8", f"{sum(agree)}/{len(agree)}", all(agree))

# ---- (4) instanton vertex: T(16) = 2 T(10) and 4 SU(2) doublets in the 16
G = gammas()
P = weyl_basis(G)
J16 = P.conj().T @ (-0.5j * G[0] @ G[1]) @ P
J10 = np.zeros((10, 10), complex)
J10[0, 1], J10[1, 0] = -1j, 1j
T16, T10 = np.trace(J16 @ J16).real, np.trace(J10 @ J10).real
c.item(
    "Tr J^2 on the 16 and on the 10 (one rotation generator): T(16)/T(10)",
    (float(T16), float(T10), float(T16 / T10)),
    abs(T16 / T10 - 2) < 1e-12,
)


def su2_1_zero_modes(dec):
    """sum over constituents of dim(SU(4) irrep) dim(SU(2)_2 irrep) x [SU(2)_1 index (2j)(2j+1)(2j+2)/6, 2j = l[3]]."""
    return sum(
        m * weyl_dim(lv[:3], 0, 3) * (lv[4] + 1) * (lv[3] * (lv[3] + 1) * (lv[3] + 2) // 6) for lv, m in dec.items()
    )


d16 = decompose(branch_d5_to_ps(spinor16(-1)), 2, 3)
d10 = decompose(branch_d5_to_ps(vector10()), 2, 3)
z16, z10 = su2_1_zero_modes(d16), su2_1_zero_modes(d10)
c.record("16 under SU(4) x SU(2)_1 x SU(2)_2", pretty(d16, 2, 3))
c.item("zero modes of a minimal SU(2)_1 instanton: 16, 10", (z16, z10), (z16, z10) == (4, 2))
allowed = [n for n in (1, 2, 4, 8, 16) if 4 % n == 0]
c.item("Z_n in U(1)_psi with the 't Hooft vertex psi^4 (charge 4) anomaly-free for n in", allowed, allowed == [1, 2, 4])
c.done()
