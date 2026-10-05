#!/usr/bin/env python3
"""N.1 -- the Z4 of one Spin(10) generation in Standard-Model language, X = 5(B-L) - 4Y mod 4, and its consequences for
neutrino masses: exact charge arithmetic (fractions) and two 3x3 / 2x2 mass matrices (< 1 s)."""

import pathlib
import sys
from fractions import Fraction as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.claimcheck import Check

c = Check("N.1")

# one generation, all left-handed Weyl fields: name -> (multiplicity = colour x isospin, Y, B-L)
GEN = {
    "Q": (6, F(1, 6), F(1, 3)),
    "u^c": (3, F(-2, 3), F(-1, 3)),
    "d^c": (3, F(1, 3), F(-1, 3)),
    "L": (2, F(-1, 2), F(-1)),
    "e^c": (1, F(1), F(1)),
    "nu^c": (1, F(0), F(1)),
}


def X(Y, BL):
    return 5 * BL - 4 * Y


def Xp(Y, BL, T3):
    return X(Y, BL) + 4 * T3


def mod4(q):
    return q % 4


# 1. X on the 16 and on the Higgs
xs = {n: X(Y, BL) for n, (_, Y, BL) in GEN.items()}
c.item(
    "16 Weyl fermions per generation (with nu^c)",
    sum(m for m, _, _ in GEN.values()),
    sum(m for m, _, _ in GEN.values()) == 16,
)
c.item(
    "every field of the 16 has X = 1 mod 4 (X of Q, u^c, d^c, L, e^c, nu^c)",
    ", ".join(f"{n} {x}" for n, x in xs.items()),
    all(x.denominator == 1 and mod4(x) == 1 for x in xs.values()),
)
XH = X(F(1, 2), F(0))
c.item(
    "the SM Higgs (Y = 1/2, B-L = 0): X, X mod 4 (the generator acts as i on the 16, -1 on the 10: the Spin(10) "
    "centre)",
    (XH, mod4(XH)),
    mod4(XH) == 2,
    "{0[0]} = {0[1]} mod 4",
)
c.item(
    "every fermion bilinear has X = 2 mod 4 (every Majorana mass Z4-odd); every Yukawa psi psi H has X = 0",
    True,
    all(mod4(2 * x) == 2 for x in xs.values()) and mod4(1 + 1 + XH) == 0,
)

# 2. the anomaly count (Spin-Z4: nu = n(X = 1) - n(X = 3) = 0 mod 16, K1.4)
nu = sum(m for m, _, _ in GEN.values())
c.item(
    "anomaly count: one generation nu (mod 16); without nu^c",
    (nu % 16, nu - 1),
    nu % 16 == 0 and (nu - 1) % 16 != 0,
    "{0[0]}; {0[1]}",
)
c.item(
    "nu^c alone: nu = 1 != 0 mod 16 -- no Z4-symmetric gap for nu^c by itself (anomaly matching)", 1 % 16, 1 % 16 != 0
)

# 3. after electroweak breaking
c.item(
    "the neutral Higgs component (T3 = -1/2, Y = 1/2) has X' = X + 4 T3 = 0 mod 4: <H> preserves Z4'",
    mod4(Xp(F(1, 2), F(0), F(-1, 2))),
    mod4(Xp(F(1, 2), F(0), F(-1, 2))) == 0,
    "{}",
)
comps = [
    ("nu_L", F(-1, 2), F(-1), F(1, 2)),
    ("e_L", F(-1, 2), F(-1), F(-1, 2)),
    ("nu^c", F(0), F(1), F(0)),
    ("u_L", F(1, 6), F(1, 3), F(1, 2)),
    ("d_L", F(1, 6), F(1, 3), F(-1, 2)),
    ("u^c", F(-2, 3), F(-1, 3), F(0)),
    ("d^c", F(1, 3), F(-1, 3), F(0)),
    ("e^c", F(1), F(1), F(0)),
]
xp = {n: mod4(Xp(Y, BL, T3)) for n, Y, BL, T3 in comps}
c.item(
    "every fermion keeps an odd X' (Z4'^2 = (-1)^F)",
    ", ".join(f"{n} {v}" for n, v in xp.items()),
    all(v % 2 == 1 for v in xp.values()),
)
c.item(
    "X' = 5(B-L) - 4Q mod 4 on every component",
    True,
    all(mod4(Xp(Y, BL, T3) - (5 * BL - 4 * (T3 + Y))) == 0 for _, Y, BL, T3 in comps),
)

# 4. selection rules with Z4' exact
nuL, nuc = (F(-1, 2), F(-1), F(1, 2)), (F(0), F(1), F(0))
c.item(
    "nu_L nu_L (Majorana, incl. the Weinberg operator): X' mod 4 -- forbidden",
    mod4(2 * Xp(*nuL)),
    mod4(2 * Xp(*nuL)) == 2,
    "{}",
)
c.item("nu^c nu^c (Majorana M_R): X' mod 4 -- forbidden", mod4(2 * Xp(*nuc)), mod4(2 * Xp(*nuc)) == 2, "{}")
c.item("nu_L nu^c (Dirac): X' mod 4 -- allowed", mod4(Xp(*nuL) + Xp(*nuc)), mod4(Xp(*nuL) + Xp(*nuc)) == 0, "{}")
c.item(
    "every |Delta L| = 2, Delta B = Delta Q = 0 amplitude (0vbb) has Delta X' = 2 mod 4 -- vanishes while Z4' is exact",
    True,
    all(mod4(5 * dBL) == 2 for dBL in (F(2), F(-2))),
)
c.item("sphalerons (Delta B = Delta L): Delta X' = 5 Delta(B-L) = 0 -- compatible", True, mod4(5 * (F(3) - F(3))) == 0)
c.item(
    "a condensate that permits M_R carries B-L = +-2 with Q = 0 (X' = 2): the 126-like Delta_R",
    True,
    mod4(5 * F(2)) == 2 and mod4(X(F(1), F(2))) == 2 and mod4(X(F(0), F(2))) == 2,
)

# 5. neutral-fermion counting (mass terms pair X' = 1 with X' = 3 only: bipartite mass matrix)
c.item(
    "nu_L has X' = 3, nu^c has X' = 1: a Dirac pair is allowed",
    (xp["nu_L"], xp["nu^c"]),
    {xp["nu_L"], xp["nu^c"]} == {1, 3},
    "{0[0]}, {0[1]}",
)
mD, M = 0.1, 100.0
Mmat = np.array([[0, 0, mD], [0, 0, M], [mD, M, 0]], float)  # basis (nu_L, S | nu^c), X' = 3, 3 | 1
sv = np.sort(np.abs(np.linalg.eigvalsh(Mmat)))
c.item(
    "nu_L + S (X' = 3) against nu^c (X' = 1), m_D = 0.1, M = 100: rank <= 1, one exactly massless neutral fermion "
    "(|eigenvalues|)",
    tuple(sv),
    sv[0] < 1e-12 and abs(sv[1] - np.hypot(mD, M)) < 1e-9,
    "{0[0]:.0e}, {0[1]:.4g}, {0[2]:.4g}",
)
Ms = np.array([[0, mD], [mD, M]], float)
light = np.sort(np.abs(np.linalg.eigvalsh(Ms)))[0]
c.item(
    "control: with the Z4'-odd entry M nu^c nu^c the light eigenvalue is the seesaw m_D^2/M (relative deviation)",
    abs(light - mD**2 / M) / (mD**2 / M),
    abs(light - mD**2 / M) / (mD**2 / M) < 1e-3,
    "{:.1e}",
)
c.done()
