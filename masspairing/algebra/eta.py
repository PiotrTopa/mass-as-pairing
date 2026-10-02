"""eta invariants of twisted Dirac operators on the lens spaces S^5/Z_n and the Dai-Freed anomaly of 4d Weyl fermions
with Spin x Z_n or Spin-Z_{2m} symmetry.

Lens space: Z_n acts on C^3 (S^5) by z_j -> exp(2 pi i a_j / n) z_j. The rotation g by angles theta_j = 2 pi a_j / n
lifts
to g~ in Spin(6) with half-angles theta_j / 2, and g~^n = (-1)^3 = -1.

* Spin x Z_n (n odd): the lift of order n is -g~; a charge-s fermion sees (-g~)^l (x) exp(2 pi i l s / n).
* Spin-Z_{2m} = (Spin x Z_{2m}) / Z_2: a structure on S^5/Z_n is a pair [(g~, h)], h in Z_{2m}, of order n, i.e.
  n h = m mod 2m; a fermion of odd charge q sees g~^l (x) exp(2 pi i l h q / (2m)).

Equivariant APS on the ball B^6 with a positive-scalar-curvature metric (no harmonic spinors) gives, for g != 1,
eta_g(S^5) = 2 L(g) with the Atiyah-Bott fixed-point number L(g) = prod_j i / (2 sin(theta_j / 2)); then

    eta(S^5/Z_n, chi) = (1/n) sum_{l=1}^{n-1} chi(l) 2 L(g^l),   xi = eta / 2,

and the anomaly phase of nu Weyl fermions is exp(-2 pi i nu xi). ``eta_spectral`` recomputes eta_g(S^5) independently
from the Dirac spectrum of the round S^5 (eigenvalues +-(k + 5/2), eigenspaces the Spin(6) irreps
(k + 1/2, 1/2, +-1/2)) with zeta regularisation.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from math import comb, lcm

import numpy as np


def lefschetz(half_angles):
    """Atiyah-Bott Dirac fixed-point number prod_j i / (2 sin phi_j), phi_j = theta_j / 2."""
    out = 1.0 + 0j
    for p in half_angles:
        out *= 1j / (2 * np.sin(p))
    return out


def xi_lens(n, twist, a=(1, 1, 1)):
    """xi = eta / 2 on S^5/Z_n with the character twist(l) multiplying g~^l (imaginary part asserted ~0)."""
    tot = 0j
    for l in range(1, n):
        phi = [np.pi * l * aj / n for aj in a]
        tot += twist(l) * lefschetz(phi)
    tot /= n
    assert abs(tot.imag) < 1e-12, tot
    return tot.real


def xi_spin_zn(n, s, a=(1, 1, 1)):
    """Spin x Z_n (n odd), charge s."""
    assert n % 2 == 1
    return xi_lens(n, lambda l: (-1) ** l * np.exp(2j * np.pi * l * s / n), a)


def spin_z2m_structures(n, m):
    """The h in Z_{2m} (mod m, since (g~, h) ~ (-g~, h + m)) with n h = m mod 2m."""
    return [h for h in range(m) if (n * h - m) % (2 * m) == 0]


def xi_spin_z2m(n, m, h, q, a=(1, 1, 1)):
    """Spin-Z_{2m} structure h on S^5/Z_n, fermion of odd charge q."""
    assert q % 2 == 1 and (n * h - m) % (2 * m) == 0
    return xi_lens(n, lambda l: np.exp(2j * np.pi * l * h * q / (2 * m)), a)


def as_fraction(x, maxden=4096):
    f = Fraction(x).limit_denominator(maxden)
    assert abs(float(f) - x) < 1e-9, (x, f)
    return f


def order_mod1(x, maxden=4096):
    """Smallest nu >= 1 with nu x integer."""
    return as_fraction(x, maxden).denominator


# ---- anomaly tables ----------------------------------------------------------------------------------
def anomaly_table(m, q=1, nus=(16, 32, 48)):
    """Rows (n, h, xi as an exact fraction, {nu: nu xi mod 1}) over the lens spaces S^5/Z_n,
    n <= 4m, for Spin-Z_{2m}."""
    rows = []
    for n in range(1, 4 * m + 1):
        for h in spin_z2m_structures(n, m):
            f = as_fraction(xi_spin_z2m(n, m, h, q))
            rows.append((n, h, f, {nu: (nu * f) % 1 for nu in nus}))
    return rows


def min_anomaly_free_nu(m, q):
    """Minimal number of charge-q Weyl fermions anomaly-free on every lens space S^5/Z_n (n <= 4m), Spin-Z_{2m}."""
    o = 1
    for n in range(1, 4 * m + 1):
        for h in spin_z2m_structures(n, m):
            o = lcm(o, order_mod1(xi_spin_z2m(n, m, h, q)))
    return o


def hsieh_min_nu(m, q):
    """Minimal nu from the anomaly conditions (1.3) of Hsieh, arXiv:1808.02881, for nu fermions of odd charge q:
    (2m^2 + m + 1) s3 - (m + 3) s1 = 0 mod 48 m and m s3 + s1 = 0 mod 2m, s3 = nu q^3, s1 = nu q."""
    for nu in range(1, 100000):
        s3, s1 = nu * q**3, nu * q
        if ((2 * m * m + m + 1) * s3 - (m + 3) * s1) % (48 * m) == 0 and (m * s3 + s1) % (2 * m) == 0:
            return nu
    raise ValueError("no solution below 100000")


# ---- spectral cross-check --------------------------------------------------------------------------------
def _sym_char(xs, kmax):
    """Characters h_k of Sym^k of a representation with eigenvalues xs, k = 0..kmax (Newton)."""
    xs = np.array(xs, complex)
    p = [None] + [np.sum(xs**j) for j in range(1, kmax + 1)]
    h = [1.0 + 0j]
    for k in range(1, kmax + 1):
        h.append(sum(p[j] * h[k - j] for j in range(1, k + 1)) / k)
    return h


def dirac_eigen_characters(half_angles, kmax):
    """Traces of g~ on the Dirac eigenspaces of round S^5: A_k for +(k + 5/2), B_k for -(k + 5/2), k = 0..kmax.

    From H_k (x) S^+- = A_k + B_{k-1} (resp. B_k + A_{k-1}), H_k the harmonic polynomials of degree k."""
    xs = [np.exp(2j * p) for p in half_angles] + [np.exp(-2j * p) for p in half_angles]
    h = _sym_char(xs, kmax)
    H = [h[k] - (h[k - 2] if k >= 2 else 0) for k in range(kmax + 1)]
    Sp = sum(
        np.exp(1j * sum(e * p for e, p in zip(eps, half_angles, strict=True)))
        for eps in itertools.product((1, -1), repeat=3)
        if np.prod(eps) == 1
    )
    Sm = sum(
        np.exp(1j * sum(e * p for e, p in zip(eps, half_angles, strict=True)))
        for eps in itertools.product((1, -1), repeat=3)
        if np.prod(eps) == -1
    )
    A, B = [], []
    for k in range(kmax + 1):
        A.append(H[k] * Sp - (B[k - 1] if k else 0))
        B.append(H[k] * Sm - (A[k - 1] if k else 0))
    return np.array(A), np.array(B)


def _bernoulli_poly(j, x):
    """B_j(x) = sum_i C(j, i) B_i x^(j-i) (Bernoulli numbers with B_1 = -1/2), exact."""
    B = [Fraction(1)]
    for mth in range(1, j + 1):
        B.append(-sum(comb(mth + 1, i) * B[i] for i in range(mth)) / (mth + 1))
    return sum(comb(j, i) * B[i] * Fraction(x) ** (j - i) for i in range(j + 1))


def _reg_sum(j, z, c):
    """Zeta-regularised sum_{k>=0} (k + c)^j z^k: Hurwitz -B_{j+1}(c)/(j+1) at z = 1, and the Abel/Lerch value
    (z d/dz + c)^j [1/(1 - z)] on |z| = 1, z != 1."""
    if abs(z - 1) < 1e-12:
        return complex(-_bernoulli_poly(j + 1, c) / (j + 1))
    u = 1 / (1 - z)
    coef = {1: 1.0}  # polynomial in u; z d/dz u^p = p (u^(p+1) - u^p)
    for _ in range(j):
        new = {}
        for p, a in coef.items():
            new[p + 1] = new.get(p + 1, 0) + p * a
            new[p] = new.get(p, 0) + (c - p) * a
        coef = new
    return sum(a * u**p for p, a in coef.items())


def eta_spectral(n, l, a=(1, 1, 1), kmax=400, deg=6):
    """(eta_{g^l}(S^5) from the spectrum, fit residual): sum_k [A_k - B_k] (k + 5/2)^(-s) at s = 0.

    The k-dependence of A_k - B_k is resolved into sum_r P_r(k) zeta^(r k), zeta = exp(i pi / n), deg P_r <= deg, by a
    least-squares fit on the first half of the k range (the residual is checked on all of it), then regularised
    termwise."""
    half = [np.pi * l * aj / n for aj in a]
    A, B = dirac_eigen_characters(half, kmax)
    F = A - B
    N = 2 * n
    zeta = np.exp(1j * np.pi / n)
    ks = np.arange(kmax + 1)
    cols, labels = [], []
    for r in range(N):
        for j in range(deg + 1):
            cols.append((ks / kmax) ** j * zeta ** (r * ks))
            labels.append((r, j))
    M = np.array(cols).T
    fit_k = ks[: kmax // 2]
    coef, *_ = np.linalg.lstsq(M[fit_k], F[fit_k], rcond=None)
    resid = np.max(np.abs(M @ coef - F)) / np.max(np.abs(F))
    assert resid < 1e-7, resid
    c = 2.5
    total = 0j
    for (r, j), cf in zip(labels, coef, strict=True):
        if abs(cf) < 1e-9:
            continue
        for i in range(j + 1):  # (k/kmax)^j = kmax^-j sum_i C(j, i) (k + c)^i (-c)^(j-i)
            total += cf * kmax ** (-j) * comb(j, i) * (-c) ** (j - i) * _reg_sum(i, zeta**r, c)
    return total, resid
