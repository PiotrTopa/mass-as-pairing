"""Zolotarev optimal rational approximation to x^(-1/2) on [1, b] and the partial fractions of the RHMC forces.

The plain construction (``zolotarev_invsqrt``) is exact to rounding for windows b = hi/lo up to about 10^8; beyond
that its residues overflow. ``zolotarev_invsqrt_robust`` returns the same approximation built from the complementary
parameter m1 = 1/b (theta series in the small nome, reflection c_l c_{2n+1-l} = b), accurate on windows up to 1e13.
The ``*_pf`` helpers use the plain construction and fall back to the robust one only where the plain one is not
finite, so the plain numbers are unchanged wherever they exist.
"""

import numpy as np
from scipy.special import ellipj, ellipk


def zolotarev_invsqrt(n, b):
    """Degree-(n,n) Zolotarev approximation R(x) = d0 Π_l (x+c_{2l})/(x+c_{2l-1}) ≈ x^{-1/2}, x∈[1,b].

    Returns dict with d0, poles p_l = c_{2l-1}, zeros z_l = c_{2l}, residues r_l such that
    R(x) = d0 (1 + Σ_l r_l/(x + p_l)), and the max relative error on [1,b].
    """
    m = 1.0 - 1.0 / b  # parameter m = κ'² with κ' = sqrt(1 − 1/b)
    K = ellipk(m)
    l = np.arange(1, 2 * n + 1)
    sn, cn, dn, _ = ellipj(l * K / (2 * n + 1), m)
    c = sn**2 / cn**2
    p, z = c[0::2], c[1::2]  # c_{odd} poles, c_{even} zeros
    x = np.exp(np.linspace(0, np.log(b), 20001))
    Rt = np.prod((x[:, None] + z) / (x[:, None] + p), 1) * np.sqrt(x)
    d0 = 2.0 / (Rt.max() + Rt.min())
    err = np.max(np.abs(d0 * Rt - 1.0))
    r = np.array([np.prod(z - p[i]) / np.prod(np.delete(p, i) - p[i]) for i in range(n)])
    return dict(n=n, b=b, d0=d0, poles=p, zeros=z, residues=r, maxerr=err)


def invsqrt_partial_fractions(n, lam_min, lam_max):
    """A^{-1/2} ≈ c0 + Σ_l rho_l (A + P_l)^{-1} valid for spec(A) ⊂ [lam_min, lam_max]."""
    zz = zolotarev_invsqrt(n, lam_max / lam_min)
    s = lam_min**-0.5
    return dict(
        c0=s * zz["d0"],
        rho=s * zz["d0"] * zz["residues"] * lam_min,
        poles=zz["poles"] * lam_min,
        maxerr=zz["maxerr"],
        n=n,
    )


def eval_partial_fractions(pf, x):
    x = np.asarray(x, float)
    return pf["c0"] + np.sum(pf["rho"][None, :] / (x[:, None] + pf["poles"][None, :]), 1)


# ---- partial fractions for the Hasenbusch terms ---------------------------------------------------
def shifted_invsqrt_partial_fractions(n, b, lam_min, lam_max):
    """(A + b)^{-1/2} ≈ c0 + Σ_l rho_l (A + P_l)^{-1}, P_l > b, valid for spec(A) ⊂ [lam_min, lam_max]."""
    pf = invsqrt_partial_fractions(n, lam_min + b, lam_max + b)
    pf["poles"] = pf["poles"] + b
    pf["shift"] = b
    return pf


def ratio_sqrt_partial_fractions(n, a, b, lam_min, lam_max):
    """g(A) = [(A + b)/(A + a)]^{1/2} ≈ c0 + Σ_l rho_l (A + s_l)^{-1}, a < s_l < b, spec(A) ⊂ [lam_min, lam_max].

    With r(t) = (t+a)/(t+b) ∈ [r_min, r_max] ⊂ (0, 1] and x = r/r_min ∈ [1, B], g = r^{-1/2} = r_min^{-1/2} x^{-1/2};
    the Zolotarev x^{-1/2} ≈ d0(1 + Σ res_l/(x + p_l)) is mapped back with 1/(r + q) = (t+b)/((1+q)(t + s)),
    s = (a + q b)/(1 + q), q = r_min p_l, so every pole lands strictly between a and b.  The maximal relative
    error is the Zolotarev one on [1, B], B = r_max/r_min ≈ (lam_min + b)/(lam_min + a) for lam_max ≫ b.
    """
    assert 0.0 <= a < b
    r_min = (lam_min + a) / (lam_min + b)
    r_max = (lam_max + a) / (lam_max + b)
    B = r_max / r_min
    zz = zolotarev_invsqrt(n, B)
    q = r_min * zz["poles"]
    s = (a + q * b) / (1.0 + q)
    pref = r_min**-0.5 * zz["d0"]
    c0 = pref * (1.0 + np.sum(zz["residues"] * r_min / (1.0 + q)))
    rho = pref * zz["residues"] * r_min * (b - a) / (1.0 + q) ** 2
    return dict(c0=c0, rho=rho, poles=s, maxerr=zz["maxerr"], n=n, a=a, b=b)


# ---- robust construction for wide windows ------------------------------------------------------------
# The plain construction fails on windows b = hi/lo above about 2.5e8: (i) the residues are ratios of two
# products of
# n factors as large as 40 b, which overflow once n log10(40 b) > 308; (ii) the positions c_l = sn^2/cn^2 come
# from
# ellipj at m = 1 - 1/b, which carries 1/b only to a relative 1e-16 b. The construction below avoids both in
# double
# precision: the residues are products of n ratios of order one (poles and zeros interlace), and everything is
# a
# function of the complementary parameter m1 = 1/b: sc(u|m) = -i sn(iu|m1) is summed from the theta series in
# the
# nome of the small parameter for u <= K/2, and the upper half follows from the reflection c_l c_{2n+1-l} = b.
def _sc2_small_m1(u, m1, K, K1, kmax=16):
    """sc²(u | m = 1 − m1) for 0 < u ≤ K/2 from the theta series in the nome of m1 (relative accuracy ≈ 1e-15)."""
    q = np.exp(-np.pi * K / K1)
    y = np.pi * np.asarray(u, float) / (2.0 * K1)
    k = np.arange(kmax)
    sgn = (-1.0) ** k
    with np.errstate(under="ignore"):
        qa = q ** (k * (k + 1.0))  # θ1, θ2 weights
        qb = q ** (k * k * 1.0)  # θ3, θ4 weights
        S1 = np.sum(sgn * qa * np.sinh((2 * k + 1.0) * y[:, None]), 1)
        T4 = 1.0 + 2.0 * np.sum((sgn * qb)[1:] * np.cosh(2.0 * k[1:] * y[:, None]), 1)
    th2 = np.sum(qa)  # θ2(0)/(2 q^{1/4})
    th3 = 1.0 + 2.0 * np.sum(qb[1:])
    return (th3 * S1 / (th2 * T4)) ** 2


def zolotarev_invsqrt_robust(n, b):
    """Same approximation and same return value as `zolotarev_invsqrt` (degree (n, n), x ∈ [1, b]), built so that it
    stays accurate to rounding on wide windows (tested for b = 10 ... 1e13, n <= 80)."""
    from scipy.special import ellipkm1

    n = int(n)
    b = float(b)
    m1 = 1.0 / b
    K, K1 = float(ellipkm1(m1)), float(ellipk(m1))  # K(m), K(m1) with m = 1 − m1
    l = np.arange(1, n + 1)  # u_l = l K/(2n+1) ≤ K/2
    low = _sc2_small_m1(l * K / (2 * n + 1), m1, K, K1)  # c_1 … c_n
    c = np.concatenate([low, (b / low)[::-1]])  # c_{2n+1−l} = b / c_l
    p, z = c[0::2], c[1::2]
    x = np.exp(np.linspace(0, np.log(b), 20001))
    x[-1] = b
    Rt = np.prod((x[:, None] + z) / (x[:, None] + p), 1) * np.sqrt(x)
    d0 = 2.0 / (Rt.max() + Rt.min())
    err = np.max(np.abs(d0 * Rt - 1.0))
    r = np.empty(n)
    for i in range(n):
        o = np.arange(n) != i
        r[i] = (z[i] - p[i]) * np.prod((z[o] - p[i]) / (p[o] - p[i]))
    return dict(n=n, b=b, d0=d0, poles=p, zeros=z, residues=r, maxerr=err)


def invsqrt_partial_fractions_robust(n, lam_min, lam_max):
    zz = zolotarev_invsqrt_robust(n, lam_max / lam_min)
    s = lam_min**-0.5
    return dict(
        c0=s * zz["d0"],
        rho=s * zz["d0"] * zz["residues"] * lam_min,
        poles=zz["poles"] * lam_min,
        maxerr=zz["maxerr"],
        n=n,
    )


def shifted_invsqrt_partial_fractions_robust(n, b, lam_min, lam_max):
    pf = invsqrt_partial_fractions_robust(n, lam_min + b, lam_max + b)
    pf["poles"] = pf["poles"] + b
    pf["shift"] = b
    return pf


def ratio_sqrt_partial_fractions_robust(n, a, b, lam_min, lam_max):
    """`ratio_sqrt_partial_fractions` on the robust Zolotarev coefficients (same mapping)."""
    assert 0.0 <= a < b
    r_min = (lam_min + a) / (lam_min + b)
    r_max = (lam_max + a) / (lam_max + b)
    zz = zolotarev_invsqrt_robust(n, r_max / r_min)
    q = r_min * zz["poles"]
    s = (a + q * b) / (1.0 + q)
    pref = r_min**-0.5 * zz["d0"]
    c0 = pref * (1.0 + np.sum(zz["residues"] * r_min / (1.0 + q)))
    rho = pref * zz["residues"] * r_min * (b - a) / (1.0 + q) ** 2
    return dict(c0=c0, rho=rho, poles=s, maxerr=zz["maxerr"], n=n, a=a, b=b)


# ---- the force fractions of the RHMC
# --------------------------------------------------------------------------
def _finite(pf):
    return bool(np.isfinite(pf["c0"]) and np.all(np.isfinite(pf["rho"])) and np.all(np.isfinite(pf["poles"])))


def force_pf(kind, n, lo, hi, a=None, b=None):
    """Partial fractions of a force term on the window [lo, hi]; the robust construction where the plain is not finite.

    kind: "invsqrt" (t^-1/2), "shifted" ((t + a)^-1/2), "ratio" (((t + b)/(t + a))^1/2).
    """
    plain = {
        "invsqrt": lambda: invsqrt_partial_fractions(n, lo, hi),
        "shifted": lambda: shifted_invsqrt_partial_fractions(n, a, lo, hi),
        "ratio": lambda: ratio_sqrt_partial_fractions(n, a, b, lo, hi),
    }[kind]
    robust = {
        "invsqrt": lambda: invsqrt_partial_fractions_robust(n, lo, hi),
        "shifted": lambda: shifted_invsqrt_partial_fractions_robust(n, a, lo, hi),
        "ratio": lambda: ratio_sqrt_partial_fractions_robust(n, a, b, lo, hi),
    }[kind]
    with np.errstate(all="ignore"):
        pf = plain()
    if _finite(pf):
        return pf
    pf = robust()
    if not _finite(pf):
        raise ValueError(f"force partial fraction {kind} on [{lo:.3g}, {hi:.3g}] is not finite (n = {n})")
    return pf
