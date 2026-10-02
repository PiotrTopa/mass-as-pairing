"""Finite-size-scaling estimators of K7.1 and the walking-versus-power-law discrimination mock of K7.4.

K7.1: the RG-invariant ratios of the staggered scalar channel at fixed kappa,

    R4 = <m^4> / <m^2>^2        (5/3 for Gaussian 3-vector fluctuations, 1 when ordered),  m^2 = |Sigma_stag|^2
    xi_2,stag / L              (second-moment length from S(pi) and S(pi + p_min)),

per chain over all stored trajectories (no cut), blocked jackknife with block = ceil(2 tau_int(m^2)).

K7.4: can xi(y) on a grid of |y - y_c| tell walking, xi = a exp(c/sqrt(t)), from a power law, xi = a t^-nu? The true
curve (matched dynamic range, xi(t_max) = 1.5, nu = 0.61) is sampled at the grid points with relative errors eps; only
points with xi <= L_max/3 are usable; the wrong model is fitted with (log a, exponent) and optionally a shift of y_c;
its minimum chi^2 is the expected Delta chi^2 (Gaussian errors, no corrections to scaling: an optimistic bound).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .stats import jackknife, tau_int

# ---------------------------------------------------------------------------------------------- K7.1
# the analysed prefix (trajectories) of each kappa = -0.01 chain: L -> {y: n}
K71_PREFIX = {
    6: {2.028: 1500, 2.283: 1500, 2.368: 1500, 2.41: 1300, 2.452: 1200, 2.537: 1500, 2.792: 1500},
    8: {2.028: 1000, 2.283: 100, 2.41: 900, 2.537: 900, 2.792: 1000},
}


def ratio_row(L, Sigma_stag, S_pi, S_pi_pmin):
    """R4, xi_2,stag/L and <|Sigma_stag|> of one chain (all trajectories), with tau_int(m^2) and its window flag."""
    m2 = (np.asarray(Sigma_stag, float) ** 2).sum(1)
    Spi, Spm = np.asarray(S_pi, float), np.asarray(S_pi_pmin, float)
    tau, _, ok = tau_int(m2)
    bl = max(1, int(np.ceil(2 * tau)))
    R4, eR4 = jackknife(m2, lambda a: np.mean(a**2) / np.mean(a) ** 2, blen=bl)
    xiL, exiL = jackknife(
        np.stack([Spi, Spm], 1),
        lambda a: np.sqrt(max(np.mean(a[:, 0]) / np.mean(a[:, 1]) - 1.0, 0.0) / (4 * np.sin(np.pi / L) ** 2)) / L,
        blen=bl,
    )
    mabs, emabs = jackknife(np.sqrt(m2), np.mean, blen=bl)
    return dict(
        L=L,
        n=len(m2),
        tau=float(tau),
        ok=bool(ok),
        R4=float(R4),
        eR4=float(eR4),
        xiL=float(xiL),
        exiL=float(exiL),
        mabs=float(mabs),
        emabs=float(emabs),
    )


def slope_ratio(a6, c6, a8, c8, dy):
    """1/nu_eff = ln(s_8/s_6)/ln(8/6) from the slopes s_L = (xi/L(y*) - xi/L(y* + dy))/dy of independent chains."""
    s6 = (a6["xiL"] - c6["xiL"]) / dy
    s8 = (a8["xiL"] - c8["xiL"]) / dy
    es6 = np.hypot(a6["exiL"], c6["exiL"]) / dy
    es8 = np.hypot(a8["exiL"], c8["exiL"]) / dy
    inv_nu = np.log(s8 / s6) / np.log(8 / 6)
    e_inv_nu = np.hypot(es8 / s8, es6 / s6) / np.log(8 / 6)
    return dict(s6=s6, es6=es6, s8=s8, es8=es8, inv_nu=inv_nu, e_inv_nu=e_inv_nu)


# ---------------------------------------------------------------------------------------------- K7.4
NU = 0.61  # power-law exponent of the mock (SU(2) + staggered SMG value)
XI_MAX_T = 1.5  # xi at the farthest grid point t_max
GRIDS = {
    "pilot (t = 0.042, 0.127, 0.38)": np.array([0.0424, 0.1273, 0.3818]),
    "7 log-spaced in [0.02, 0.4]": np.exp(np.linspace(np.log(0.02), np.log(0.4), 7)),
    "7 log-spaced in [0.01, 0.4]": np.exp(np.linspace(np.log(0.01), np.log(0.4), 7)),
    "9 log-spaced in [0.003, 0.4]": np.exp(np.linspace(np.log(0.003), np.log(0.4), 9)),
    "9 log-spaced in [0.001, 0.4]": np.exp(np.linspace(np.log(0.001), np.log(0.4), 9)),
}
L_MAX = (8, 12, 16, 24)
EPS = (0.01, 0.03, 0.05)


def curves(t_min, t_max, nu=NU):
    """Power law and walking curves with the same xi(t_max) and the same dynamic range over [t_min, t_max]."""
    R = (t_max / t_min) ** nu
    aP = XI_MAX_T * t_max**nu
    c = np.log(R) / (1 / np.sqrt(t_min) - 1 / np.sqrt(t_max))
    aW = XI_MAX_T * np.exp(-c / np.sqrt(t_max))
    return (lambda t: aP * t**-nu), (lambda t: aW * np.exp(c / np.sqrt(t))), dict(R=R, c=c, nu=nu)


def power_model(t, p):
    return np.exp(p[0]) * t ** -p[1]


def walking_model(t, p):
    return np.exp(p[0]) * np.exp(p[1] / np.sqrt(t))


def wrong_model_chi2(model, t, xi, eps, free_shift):
    """Minimum chi^2 of ``model`` fitted to the exact points xi(t) with errors eps * xi (shift of y_c optional)."""

    def resid(p):
        tt = t + (p[2] if free_shift else 0.0)
        if np.any(tt <= 0):
            return 1e6 * np.ones_like(t)
        return (model(tt, p) - xi) / (eps * xi)

    best = None
    for p0 in (
        [np.log(xi[-1]), 0.6, 0.0],
        [np.log(xi[-1]), 1.0, 0.01],
        [np.log(xi[-1]), 0.3, -0.01],
        [np.log(xi[-1]), 2.0, 0.02],
    ):
        p0 = p0 if free_shift else p0[:2]
        try:
            r = least_squares(resid, p0, method="lm" if len(p0) <= len(t) else "trf", max_nfev=20000)
        except Exception:  # a failed start is skipped, the other starts decide
            continue
        if best is None or r.cost < best.cost:
            best = r
    return 2 * best.cost if best is not None else np.nan


def discrimination_table():
    """Every (grid, L_max, eps, y_c free/fixed): usable points, dynamic range, Delta chi^2 both ways."""
    rows = []
    for gname, t in GRIDS.items():
        for Lmax in L_MAX:
            P, W, info = curves(t.min(), t.max())
            use = W(t) <= Lmax / 3.0
            tu = t[use]
            for eps in EPS:
                for free_shift in (False, True):
                    if len(tu) < (3 if free_shift else 2) + 1:
                        d1 = d2 = np.nan
                    else:
                        d1 = wrong_model_chi2(power_model, tu, W(tu), eps, free_shift)  # truth walking, fit power law
                        d2 = wrong_model_chi2(walking_model, tu, P(tu), eps, free_shift)  # truth power law, fit walking
                    rows.append(
                        dict(
                            grid=gname,
                            Lmax=Lmax,
                            n_use=int(use.sum()),
                            R=float(info["R"]),
                            eps=eps,
                            free_shift=free_shift,
                            dchi2_WP=float(d1),
                            dchi2_PW=float(d2),
                        )
                    )
    return rows
