"""Walking-versus-power-law discrimination mock of K7.3.

Can xi(y) on a grid of |y - y_c| tell walking, xi = a exp(c/sqrt(t)), from a power law, xi = a t^-nu? The true
curve (matched dynamic range, xi(t_max) = 1.5, nu = 0.61) is sampled at the grid points with relative errors eps; only
points with xi <= L_max/3 are usable; the wrong model is fitted with (log a, exponent) and optionally a shift of y_c;
its minimum chi^2 is the expected Delta chi^2 (Gaussian errors, no corrections to scaling: an optimistic bound).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

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
