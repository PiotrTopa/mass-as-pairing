"""Calibration of the Yukawa axis against the published phase diagram (claim I.3).

Per stored chain (all trajectories, no cut): the mean of |Sigma_stag| with its blocked-jackknife error, the staggered
susceptibility chi_Sigma,stag = V (<|Sigma_stag|^2> - <|Sigma_stag|>^2) and the second-moment length xi_2,stag. The
block length of the derived quantities is ceil(2 max tau_int) over every stored scalar and fermion series of the
chain (the convention of the chain summaries). A rescaling y_ours = s y_ref of the published L = 8, kappa = 0.2 window
(y_1, y_2) = (1.706, 2.69) is tested by classifying the measured points (ordered iff |Sigma_stag| >= 0.15).
"""

from __future__ import annotations

import numpy as np

from .stats import summarize, susceptibility, xi2

SCALAR_KEYS = ("sigma2", "Sigma_abs", "Sigma_stag_abs", "S_0", "S_pi", "S_pmin", "S_pi_pmin")
FERMION_KEYS = ("phi_abs", "phi_stag_abs", "phi_sq", "phi_stag_sq", "O4")
Y1_REF, Y2_REF = 1.706, 2.69  # published L = 8, kappa = 0.2 window (reference units)
ORDERED = 0.15  # |Sigma_stag| threshold of the ordered (AFM) phase


def point_summary(series, meta):
    """Summary of one chain: acceptance, tau_int of every series, |Sigma_stag|, chi_Sigma,stag, xi_2,stag."""
    shape = tuple(meta["shape"])
    V = int(np.prod(shape))
    L = shape[0]
    res = dict(
        L=L,
        y=float(meta["y"]),
        kappa=float(meta["kappa"]),
        ntraj=int(len(series["ts_dH"])),
        acceptance=float(np.mean(series["ts_accepted"])),
    )
    taus = {}
    for k in SCALAR_KEYS + FERMION_KEYS:
        x = series.get("ts_" + k)
        if x is None or len(x) == 0:
            continue
        s = summarize(x)
        res[k] = s
        taus[k] = s["tau_int"]
    blen = max(1, int(np.ceil(2 * max(taus.values()))))
    res["blen"] = blen
    res["tau_max"] = max(taus.values())
    v, e = susceptibility(series["ts_Sigma_stag_abs"], V, blen)
    res["chi_Sigma_stag"] = dict(mean=float(v), err=float(e))
    v, e = xi2(series["ts_S_pi"], series["ts_S_pi_pmin"], L, blen)
    res["xi2_stag"] = dict(mean=float(v), err=float(e))
    return res


def y1_parabola(ys, chi, err, nsamples=4000, seed=0):
    """Position of the chi maximum from parabolas through three points resampled within their errors.

    Returns (median, 16th percentile, 84th percentile); fits with a non-negative curvature are discarded.
    """
    rng = np.random.default_rng(seed)
    ys = np.asarray(ys, float)
    fits = []
    for _ in range(nsamples):
        a, b, _c = np.polyfit(ys, chi + err * rng.normal(size=3), 2)
        if a < 0:
            fits.append(np.clip(-b / (2 * a), ys[0], ys[-1]))
    lo, hi = np.percentile(fits, [16, 84])
    return float(np.median(fits)), float(lo), float(hi)


def misclassified(points, s):
    """L = 8, kappa = 0.2 points (y, ordered) outside 0.15 s of either edge of (s y_1, s y_2) whose phase is wrong."""
    a, b = s * Y1_REF, s * Y2_REF
    return [y for y, o in points if min(abs(y - a), abs(y - b)) > 0.15 * s and o != (a < y < b)]


SQRT2 = float(np.sqrt(2))


def kappa_window(kappa, s=SQRT2):
    """Predicted ordered window at kappa: (s y_1, s y_2(kappa)), y_2 linear from (kappa_c, y_1) = (-0.01, 1.706) to
    (0.2, 2.69)."""
    return s * Y1_REF, s * (Y1_REF + 0.984 * max(kappa + 0.01, 0) / 0.21)
