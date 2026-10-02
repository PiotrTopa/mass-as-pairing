"""Autocorrelation times, blocked jackknife and the derived quantities of the chain analyses.

Errors: blocked jackknife with block length about 2 tau_int (Madras-Sokal automatic windowing, W >= c tau rule).
"""

from __future__ import annotations

import numpy as np


def tau_int(x, c=6.0):
    """Integrated autocorrelation time by automatic windowing. Returns (tau, window, converged)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or np.std(x) == 0:
        return 0.5, 0, True
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1))
    acf /= acf[0]
    tau = 0.5
    for W in range(1, n // 2):
        tau = 0.5 + acf[1 : W + 1].sum()
        if W >= c * tau:
            return float(max(tau, 0.5)), W, True
    return float(max(tau, 0.5)), n // 2, False


def jackknife(x, f=np.mean, nblocks=None, blen=None):
    """Blocked jackknife of the statistic f over samples x (N,) or (N, k). Returns (value, error)."""
    x = np.asarray(x)
    n = len(x)
    if blen is None:
        blen = max(1, n // (nblocks or 20))
    nb = n // blen
    if nb < 2:
        return f(x), np.nan
    x = x[: nb * blen]
    blocks = x.reshape(nb, blen, *x.shape[1:])
    full = f(x)
    parts = []
    for b in range(nb):
        rest = np.concatenate([blocks[:b], blocks[b + 1 :]]).reshape(-1, *x.shape[1:])
        parts.append(f(rest))
    parts = np.asarray(parts)
    err = np.sqrt((nb - 1) / nb * np.sum((parts - parts.mean(0)) ** 2, 0))
    return full, err


def summarize(x, blen=None):
    """Mean, blocked-jackknife error (block length ceil(2 tau_int) unless given), tau_int and its window."""
    x = np.asarray(x, float)
    tau, W, ok = tau_int(x)
    bl = blen or max(1, int(np.ceil(2 * tau)))
    mean, err = jackknife(x, np.mean, blen=bl)
    return dict(mean=float(mean), err=float(err), tau_int=tau, window=W, ok=ok, n=len(x), blen=bl)


def susceptibility(x, V, blen):
    """V (<x^2> - <x>^2) with jackknife error."""
    return jackknife(np.asarray(x, float), lambda a: V * (np.mean(a**2) - np.mean(a) ** 2), blen=blen)


def xi2(S0, Smin, L, blen):
    """Second-moment correlation length from the series S(0), S(p_min) with jackknife error."""

    def f(a):
        return np.sqrt(max(np.mean(a[:, 0]) / np.mean(a[:, 1]) - 1.0, 0.0) / (4 * np.sin(np.pi / L) ** 2))

    return jackknife(np.stack([S0, Smin], 1), f, blen=blen)


def effective_mass(C, blen, periodic=True):
    """m_eff(t) from an (N, Lt) correlator: cosh solve (periodic) or log ratio. Jackknife errors."""
    C = np.asarray(C, float)
    Lt = C.shape[1]

    def meff(a):
        c = a.mean(0)
        out = np.full(Lt, np.nan)
        for t in range(1, Lt - 1):
            if periodic:
                r = (c[t - 1] + c[t + 1]) / (2 * c[t]) if c[t] != 0 else np.nan
                out[t] = np.arccosh(r) if r >= 1 else np.nan
            else:
                out[t] = np.log(c[t] / c[t + 1]) if c[t] > 0 and c[t + 1] > 0 else np.nan
        return out

    return jackknife(C, meff, blen=blen)
