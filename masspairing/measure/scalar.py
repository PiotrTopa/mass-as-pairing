"""Observables of the sigma field (exact per configuration).

Sigma = (1/V) sum_x sigma(x), Sigma_stag = (1/V) sum_x eps(x) sigma(x), sigma2 = (1/V) sum_x |sigma|^2, sigma4;
S(p) = (1/V) sum_a |sigma_a(p)|^2 at p = 0, pi(1,1,1,1) and the neighbours p_min of both (averaged over mu and sign),
giving the second-moment correlation length xi_2^2 = (S(0)/S(p_min) - 1)/(4 sin^2(pi/L)); zero-spatial-momentum
time-slice correlators Ct and Ct_stag.
"""

from __future__ import annotations

import numpy as np

from ..lattice import to_numpy


def scalar_observables(lat, sigma) -> dict:
    s = to_numpy(sigma)
    V = lat.V
    L = lat.shape
    eps = to_numpy(lat.eps)
    out = {}
    out["Sigma"] = s.reshape(3, -1).mean(1)
    out["Sigma_stag"] = (s * eps).reshape(3, -1).mean(1)
    out["Sigma_abs"] = float(np.linalg.norm(out["Sigma"]))
    out["Sigma_stag_abs"] = float(np.linalg.norm(out["Sigma_stag"]))
    out["sigma2"] = float((s**2).sum(0).mean())
    out["sigma4"] = float(((s**2).sum(0) ** 2).mean())
    F = np.fft.fftn(s, axes=(1, 2, 3, 4))
    Sp = (np.abs(F) ** 2).sum(0) / V
    pi = tuple(n // 2 for n in L)
    out["S_0"] = float(Sp[0, 0, 0, 0])
    out["S_pi"] = float(Sp[pi])
    Smin, Spimin = [], []
    for mu in range(4):
        for sgn in (1, -1):
            idx = [0, 0, 0, 0]
            idx[mu] = sgn % L[mu]
            Smin.append(Sp[tuple(idx)])
            idx = list(pi)
            idx[mu] = (pi[mu] + sgn) % L[mu]
            Spimin.append(Sp[tuple(idx)])
    out["S_pmin"] = float(np.mean(Smin))
    out["S_pi_pmin"] = float(np.mean(Spimin))
    St = s.sum(axis=(1, 2, 3))
    Sts = (s * eps).sum(axis=(1, 2, 3))
    Lt = L[3]
    out["Ct"] = np.array([np.mean(np.sum(St * np.roll(St, -t, axis=1), 0)) for t in range(Lt)]) / (V / Lt)
    out["Ct_stag"] = np.array([np.mean(np.sum(Sts * np.roll(Sts, -t, axis=1), 0)) for t in range(Lt)]) / (V / Lt)
    return out


def xi2_from_S(S0, Smin, L):
    """Second-moment correlation length from S(0) and S(p_min) on a lattice of extent L."""
    r = S0 / Smin - 1.0
    return np.sqrt(max(r, 0.0) / (4.0 * np.sin(np.pi / L) ** 2))
