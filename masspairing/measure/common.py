"""Shared helpers of the fermion measures."""

from __future__ import annotations

import numpy as np


def ustat_pairs(vals):
    """Unbiased mean of products over distinct sample pairs: (sum_{k != k'} v_k v_k') / (k (k - 1)), k >= 2."""
    k = vals.shape[0]
    tot = vals.sum(0)
    return ((tot**2).sum() - (vals**2).sum()) / (k * (k - 1))


def n_from_blocks(S):
    """n_a(x) = -(i/2) tr[tau_a S(x)] for doublet blocks S (V, 2, 2), real (V, 3)."""
    n = np.empty((S.shape[0], 3))
    n[:, 0] = (-0.5j * (S[:, 0, 1] + S[:, 1, 0])).real
    n[:, 1] = (-0.5j * (-1j * S[:, 1, 0] + 1j * S[:, 0, 1])).real
    n[:, 2] = (-0.5j * (S[:, 0, 0] - S[:, 1, 1])).real
    return n


def epsilon_moments(ns, eps):
    """phi, phi_stag (3-vectors), norms and the (unbiased for >= 2 samples) squares from samples ns (k, V, 3)."""
    k = ns.shape[0]
    nbar = ns.mean(0)
    phi = nbar.mean(0)
    phi_stag = (nbar * eps[:, None]).mean(0)
    out = dict(
        phi=phi,
        phi_stag=phi_stag,
        phi_abs=float(np.linalg.norm(phi)),
        phi_stag_abs=float(np.linalg.norm(phi_stag)),
        O4_biased=float((nbar**2).sum(1).mean()),
    )
    if k >= 2:
        pk = ns.mean(1)
        psk = (ns * eps[None, :, None]).mean(1)
        out["phi_sq"] = float(ustat_pairs(pk))
        out["phi_stag_sq"] = float(ustat_pairs(psk))
        tot = ns.sum(0)
        out["O4"] = float((((tot**2).sum(1) - (ns**2).sum((0, 2))) / (k * (k - 1))).mean())
    else:
        out["phi_sq"] = out["phi_abs"] ** 2
        out["phi_stag_sq"] = out["phi_stag_abs"] ** 2
        out["O4"] = out["O4_biased"]
    return out
