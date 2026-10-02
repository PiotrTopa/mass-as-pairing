"""Free-theory baselines of the N1 observables.

At y = 0 and without link fields the real propagator is flavour-diagonal, G = (K - h C)^-1 (x) 1_4, with C the
taste-chiral mass C_chi (or any other same-parity pattern). ``free_point`` evaluates every key of the exact
``N1Measure`` (pair channel of C, light/heavy taste channels, momentum readouts, time-slice correlators and, optionally,
the six-fermion composites) on this G and adds the light/heavy log-ratio effective masses of the positive time-slice
correlators. Every interacting N1 number is read next to its free value at the same L, L_t, boundary condition and h.

The frozen baselines live in ``data/derived/free/`` (one JSON list of rows per file); ``free_lookup`` finds a row.
"""

from __future__ import annotations

import json

import numpy as np

from ..action import build_model
from ..data import derived
from ..lattice import Lattice, kinetic_matrix
from ..measure.n1 import N1Measure
from ..patterns import chiral_mass

FREE_DIR = derived("free")


def effective_mass(C):
    """Log-ratio effective mass m(t) = ln C(t)/C(t+1), t = 0 ... L_t/2 - 1 (nan where C is not positive)."""
    C = np.asarray(C, float)
    Lt = len(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.array([np.log(C[t] / C[t + 1]) if C[t] > 0 and C[t + 1] > 0 else np.nan for t in range(Lt // 2)])


def bc_tuple(bc: str):
    """'aaaa' / 'pppa' -> fermion boundary signs."""
    return tuple(1 if c == "p" else -1 for c in bc)


def free_point(L, Lt, bc, h, composite=True, comp=(0, 3), pattern=None):
    """Every N1 observable of the free theory on an L^3 x L_t box with boundary condition ``bc`` ('aaaa', 'pppa').

    ``pattern`` (a sparse V x V matrix) replaces C_chi in the propagator only (controls); the measured channels stay
    those of C_chi. Returns a JSON-ready dict (lists and floats) with the measure keys, mL_eff, mR_eff and the box
    data L, Lt, bc, h, V, n_boundary, p0.
    """
    lat = Lattice((L, L, L, Lt), bc=bc_tuple(bc))
    V = lat.V
    model = build_model(lat, 0.0, 0.0, 1.0, h=h, pattern="chiral")
    fm = N1Measure(model, exact=True, taste=True, six_fermion=composite, comp=comp)
    C = chiral_mass(lat, comp) if pattern is None else pattern
    Minv = np.linalg.inv(kinetic_matrix(lat).toarray() - h * C.toarray())
    G = np.zeros((V, 4, V, 4))
    for a in range(4):
        G[:, a, :, a] = Minv
    del Minv
    out = {}
    out.update(fm.pair_channel(G=G, C=fm.C0, prefix="f"))
    out.update(fm.taste_channels(G=G))
    res = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in out.items()}
    res["mL_eff"] = effective_mass(out["CL_t"]).tolist()
    res["mR_eff"] = effective_mass(out["CR_t"]).tolist()
    res.update(L=L, Lt=Lt, bc=bc, h=h, V=V, n_boundary=fm.tp.n_boundary, p0=fm.p0.tolist())
    return res


def load_free(*names):
    """Rows of the frozen baseline files ``data/derived/free/<name>`` (concatenated)."""
    rows = []
    for n in names:
        rows += json.loads((FREE_DIR / n).read_text())
    return rows


def free_lookup(rows, L, Lt, bc, h):
    """The first row with the given box and h (None if absent)."""
    for r in rows:
        if r["L"] == L and r["Lt"] == Lt and r["bc"] == bc and abs(r["h"] - h) < 1e-12:
            return r
    return None


def max_deviation(a, b, floor=1e-12):
    """Largest absolute difference between two baseline rows over their common numeric keys (nan-aware).

    The effective masses mL_eff / mR_eff are compared only where both correlator entries they use exceed ``floor``
    times the correlator's largest entry (elsewhere they are logarithms of rounding noise, e.g. on zone-boundary
    boxes)."""
    worst = 0.0
    for k in set(a) & set(b):
        if isinstance(a[k], str) or isinstance(b[k], str):
            continue
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if x.shape != y.shape:
            return np.inf
        if k in ("mL_eff", "mR_eff"):
            C = np.abs(np.asarray(a["CL_t" if k == "mL_eff" else "CR_t"], float))
            ok = np.array([min(C[t], C[t + 1]) > floor * C.max() for t in range(len(x))])
            x, y = x[ok], y[ok]
        if np.any(np.isnan(x) != np.isnan(y)):
            return np.inf
        nan = np.isnan(x)
        if np.any(~nan):
            worst = max(worst, float(np.abs(x[~nan] - y[~nan]).max()))
    return worst
