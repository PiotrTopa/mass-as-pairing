"""Lattice symmetries of the sourced model and the symmetry-zero test of one-point functions.

Signed-permutation symmetries of the kinetic matrix K: χ(x) → s(x) χ(gx) with g a site map (hypercubic
permutation/reflection,
one-site translation) and s ∈ {±1}^V found by BFS on the links of K.  Every such map is a site permutation with a
flavour-blind sign field, so the ε-vertex (s⁴ = 1), the Yukawa and σ terms (s² = 1 on every on-site bilinear) and the
κ term are
invariant; the sourced action is invariant iff the source pattern C_src is mapped to +C_src.  A one-point function
⟨½χᵀOχ⟩ (or the composite ⟨T(x) O T(z)⟩, which transforms with the same rule) is forced to vanish by the symmetry
group G of the
sourced model iff the G-average of O vanishes:  Ō = (1/|G|) Σ_g T_g O T_gᵀ = 0.  `classify` reports ‖Ō‖/‖O‖ for a
pattern.
"""

import itertools

import numpy as np
import scipy.sparse as sps

from .lattice import kinetic_matrix as build_kinetic_d


def site_map(lat, perm, refl):
    co = lat.coords.reshape(lat.d, lat.V)
    new = np.zeros_like(co)
    for mu in range(lat.d):
        if lat.shape[perm[mu]] != lat.shape[mu]:
            return None
        new[perm[mu]] = (refl[mu] * co[mu]) % lat.shape[mu]
    return np.ravel_multi_index(tuple(new), lat.shape)


def translation(lat, mu, dist=1):
    co = lat.coords.reshape(lat.d, lat.V).copy()
    co[mu] = (co[mu] + dist) % lat.shape[mu]
    return np.ravel_multi_index(tuple(co), lat.shape)


def sign_field(K, g):
    """s with s(x)s(y)K[gx,gy] = K[x,y] (BFS over the links of K); None if g is not a symmetry of K."""
    V = K.shape[0]
    Kc = K.tocsr()
    Kd = Kc.toarray()
    s = np.zeros(V)
    s[0] = 1.0
    queue = [0]
    while queue:
        x = queue.pop()
        for y in Kc.indices[Kc.indptr[x] : Kc.indptr[x + 1]]:
            if Kd[x, y] == 0.0:
                continue
            kg = Kd[g[x], g[y]]
            if kg == 0.0:
                return None
            r = Kd[x, y] / kg
            if s[y] == 0.0:
                s[y] = s[x] * r
                queue.append(y)
            elif abs(s[x] * s[y] - r) > 1e-12:
                return None
    if np.any(s == 0.0):
        return None
    T = sps.csr_matrix((s, (g, np.arange(V))), shape=(V, V))
    return s if abs(T @ Kc @ T.T - Kc).max() < 1e-12 else None


def transform(O, g, s):
    """T O Tᵀ for a dense pattern O: (T O Tᵀ)[gx, gy] = s(x) s(y) O[x, y]."""
    Og = np.empty_like(O)
    Og[np.ix_(g, g)] = (s[:, None] * s[None, :]) * O
    return Og


def generators(lat, K, translations=True):
    """Symmetries of K among the hypercubic maps (as generators: all of them) and the one-site translations: list of
    (g, s)."""
    gens = []
    for perm in itertools.permutations(range(lat.d)):
        for refl in itertools.product((1, -1), repeat=lat.d):
            g = site_map(lat, perm, refl)
            if g is None:
                continue
            s = sign_field(K, g)
            if s is not None:
                gens.append((g, s))
    if translations:
        for mu in range(lat.d):
            g = translation(lat, mu)
            s = sign_field(K, g)
            if s is not None:
                gens.append((g, s))
    return gens


def closure(gens, V, max_size=200000):
    """The group generated (as (g, s) pairs, modulo the global sign of s)."""
    key = lambda g, s: g.tobytes() + (s * s[0]).astype(np.int8).tobytes()
    ident = (np.arange(V), np.ones(V))
    elems = {key(*ident): ident}
    queue = [ident]
    while queue:
        g, s = queue.pop()
        for tg, ts in gens:
            g2 = tg[g]
            s2 = s * ts[g]
            k2 = key(g2, s2)
            if k2 not in elems:
                elems[k2] = (g2, s2)
                queue.append((g2, s2))
                if len(elems) > max_size:
                    raise RuntimeError("group too large")
    return list(elems.values())


def stabiliser(group, C_src, tol=1e-10):
    """The elements mapping the source pattern to +C_src (the symmetry group of the sourced action)."""
    C = C_src if isinstance(C_src, np.ndarray) else C_src.toarray()
    return [(g, s) for g, s in group if abs(transform(C, g, s) - C).max() < tol]


def group_average(O, group):
    Od = O if isinstance(O, np.ndarray) else O.toarray()
    acc = np.zeros_like(Od)
    for g, s in group:
        acc += transform(Od, g, s)
    return acc / len(group)


def classify(patterns, lat, C_src, translations=True, tol=1e-10):
    """For each named pattern O (dense or sparse V×V): ‖Ō‖/‖O‖ under the symmetry group of the model sourced with
    C_src (K's
    signed-permutation symmetries incl. one-site translations, restricted to the stabiliser of C_src).  0 ⇒ an exact
    symmetry zero
    (the observable cannot be used as a one-point function); 1 ⇒ invariant (allowed).  Returns dict(name → ratio),
    |G|, |Stab|.
    """
    K = build_kinetic_d(lat)
    G = closure(generators(lat, K, translations), lat.V)
    S = stabiliser(G, C_src, tol)
    out = {}
    for name, O in patterns.items():
        Od = O if isinstance(O, np.ndarray) else O.toarray()
        n = np.linalg.norm(Od)
        out[name] = float(np.linalg.norm(group_average(Od, S)) / n) if n > 0 else 0.0
    return out, len(G), len(S)
