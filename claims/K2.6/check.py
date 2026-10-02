"""K2.6: the eps-model with Majorana-mass sources solved exactly on 2^3 x 4 by the flavour-factorised fermion-bag
expansion: Wick/flavour factorisation sign +1; the light amplitude of a 1+3 source is a symmetry zero at every order;
the 2+2 split is sign-free in the bag representation (perfect squares) and in a complex-conjugate-pair HS."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import itertools

import numpy as np

from masspairing.algebra.bags import order_coefficients, site_bag_weights, wick_full
from masspairing.algebra.grassmann import pfaffian
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.patterns import chiral_mass, source_pattern

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K2.6")
lat = Lattice((2, 2, 2, 4))
V = lat.V
K = kinetic_matrix(lat).toarray()
C = chiral_mass(lat).toarray()
C0 = source_pattern(lat).toarray()
h = 0.5
sets = [()] + [B for k in (2, 4) for B in itertools.combinations(range(V), k)]
rng = np.random.default_rng(148)
big = [tuple(sorted(rng.choice(V, size=k, replace=False))) for k in (6, 8, 12, 16, 24, 32) for _ in range(34)][:200]
c.record("2^3 x 4: V, sets with |B| <= 4, random larger sets (|B| = 6 ... 32)", (V, len(sets), len(big)))


def flavour_ops(kind):
    return {
        "1+3": [K, K, K, K - h * C],
        "2+2": [K, K, K - h * C, K - h * C],
        "blind": [K - h * C] * 4,
    }[kind]


# ---- (1) Wick / flavour factorisation sign
for kind in ("1+3", "2+2", "blind"):
    Gs, W = site_bag_weights(flavour_ops(kind), sets + big)
    scale = max(abs(W[B][0]) for B in sets[1:] + big)
    worst = max(abs(wick_full(Gs, B) - W[B][0]) / (abs(W[B][0]) + 1e-6 * scale) for B in sets[1:] + big)
    c.item(
        f"{kind}: Pf(G_full[B]) in the vertex ordering = +prod_a Pf(G^a[B]) on all {len(sets) - 1 + len(big)} sets "
        "(worst relative deviation)",
        worst,
        worst < 1e-8,
        fmt="{:.1e}",
    )

# ---- (2) 1+3: the light amplitude is a symmetry zero at every order
Ms = flavour_ops("1+3")
for name, Cp in (("C_chi", C), ("C0", C0)):
    Nl, Zk = order_coefficients(Ms, [Cp, Cp, Cp, None], sets)
    Nh, _ = order_coefficients(Ms, [None, None, None, Cp], sets)
    for k in (0, 2, 4):
        c.item(
            f"1+3, C' = {name}, order U^{k}: light numerator, heavy numerator, Z_k",
            (Nl[k], f"{Nh[k]:.6e}", f"{Zk[k]:.6e}"),
            abs(Nl[k]) < 1e-12 * max(abs(Nh[k]), 1e-300) + 1e-14,
        )
    Ns, _ = order_coefficients(Ms, [Cp, Cp, Cp, None], big)
    c.item(
        f"1+3, C' = {name}: light numerators on the random large sets (max |.|)",
        max(abs(v) for v in Ns.values()),
        all(abs(v) < 1e-12 for v in Ns.values()),
        fmt="{:.1e}",
    )
c.item(
    "heavy amplitude O(h) != 0 for C_chi, = 0 for the nodal C0 (order U^0 numerators)",
    (
        f"{order_coefficients(Ms, [None, None, None, C], [()])[0][0]:.4e}",
        order_coefficients(Ms, [None, None, None, C0], [()])[0][0],
    ),
    order_coefficients(Ms, [None, None, None, C], [()])[0][0] != 0
    and order_coefficients(Ms, [None, None, None, C0], [()])[0][0] == 0,
)

# ---- (4) 2+2: perfect-square bag weights
Gs, W = site_bag_weights(flavour_ops("2+2"), sets + big)
neg = sum(1 for B in sets + big if W[B][0] < -1e-300)
sq = max(abs(W[B][0] - (W[B][1][0] * W[B][1][2]) ** 2) / max(abs(W[B][0]), 1e-300) for B in sets[1:] + big)
c.item(
    f"2+2: all {len(sets) + len(big)} bag weights = [Pf(G1[B]) Pf(G3[B])]^2 >= 0: negatives, max rel. deviation",
    (neg, sq),
    neg == 0 and sq < 1e-10,
)
_, W13 = site_bag_weights(flavour_ops("1+3"), sets + big)
c.record("1+3 weights negative on (not squares; recorded)", f"{sum(1 for B in sets + big if W13[B][0] < 0)} sets")

# ---- (5) complex-conjugate-pair HS of the 2+2 model
worst, minprod = 0.0, np.inf
for _ in range(20):
    z = rng.normal(size=V) + 1j * rng.normal(size=V)
    Z = np.diag(z)
    M13 = np.block([[K, Z], [-Z, K - h * C]])
    M24 = np.block([[K, Z.conj()], [-Z.conj(), K - h * C]])
    p13, p24 = pfaffian(M13), pfaffian(M24)
    worst = max(worst, abs(p24 - np.conj(p13)) / abs(p13))
    minprod = min(minprod, (p13 * p24).real)
c.item(
    "Pf(M24) = conj Pf(M13) on 20 random z (worst rel.); min of the weight Pf(M13) Pf(M24) = |Pf M13|^2",
    (f"{worst:.1e}", f"{minprod:.4g}"),
    worst < 1e-10 and minprod > 0,
)
Nz = 200000
z = (rng.normal(size=Nz) + 1j * rng.normal(size=Nz)) * np.sqrt(0.7 / 2)
mz = (abs(np.mean(z)), abs(np.mean(z**2)), np.mean(abs(z) ** 2))
c.item(
    "Gaussian z with <|z|^2> = 0.7: |<z>|, |<z^2>|, <|z|^2> (<e^{za + zbar b}> = 1 + <|z|^2> ab, a^2 = b^2 = 0)",
    tuple(round(float(x), 4) for x in mz),
    mz[0] < 5e-3 and mz[1] < 5e-3 and abs(mz[2] - 0.7) < 5e-3,
)

# ---- caveat: the 2^3 x 4 lattice has no spatial hopping
Ks = kinetic_matrix(lat)
Ks.eliminate_zeros()
nnz = np.unique(np.diff(Ks.indptr)).tolist()
ent = np.unique(np.round(abs(chiral_mass(lat).data), 4)).tolist()
ent4 = np.unique(np.round(abs(chiral_mass(Lattice((4,) * 4)).data), 4)).tolist()
c.item(
    "caveat: on 2^3 x 4 K has 2 nonzeros per row (time hopping only); |C_chi| entries vs 4^4",
    (nnz, ent, ent4),
    nnz == [2] and min(ent) > 1.5 * max(ent4),
)
c.done()
