"""K1.2: the unique holomorphic quartic Phi_10^2 of one Weyl 16; Phi_126^2 = 0 identically; 126 (x) 126; the unique
charge-8 operator (Phi_10.Phi_10)^2, Fierz-identical in the 10 and 126 channels."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import itertools
import time

import numpy as np

from masspairing.algebra import grassmann as gr
from masspairing.algebra.characters import (
    char126,
    decompose,
    dims,
    invariants,
    ms_add,
    ms_dim,
    ms_dual,
    ms_ext_powers,
    ms_schur,
    ms_sym_powers,
    ms_tensor,
    pretty,
    spinor16,
    vector10,
)
from masspairing.algebra.spin10 import antisym_products, charge_conjugation, gammas, root_vectors, weyl_weight_basis
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers

c = Check("K1.2")
s16 = spinor16(-1)

# ---- (1) holomorphic quartics: Lorentz scalars in Lambda^4(16 (x) 2) = S_(2,2)(16) (Cauchy)
S22 = ms_schur(s16, (2, 2))
dec = decompose(S22)
c.record("S_(2,2)(16)", pretty(dec))
c.item("dim S_(2,2)(16)", ms_dim(S22), ms_dim(S22) == 5440)
c.item("Spin(10) singlets in S_(2,2)(16)", invariants(S22), invariants(S22) == 1)
others = [invariants(ms_schur(s16, lam)) for lam in ((4,), (2, 1, 1), (1, 1, 1, 1))]
c.item("singlets in Sym^4(16), S_(2,1,1)(16), Lambda^4(16)", others, others == [0, 0, 0])

# direct: so(10) + sl(2) invariants of Lambda^4(32) = kernel of all root vectors on the zero-weight subspace
G = gammas()
C = charge_conjugation(G)
W, wts = weyl_weight_basis(G)
wt32 = np.array(
    [tuple(int(round(2 * x)) for x in wts[a]) + (1 if al == 0 else -1,) for a in range(16) for al in (0, 1)]
)
monos = [S for S in itertools.combinations(range(32), 4) if not np.any(wt32[list(S)].sum(0))]
c.item("zero-weight monomials in Lambda^4(32)", len(monos), len(monos) == 240)
ops = [np.kron(W.conj().T @ E @ W, np.eye(2)) for E in root_vectors(G).values()]
sp = np.array([[0, 1], [0, 0]], float)
ops += [np.kron(np.eye(16), sp), np.kron(np.eye(16), sp.T)]
N = len(monos)
Gram = np.zeros((N, N), complex)
for X in ops:
    D, _ = gr.derivation_matrix(X, monos)
    Gram += (D.conj().T @ D).toarray()
ev = np.linalg.eigvalsh(Gram)
nker = int((ev < 1e-8 * ev[-1]).sum())
c.item(
    "dimension of the joint kernel of the 40 + 2 root vectors (Gram spectrum head)",
    (nker, np.round(ev[:3], 8)),
    nker == 1,
)

# Grassmann: Phi_10.Phi_10 is nonzero and spans the kernel; sum over the 252 five-forms of Phi_I Phi_I vanishes
B1 = [W.T @ C @ g @ W for g in G]
B5 = [W.T @ C @ M @ W for M in antisym_products(G, 5)]
Phi10 = [gr.bilinear(M) for M in B1]
P10sq = {}
for p in Phi10:
    P10sq = gr.add(P10sq, gr.mul(p, p))
P10sq = gr.clean(P10sq)
c.item(
    "Phi_10.Phi_10: monomials, norm",
    (len(P10sq), round(gr.norm(P10sq), 6)),
    len(P10sq) == 240 and gr.norm(P10sq) > 1e-6 * sum(gr.norm(p) ** 2 for p in Phi10),
)
idx = {sum(1 << i for i in S): j for j, S in enumerate(monos)}
v = np.zeros(N, complex)
for m, cf in P10sq.items():
    v[idx[m]] = cf
c.item(
    "  annihilated by every root vector (|Gram v| / |v|)",
    np.linalg.norm(Gram @ v) / np.linalg.norm(v),
    np.linalg.norm(Gram @ v) < 1e-9 * np.linalg.norm(v),
    fmt="{:.1e}",
)
P126sq = {}
ref = 0.0
for M in B5:
    p = gr.bilinear(M)
    pp = gr.mul(p, p)
    P126sq = gr.add(P126sq, pp)
    ref += gr.norm(pp)
c.item(
    "sum_I Phi_I Phi_I over the 252 five-forms: norm (reference scale)",
    (gr.norm(P126sq), round(ref, 3)),
    gr.norm(P126sq) < 1e-10 * ref,
)

# ---- (2) 126 (x) 126 by characters
c.item(
    "Sym^2(16) = 10 + 126",
    pretty(decompose(ms_sym_powers(s16, 2)[2])),
    decompose(ms_sym_powers(s16, 2)[2]) == {(2, 0, 0, 0, 0): 1, (2, 2, 2, 2, -2): 1},
)
c.item(
    "Lambda^2(16) = 120",
    pretty(decompose(ms_ext_powers(s16, 2)[2])),
    decompose(ms_ext_powers(s16, 2)[2]) == {(2, 2, 2, 0, 0): 1},
)
c126 = char126()
tt = ms_tensor(c126, c126)
dtt = decompose(tt)
c.item("126 x 126 dimensions", dims(dtt), dims(dtt) == [54, 945, 1050, 2772, 4125, 6930])
S2, L2 = ms_sym_powers(c126, 2)[2], ms_ext_powers(c126, 2)[2]
c.item("Sym^2(126) dimensions", dims(decompose(S2)), dims(decompose(S2)) == [54, 1050, 2772, 4125])
c.item("Lambda^2(126) dimensions", dims(decompose(L2)), dims(decompose(L2)) == [945, 6930])
n_tt, n_tbar = invariants(tt), invariants(ms_tensor(c126, ms_dual(c126)))
c.item("singlets in 126 x 126 and in 126 x 126bar", (n_tt, n_tbar), n_tt == 0 and n_tbar == 1)
c54 = ms_add(ms_sym_powers(vector10(), 2)[2], {(0,) * 5: -1})
c.item(
    "54 is real: self-conjugate character, singlets in 54 x 54",
    invariants(ms_tensor(c54, c54)),
    ms_dim(c54) == 54 and ms_dual(c54) == c54 and invariants(ms_tensor(c54, c54)) == 1,
)
n4, ns4 = invariants(ms_tensor(tt, tt)), invariants(ms_sym_powers(c126, 4)[4])
c.item("singlets in 126^(x4) and in Sym^4(126)", (n4, ns4), n4 == 3 and ns4 == 1)

# ---- (3) charge-6 and charge-8 holomorphic Lorentz-scalar invariants: S_(2^k)(16)
n6 = invariants(ms_schur(s16, (2, 2, 2)))
c.item("charge-6 holomorphic Lorentz-scalar invariants", n6, n6 == 0)
t0 = time.time()
n8 = invariants(ms_schur(s16, (2, 2, 2, 2)))
c.item(f"charge-8 holomorphic Lorentz-scalar invariants ({time.time() - t0:.0f} s)", n8, n8 == 1)

# Grassmann realisation: the 54-plets of the 10 and 126 channels and the three octics
combos5 = list(itertools.combinations(range(10), 5))
Phi5 = {I: gr.bilinear(W.T @ C @ M @ W) for I, M in zip(combos5, antisym_products(G, 5), strict=True)}


def form(index):
    """Phi_126 with an unordered five-index label (antisymmetric)."""
    if len(set(index)) < 5:
        return {}
    perm = sorted(range(5), key=lambda i: index[i])
    sgn = int(round(np.linalg.det(np.eye(5)[perm])))
    return gr.scale(Phi5[tuple(sorted(index))], sgn)


tr10 = {}
for p in Phi10:
    tr10 = gr.add(tr10, gr.mul(p, p))
pairs = [(m, n) for m in range(10) for n in range(m, 10)]
Q10 = [gr.clean(gr.add(gr.mul(Phi10[m], Phi10[n]), tr10, -0.1 if m == n else 0.0)) for m, n in pairs]
Qraw = {}
for m, n in pairs:
    q = {}
    for J in itertools.combinations([i for i in range(10) if i not in (m, n)], 4):
        q = gr.add(q, gr.mul(form((m,) + J), form((n,) + J)))
    Qraw[(m, n)] = gr.clean(q)
tr126 = {}
for m in range(10):
    tr126 = gr.add(tr126, Qraw[(m, m)])
c.item(
    "trace of the 126-channel 54 candidate (Phi_126 Phi_126)_mm", gr.norm(tr126), gr.norm(tr126) < 1e-9, fmt="{:.1e}"
)
Q126 = [gr.clean(gr.add(Qraw[(m, n)], tr126, -0.1 if m == n else 0.0)) for m, n in pairs]
r10, _ = gr.rank_of(Q10)
r126, _ = gr.rank_of(Q126)
rb, _ = gr.rank_of(Q10 + Q126)
c.item("rank of the 54-plet quartics: 10 channel, 126 channel, together", (r10, r126, rb), r10 == r126 == rb == 54)
k0 = next(iter(Q10[1]))
ratio = Q126[1][k0] / Q10[1][k0]
same = all(abs(Q126[i].get(m, 0) - ratio * cf) < 1e-9 for i in range(55) for m, cf in Q10[i].items())
c.item(
    "Fierz: (Phi_126 Phi_126)_54 = ratio x (Phi_10 Phi_10)_54 identically; ratio",
    np.round(ratio, 9),
    same and abs(ratio + 10) < 1e-9,
)
O8 = gr.mul(tr10, tr10)
O54_10, O54_126 = {}, {}
for q1, q2, (m, n) in zip(Q10, Q126, pairs, strict=True):
    wgt = 1.0 if m == n else 2.0
    O54_10 = gr.add(O54_10, gr.mul(q1, q1), wgt)
    O54_126 = gr.add(O54_126, gr.mul(q2, q2), wgt)
r, _ = gr.rank_of([O8, O54_10, O54_126])
c.item(
    "octics (Phi_10.Phi_10)^2, sum (Phi_10 Phi_10)_54^2, sum (Phi_126 Phi_126)_54^2: norms",
    [round(gr.norm(x), 3) for x in (O8, O54_10, O54_126)],
    gr.norm(O8) > 1,
)
c.item("  rank of the three octics", r, r == 1)
c.done()
