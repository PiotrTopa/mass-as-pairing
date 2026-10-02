"""K1.3: the Lorentz-scalar Spin(10) x U(1)_psi-invariant psi psi psibar psibar quartics of one Weyl 16 are spanned by
|Phi_10|^2 and |Phi_126|^2; the current-current terms J.J (16 x 16bar = 1 + 45 + 210) are combinations of them."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from fractions import Fraction

import numpy as np

from masspairing.algebra import grassmann as gr
from masspairing.algebra.characters import decompose, invariants, ms_ext_powers, ms_tensor, pretty, weyl_lorentz
from masspairing.algebra.spin10 import (
    antisym_products,
    charge_conjugation,
    gammas,
    so10_generators,
    weyl_weight_basis,
)
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers

c = Check("K1.3")

# (1) count under D5 x A1 x A1 (complexified Lorentz group): psi in 16 (x) (1/2, 0), psibar in 16bar (x) (0, 1/2)
V, Vb = weyl_lorentz(-1), weyl_lorentz(-1, dotted=True)
L2V, L2Vb = ms_ext_powers(V, 2)[2], ms_ext_powers(Vb, 2)[2]
dec = decompose(L2V, k=2)
c.item(
    "Lambda^2(16 (x) (1/2,0)) = (10,(0,0)) + (126,(0,0)) + (120,(1,0))",
    pretty(dec, k=2),
    dec == {(2, 0, 0, 0, 0, 0, 0): 1, (2, 2, 2, 2, -2, 0, 0): 1, (2, 2, 2, 0, 0, 2, 0): 1},
)
ninv = invariants(ms_tensor(L2V, L2Vb), k=2)
c.item("invariants in Lambda^2 V (x) Lambda^2 Vbar", ninv, ninv == 2)

# (2) explicit Grassmann quartics on 64 generators: psi_{a alpha} -> 2a + alpha, psibar_{a ad} -> 32 + 2a + ad
G = gammas()
C = charge_conjugation(G)
W, _ = weyl_weight_basis(G)
M10 = [W.T @ C @ g @ W for g in G]
M126 = [W.T @ C @ X @ W for X in antisym_products(G, 5)]


def abs2(Ms):
    """sum_A Phi_A Phibar_A, Phi_A = M^A_ab eps psi psi, Phibar_A = (M^A_ab)* eps psibar psibar."""
    out = {}
    for M in Ms:
        out = gr.add(out, gr.mul(gr.bilinear(M), gr.bilinear(M.conj(), offset=32)))
    return gr.clean(out)


def onb(mats):
    A = np.array([m.ravel() for m in mats])
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    return [vh[i].reshape(16, 16) for i in range(int((s > 1e-8).sum()))]


A10, A126 = abs2(M10), abs2(M126)
T1 = [np.eye(16) / 4]
T45 = onb([W.conj().T @ S @ W for S in so10_generators(G)])
T210 = onb([W.conj().T @ X @ W for X in antisym_products(G, 4)])
c.item(
    "orthonormal generator sets of 16 x 16bar: sizes",
    (len(T1), len(T45), len(T210)),
    (len(T45), len(T210)) == (45, 210),
)
JJ1, JJ45, JJ210 = gr.current_current(T1), gr.current_current(T45), gr.current_current(T210)
names = ["|Phi10|^2", "|Phi126|^2", "J1.J1", "J45.J45", "J210.J210"]
polys = [A10, A126, JJ1, JJ45, JJ210]
c.record("monomial counts", dict(zip(names, map(len, polys), strict=True)))


# (3) invariance under so(10), the full Lorentz algebra sl(2, C) (U(1)_psi charge 0 by construction)
def gen(Xpsi, Xbar):
    X = np.zeros((64, 64), complex)
    X[:32, :32] = Xpsi
    X[32:, 32:] = Xbar
    return X


gens = []
rng = np.random.default_rng(1)
for _ in range(6):
    S = sum(cf * (W.conj().T @ X @ W) for cf, X in zip(rng.normal(size=45), so10_generators(G), strict=True))
    gens.append(gen(np.kron(S, np.eye(2)), np.kron(-S.T, np.eye(2))))
sxm, sym_, szm = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])
for A in (sxm, sym_, szm, 1j * sxm, 1j * sym_, 1j * szm):
    gens.append(gen(np.kron(np.eye(16), A), np.kron(np.eye(16), A.conj())))
viol = max(max(gr.norm(gr.derivation(p, X)) for X in gens) / gr.norm(p) for p in polys)
c.item("max relative violation of so(10) + sl(2,C) invariance over the five quartics", viol, viol < 1e-10, fmt="{:.1e}")

# (4) rank and the relation matrix
r, s = gr.rank_of(polys)
c.item("rank of {|Phi10|^2, |Phi126|^2, J1.J1, J45.J45, J210.J210}", r, r == 2)
vecs = gr.coefficient_vectors(polys)
Bm = np.array(vecs[:2]).T
exact = {
    "J1.J1": (Fraction(-1, 256), Fraction(-1, 512)),
    "J45.J45": (Fraction(27, 256), Fraction(-5, 512)),
    "J210.J210": (Fraction(-21, 128), Fraction(-5, 256)),
}
rel = {}
for name, v in zip(names[2:], vecs[2:], strict=True):
    x, *_ = np.linalg.lstsq(Bm, v, rcond=None)
    resid = np.linalg.norm(Bm @ x - v) / np.linalg.norm(v)
    rel[name] = (x[0].real, x[1].real)
    e = exact[name]
    ok = resid < 1e-10 and abs(x[0].imag) < 1e-9 and abs(x[1].imag) < 1e-9
    ok &= abs(x[0].real - float(e[0])) < 1e-10 and abs(x[1].real - float(e[1])) < 1e-10
    c.item(
        f"{name} = a |Phi10|^2 + b |Phi126|^2: (a, b) = ({e[0]}, {e[1]}), residual",
        (round(x[0].real, 8), round(x[1].real, 8), f"{resid:.1e}"),
        ok,
    )
R = np.array([[float(v) for v in exact[k]] for k in ("J1.J1", "J45.J45")])
inv = np.linalg.inv(R)  # rows: |Phi10|^2, |Phi126|^2 in terms of (J1.J1, J45.J45)
c.item(
    "inverse: |Phi10|^2 = -40 J1.J1 + 8 J45.J45, |Phi126|^2 = -432 J1.J1 - 16 J45.J45",
    np.round(inv, 9),
    np.allclose(inv, [[-40, 8], [-432, -16]], atol=1e-9),
)
c.done()
