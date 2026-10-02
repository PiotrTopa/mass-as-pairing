#!/usr/bin/env python3
"""results/algebra_fierz.json: the Fierz relation matrix of the psi psi psibar psibar quartics of one Weyl 16 (claim
K1.3): J_A.J_A = a |Phi_10|^2 + b |Phi_126|^2 for A = 1, 45, 210 (least squares on the explicit Grassmann polynomials,
with the residuals) and the exact rationals they equal."""

import json
import pathlib
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.algebra import grassmann as gr
from masspairing.algebra.spin10 import (
    antisym_products,
    charge_conjugation,
    gammas,
    so10_generators,
    weyl_weight_basis,
)
from masspairing.data import RESULTS


def main():
    G = gammas()
    C = charge_conjugation(G)
    W, _ = weyl_weight_basis(G)

    def abs2(Ms):
        out = {}
        for M in Ms:
            out = gr.add(out, gr.mul(gr.bilinear(M), gr.bilinear(M.conj(), offset=32)))
        return gr.clean(out)

    def onb(mats):
        A = np.array([m.ravel() for m in mats])
        _, s, vh = np.linalg.svd(A, full_matrices=False)
        return [vh[i].reshape(16, 16) for i in range(int((s > 1e-8).sum()))]

    A10 = abs2([W.T @ C @ g @ W for g in G])
    A126 = abs2([W.T @ C @ X @ W for X in antisym_products(G, 5)])
    Ts = {
        "J1.J1": [np.eye(16) / 4],
        "J45.J45": onb([W.conj().T @ S @ W for S in so10_generators(G)]),
        "J210.J210": onb([W.conj().T @ X @ W for X in antisym_products(G, 4)]),
    }
    polys = [A10, A126] + [gr.current_current(T) for T in Ts.values()]
    vecs = gr.coefficient_vectors(polys)
    Bm = np.array(vecs[:2]).T
    rows = []
    for name, v in zip(Ts, vecs[2:], strict=True):
        x, *_ = np.linalg.lstsq(Bm, v, rcond=None)
        resid = float(np.linalg.norm(Bm @ x - v) / np.linalg.norm(v))
        a, b = (Fraction(float(t.real)).limit_denominator(4096) for t in x)
        rows.append(
            dict(term=name, a=float(x[0].real), b=float(x[1].real), a_exact=str(a), b_exact=str(b), residual=resid)
        )
    out = {
        "basis": ["|Phi_10|^2 (10 vector bilinears)", "|Phi_126|^2 (all 252 five-forms)"],
        "normalisation": "generators orthonormal under Tr(T^dagger T), T^1 = 1/4",
        "relations": rows,
        "inverse": {"|Phi_10|^2": "-40 J1.J1 + 8 J45.J45", "|Phi_126|^2": "-432 J1.J1 - 16 J45.J45"},
    }
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "algebra_fierz.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(dst)


if __name__ == "__main__":
    main()
