"""K1.1: the 16 (x) 16 bilinears of Spin(10) and the ranks of the 126 and 10 mass matrices on one Weyl 16."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.algebra.spin10 import bilinear_blocks, su5_singlet_126
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers

c = Check("K1.1")

# (1) span dimension and symmetry of P^T C Gamma^[k] P, k = 0..5
expected = {0: (0, None), 1: (10, "sym"), 2: (0, None), 3: (120, "anti"), 4: (0, None), 5: (126, "sym")}
blocks = {}
for k, (dim, symm) in expected.items():
    B = bilinear_blocks(k)
    blocks[k] = B
    rk = int(np.linalg.matrix_rank(np.array([m.ravel() for m in B]), tol=1e-8))
    sym_ok = True
    if symm == "sym":
        sym_ok = all(np.allclose(m, m.T) for m in B)
    if symm == "anti":
        sym_ok = all(np.allclose(m, -m.T) for m in B)
    c.item(f"k = {k}: span dimension (expected {dim}, {symm or 'vanishes'})", rk, rk == dim and sym_ok)
c.item(
    "Sym^2(16) = 10 + 126 and Lambda^2(16) = 120 exhaust 16 x 16 (dimensions)",
    10 + 126 + 120,
    10 + 126 + 120 == 16 * 16,
)

# (2) a generic 126 vev has full rank 16
rng = np.random.default_rng(20260925)
smin, ranks = [], []
for _ in range(500):
    w = rng.normal(size=252) + 1j * rng.normal(size=252)
    M = sum(cw * m for cw, m in zip(w, blocks[5], strict=True))
    ranks.append(int(np.linalg.matrix_rank(M, tol=1e-8)))
    smin.append(np.linalg.svd(M, compute_uv=False)[-1] / np.linalg.norm(M))
c.item("generic 126 vev (500 random complex five-form coefficients): minimal rank", min(ranks), min(ranks) == 16)
c.item("  smallest normalised singular value over the 500 samples", min(smin), min(smin) > 1e-4, fmt="{:.2e}")

# (3) the SU(5)-singlet direction of the 126 has rank 1
M = su5_singlet_126()
r1 = int(np.linalg.matrix_rank(M, tol=1e-8))
c.item("SU(5)-singlet 126 vev (Omega = wedge_k (e_2k + i e_2k+1)): rank", r1, r1 == 1 and np.linalg.norm(M) > 1e-8)

# (4) a real 10 vev gives all 16 the same mass
rng = np.random.default_rng(4)
spread, rmin = 0.0, 16
for _ in range(20):
    v = rng.normal(size=10)
    M = sum(cv * m for cv, m in zip(v, blocks[1], strict=True))
    s = np.linalg.svd(M, compute_uv=False)
    spread = max(spread, (s.max() - s.min()) / s.max())
    rmin = min(rmin, int(np.linalg.matrix_rank(M, tol=1e-8)))
c.item(
    "real 10 vev (20 random v in R^10): max relative spread of the 16 singular values",
    spread,
    spread < 1e-12,
    fmt="{:.1e}",
)
c.item("  rank", rmin, rmin == 16)
c.done()
