"""Corner blocks of the fermion propagator and the frequency-loop readouts.

For a lattice fermion operator D the corner block at a base momentum p is
    G_B(p)_{(A, d), (B, d')} = <p + pi A, d | D^-1 | p + pi B, d'>,   A, B in {0, 1}^4,
the (16 n_int) x (16 n_int) compression of D^-1 onto the plane waves e^{i(p + pi A).x}/sqrt(V) (x) e_d. The 16 corner
shifts are the 16 Weyl components of one reduced staggered field; on a background sigma the block is exact (a
compression of D^-1, not (D_B)^-1), and the physical propagator is its ensemble average.

In the anti-hermitian (tau_2 K) class the block is anti-hermitian, its determinant is real, and the winding of det G_B
around the antiperiodic time loop vanishes identically. The loop carries the parity split under p -> -p:
    E(p) = [G_B(p) + G_B(-p)]/2 (p-even: the Majorana / Z4-odd mass part), O(p) = [G_B(p) - G_B(-p)]/2 (p-odd, kinetic),
the Majorana fraction f_M = |E|^2/(|E|^2 + |O|^2), the frequency exponent of the odd part between the two lowest shells
    alpha = ln(|O(p3)|/|O(p1)|) / ln(sin p3 / sin p1)    (-1: pole at the corner; -> +1: Luttinger zero with a large
    gap),
and the branch-resolved winding N_odd = [n - tr(sgn H(p) sgn H(-p))]/2, H = i G_B. All three are quantised only when the
gap is much larger than 2 sin(pi/L_t).
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla


# ------------------------------------------------------------------------------------------------------------
# ---
# momenta and plane waves
# ------------------------------------------------------------------------------------------------------------
# ---
def corner_shifts(d=4):
    """The 2^d corner shifts A in {0,1}^d, bit r <-> axis r (A as an integer, as in patterns.pauli_string)."""
    return [np.array([(A >> r) & 1 for r in range(d)]) for A in range(1 << d)]


def p0_loop(shape, bc):
    """The fermionic time-momentum loop p₀ = 2π(n + θ)/L_t (θ = ½ antiperiodic), n = 0 … L_t − 1, and the spatial
    base momentum nearest to zero (0 for periodic axes, +π/L for antiperiodic ones)."""
    d = len(shape)
    Lt = shape[-1]
    th = 0.5 if bc[-1] == -1 else 0.0
    loop = 2 * np.pi * (np.arange(Lt) + th) / Lt
    base = np.array([np.pi / shape[r] if bc[r] == -1 else 0.0 for r in range(d - 1)])
    return loop, base


def plane_waves(shape, p, n_int=1):
    """(n_int·V) × (2^d·n_int) matrix W of normalised plane waves e^{i(p + πA)·x}/√V ⊗ e_d (site-major, internal index
    last); column index = A n_int + d."""
    shape = tuple(shape)
    d = len(shape)
    V = int(np.prod(shape))
    coords = np.indices(shape).reshape(d, V).T.astype(float)  # (V, d)
    cols = []
    for A in corner_shifts(d):
        ph = np.exp(1j * coords @ (np.asarray(p, float) + np.pi * A)) / np.sqrt(V)  # (V,)
        for j in range(n_int):
            v = np.zeros((V, n_int), complex)
            v[:, j] = ph
            cols.append(v.reshape(-1))
    return np.stack(cols, 1)


# ------------------------------------------------------------------------------------------------------------
# ---
# operators
# ------------------------------------------------------------------------------------------------------------
# ---
def doublet_operator(lat, y, sigma, mass=0.0, h=0.0, pattern="chiral", comp=(0, 3), dual_sign=+1):
    """Sparse (2V x 2V, CSC) D_c(sigma) = K (x) 1_2 + i y sigma.tau (+ m) - h C (x) 1_2 with C =
    source_matrix(pattern)."""
    from .lattice import kinetic_matrix
    from .operator import Yukawa
    from .patterns import source_matrix

    V = lat.V
    K = kinetic_matrix(lat).astype(complex)
    Y = Yukawa(np.asarray(sigma), y).blocks()
    D = sps.kron(K, sps.eye(2), format="csr") + sps.block_diag([sps.coo_matrix(Y[0])] + list(Y[1:]), format="csr")
    if mass:
        D = D + mass * sps.eye(2 * V)
    if h:
        C = source_matrix(lat, pattern, tuple(comp), dual_sign)
        D = D - h * sps.kron(C.astype(complex), sps.eye(2), format="csr")
    return D.tocsc()


class CornerBlocks:
    """Corner blocks G_B(p) of D⁻¹ from one sparse LU (or dense LU) factorisation of D."""

    def __init__(self, D, shape, n_int=1):
        self.shape = tuple(shape)
        self.n_int = int(n_int)
        self.V = int(np.prod(self.shape))
        self.n = (1 << len(self.shape)) * self.n_int
        if sps.issparse(D):
            self.lu = spla.splu(sps.csc_matrix(D, dtype=complex))
            self._solve = self.lu.solve
        else:
            Dd = np.asarray(D, complex)
            self.lu = sla.lu_factor(Dd)
            self._solve = lambda B: sla.lu_solve(self.lu, B)
        self._cache = {}

    def block(self, p):
        key = tuple(np.round(np.asarray(p, float), 12))
        if key not in self._cache:
            W = plane_waves(self.shape, p, self.n_int)
            X = self._solve(W)
            self._cache[key] = W.conj().T @ X
        return self._cache[key]

    def loop_blocks(self, bc):
        """[G_B(p₀^n, p⃗_base)] over the time loop, the loop values and the base momentum."""
        loop, base = p0_loop(self.shape, bc)
        blocks = [self.block(np.concatenate([base, [p0]])) for p0 in loop]
        return blocks, loop, base


# ------------------------------------------------------------------------------------------------------------
# ---
# invariants
# ------------------------------------------------------------------------------------------------------------
# ---
def det_winding(blocks):
    """The winding of det G_B around the loop: Σ principal Arg(det_{n+1}/det_n)/2π; also the number of
    half-turn jumps (|Arg| > π/2, i.e. sign changes of a real determinant) and the largest |Im det|/|det| (reality).
    """
    dets = np.array([np.linalg.det(G) for G in blocks])
    ratios = np.roll(dets, -1) / dets
    args = np.angle(ratios)
    return dict(
        winding=float(args.sum() / (2 * np.pi)),
        n_half_turns=int((np.abs(args) > np.pi / 2).sum()),
        max_im_ratio=float(np.max(np.abs(dets.imag) / np.abs(dets))),
        dets=dets,
    )


def parity_split(Gp, Gm):
    """E, O, f_M for the blocks at p and −p."""
    E = (Gp + Gm) / 2
    O = (Gp - Gm) / 2
    e2, o2 = np.sum(np.abs(E) ** 2), np.sum(np.abs(O) ** 2)
    return E, O, float(e2 / (e2 + o2))


def odd_exponent(O1, O3, p1, p3):
    """α = ln(‖O(p₃)‖/‖O(p₁)‖)/ln(sin p₃/sin p₁): −1 for a pole at the corner, +1 for a zero with a large gap."""
    r = np.linalg.norm(O3) / np.linalg.norm(O1)
    s = np.sin(p3) / np.sin(p1)
    if abs(np.log(s)) < 1e-12:
        return float("nan")
    return float(np.log(r) / np.log(s))


def _sgn(H):
    w, U = np.linalg.eigh((H + H.conj().T) / 2)
    return (U * np.sign(w)) @ U.conj().T


def n_odd(Gp, Gm):
    """Branch-resolved winding [n − tr(sgn H(p) sgn H(−p))]/2 with H = iG_B (used in the anti-hermitian class;
    the hermiticity defect of H is returned as well)."""
    Hp, Hm = 1j * Gp, 1j * Gm
    herm = max(np.abs(Hp - Hp.conj().T).max(), np.abs(Hm - Hm.conj().T).max()) / max(np.abs(Hp).max(), 1e-300)
    n = Gp.shape[0]
    return float((n - np.trace(_sgn(Hp) @ _sgn(Hm)).real) / 2), float(herm)


def analyse(cb, bc, projectors=None):
    """Per-configuration record: the det winding around the loop, and at the lowest shell f_M, N_odd, the hermiticity
    defect and the anti-hermiticity defect of G_B, the norms of E and O at p₁ and p₃, and α (NaN when the two shells
    have equal |sin p₀|, i.e. L_t = 4).  `projectors`: {name: P} (n × n) restricts G_B to P G_B P (heavy/light halves).
    """
    blocks, loop, base = cb.loop_blocks(bc)
    return analyse_blocks(blocks, loop, base, projectors)


def analyse_blocks(blocks, loop, base=(0.0, 0.0, 0.0), projectors=None):
    """`analyse` on precomputed blocks [G_B(p₀^n)] over the loop values `loop` (synthetic blocks included)."""
    Lt = len(loop)
    n = blocks[0].shape[0]
    i1 = 0  # p₀ = π/L_t (antiperiodic) or 0 (periodic)
    im1 = Lt - 1  # −π/L_t
    i3, im3 = 1 % Lt, (Lt - 2) % Lt  # 3π/L_t and −3π/L_t
    p1, p3 = loop[i1], loop[i3]
    out = dict(loop=[float(v) for v in loop], base=[float(v) for v in base])
    projs = {"all": np.eye(n)}
    if projectors:
        projs.update(projectors)
    for name, P in projs.items():
        B = [P @ G @ P for G in blocks]
        dw = det_winding(B) if name == "all" else None
        Gp, Gm = B[i1], B[im1]
        E1, O1, fM = parity_split(Gp, Gm)
        E3, O3, fM3 = parity_split(B[i3], B[im3])
        nod, herm = n_odd(Gp, Gm)
        antiherm = float(np.abs(Gp + Gp.conj().T).max() / np.abs(Gp).max())
        rec = dict(
            f_M=fM,
            f_M3=fM3,
            N_odd=nod,
            herm_defect=herm,
            antiherm_defect=antiherm,
            normE1=float(np.linalg.norm(E1)),
            normO1=float(np.linalg.norm(O1)),
            normE3=float(np.linalg.norm(E3)),
            normO3=float(np.linalg.norm(O3)),
            normG1=float(np.linalg.norm(Gp)),
            normG3=float(np.linalg.norm(B[i3])),
            alpha=odd_exponent(O1, O3, p1, p3),
            p1=float(p1),
            p3=float(p3),
        )
        if dw is not None:
            rec.update(
                det_winding=dw["winding"],
                det_half_turns=dw["n_half_turns"],
                det_max_im_ratio=dw["max_im_ratio"],
                det_abs=np.abs(dw["dets"]).tolist(),
            )
        out[name] = rec
    return out


def classify(rec, f_thr=0.5, a_thr=0.0):
    """The pre-registered three-way reading of one record: 'majorana-pole' (f_M ≥ f_thr), else 'zero' (α > a_thr),
    else 'gapless' (α ≤ a_thr); 'undefined' when α is NaN and f_M < f_thr."""
    if rec["f_M"] >= f_thr:
        return "majorana-pole"
    a = rec["alpha"]
    if a != a:
        return "undefined"
    return "zero" if a > a_thr else "gapless"


def quantised(rec, n, tol_N=1.0, tol_f=0.03):
    """The pre-registered quantisation test of one record: N_odd within tol_N of 0 or n AND f_M within tol_f of 0 or
    1."""
    N = rec["N_odd"]
    f = rec["f_M"]
    return bool((min(N, n - N) <= tol_N) and (min(f, 1 - f) <= tol_f))


# ------------------------------------------------------------------------------------------------------------
# ---
# corner-space helpers (single flavour, n_int = 1)
# ------------------------------------------------------------------------------------------------------------
# ---
def corner_block_of(Msparse, shape, p, n_int=1):
    """W(p)† M W(p) for a sparse operator M (e.g. the corner block of C_χ or of K)."""
    W = plane_waves(shape, p, n_int)
    return W.conj().T @ (Msparse @ W)


def range_projectors(C_B, tol=1e-8):
    """(P_heavy, P_light): projectors onto the range and the kernel of a corner-space operator block."""
    U, s, Vh = np.linalg.svd(C_B)
    k = int((s > tol * s.max()).sum()) if s.max() > 0 else 0
    Ph = U[:, :k] @ U[:, :k].conj().T
    return Ph, np.eye(C_B.shape[0]) - Ph
