"""The 2+1d Hamiltonian version of the four-flavour model on small clusters: Jordan-Wigner operators, the BdG ->
Majorana conversion, Hubbard-Stratonovich ensembles and their Majorana-time-reversal (MTR) classification.

Model: four complex fermions xi^a(r) per site,

    H = -t sum_<rr'> sum_a (xi^a+(r) xi^a(r') + h.c.) + V sum_r (xi^1 xi^2 xi^3 xi^4 (r) + h.c.)
        - g sum_<rr'> sum_ab Phi^ab(r, r')+ Phi^ab(r, r'),     Phi^ab(r, r') = xi^a(r) xi^b(r') + xi^b(r) xi^a(r').

MTR criterion (Li-Jiang-Yao, PRL 117, 267002 (2016); Wei-Wu-Li-Yao, PRL 116, 250601 (2016)). A decoupled factor is
exp(c O), O = (i/4) gamma^T A gamma hermitian (A real antisymmetric in the Majorana basis), c real ("herm") or
imaginary ("anti"). T = U K (U real orthogonal) commutes with it iff U A U^T = sigma A, sigma = -1 (herm) or +1
(anti). L = {X : X A_j = sigma_j A_j X for all j} is a module over the commutant C = {X : X A_j = A_j X}; the MTR
symmetries are the orthogonal elements of L with U^T = +U (T^2 = +1) or U^T = -U (T^2 = -1), and any invertible
(anti)symmetric element of L polarises to an orthogonal one inside L. The sign-free classes {T+, T-} (Majorana) and
{T-, T-} (Kramers) both need a T- and an orthogonal Q in C anticommuting with it (Q symmetric: Majorana class,
antisymmetric: Kramers class).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .staggered import NF, su4_generators


# ---- Jordan-Wigner cluster ----------------------------------------------------------------------------------
def jw_ops(nmodes):
    """Annihilation operators c_k as dense 2^n x 2^n matrices (Jordan-Wigner)."""
    I = np.eye(2)
    Z = np.diag([1.0, -1.0])
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = []
    for k in range(nmodes):
        f = [Z] * k + [a] + [I] * (nmodes - k - 1)
        m = f[0]
        for ff in f[1:]:
            m = np.kron(m, ff)
        ops.append(m)
    return ops


def majorana_ops(cs):
    """gamma_{2k} = c_k + c_k^+, gamma_{2k+1} = -i (c_k - c_k^+)."""
    g = []
    for c in cs:
        g.append(c + c.conj().T)
        g.append(-1j * (c - c.conj().T))
    return g


def quadratic_from_A(A, gam):
    """(i/4) sum_pq A_pq gamma_p gamma_q as a dense operator."""
    n = A.shape[0]
    H = np.zeros_like(gam[0])
    for p in range(n):
        for q in range(n):
            if A[p, q] != 0:
                H = H + 0.25j * A[p, q] * (gam[p] @ gam[q])
    return H


def cluster_ops(nsites, bonds, t=1.0):
    """Dense operators of the cluster: xi(s, a), H_band, H_eps = sum_s (xi1 xi2 xi3 xi4 + h.c.), D_s = xi1 xi2 + xi3 xi4
    (D^2 = 2 xi1 xi2 xi3 xi4) and Phi^ab(s, u) for a <= b on every bond."""
    nm = NF * nsites
    cs = jw_ops(nm)
    xi = lambda s, a: cs[s * NF + a]
    dim = 2**nm
    Hband = np.zeros((dim, dim), complex)
    for s, u in bonds:
        for a in range(NF):
            Hband += -t * (xi(s, a).conj().T @ xi(u, a) + xi(u, a).conj().T @ xi(s, a))
    Heps = np.zeros_like(Hband)
    Dops = []
    for s in range(nsites):
        q = xi(s, 0) @ xi(s, 1) @ xi(s, 2) @ xi(s, 3)
        Heps += q + q.conj().T
        Dops.append(xi(s, 0) @ xi(s, 1) + xi(s, 2) @ xi(s, 3))
    Phis = {}
    for s, u in bonds:
        for a in range(NF):
            for b in range(a, NF):
                Phis[(s, u, a, b)] = xi(s, a) @ xi(u, b) + xi(s, b) @ xi(u, a)
    return dict(cs=cs, xi=xi, Hband=Hband, Heps=Heps, Dops=Dops, Phis=Phis, dim=dim)


def h10_operator(ops, g=1.0):
    """H_10 = -g sum_bonds sum_ab Phi^ab+ Phi^ab (attractive for g > 0; a != b counted twice)."""
    H = np.zeros((ops["dim"], ops["dim"]), complex)
    for k, P in ops["Phis"].items():
        mult = 1.0 if k[2] == k[3] else 2.0
        H += -g * mult * (P.conj().T @ P)
    return H


def parity_sectors(dim):
    """Index sets of the even and odd fermion-parity states of the Jordan-Wigner basis (parity = popcount of the index).
    Every bilinear, every quartic and every exponential of them is block diagonal in these sectors."""
    par = np.array([bin(i).count("1") % 2 for i in range(dim)])
    return [np.flatnonzero(par == 0), np.flatnonzero(par == 1)]


class ExpOp:
    """exp(c O) for a fixed hermitian, fermion-parity-even O and any complex c, from one eigendecomposition per parity
    sector."""

    def __init__(self, O, sectors):
        assert np.allclose(O, O.conj().T)
        self.sectors = sectors
        assert abs(O[np.ix_(sectors[0], sectors[1])]).max() == 0.0, "operator mixes the parity sectors"
        self.blocks = []
        for s in sectors:
            w, v = eigh(O[np.ix_(s, s)])
            self.blocks.append((w, v, v.conj().T))

    def __call__(self, c):
        """The full matrix exp(c O)."""
        n = sum(len(s) for s in self.sectors)
        out = np.zeros((n, n), complex)
        for s, (w, v, vh) in zip(self.sectors, self.blocks, strict=True):
            out[np.ix_(s, s)] = (v * np.exp(c * w)) @ vh
        return out

    def right(self, Mt, c):
        """Mt exp(c O) for Mt given as its list of parity blocks."""
        return [((M @ v) * np.exp(c * w)) @ vh for M, (w, v, vh) in zip(Mt, self.blocks, strict=True)]


def trace_weights(sampler, nsamp, ntau, eB, sectors):
    """Exact HS weights Tr prod_tau [prod_i E_i(c_i) e^{-dtau H_band}] for ``nsamp`` draws of sampler() -> [(E, c)],
    computed blockwise in the two parity sectors."""
    eBb = [eB[np.ix_(s, s)] for s in sectors]
    ws = []
    for _ in range(nsamp):
        Mt = [np.eye(len(s), dtype=complex) for s in sectors]
        for _tau in range(ntau):
            for E_, c in sampler():
                Mt = E_.right(Mt, c)
            Mt = [M @ e for M, e in zip(Mt, eBb, strict=True)]
        ws.append(sum(np.trace(M) for M in Mt))
    return np.array(ws)


# ---- BdG data and the Majorana matrix ---------------------------------------------------------------------------
def majorana_matrix_from_bdg(h, Delta):
    """Real antisymmetric A with c^+ h c + 1/2 (c^+ Delta c^+T + h.c.) = (i/4) gamma^T A gamma + const
    (c_k = (gamma_{2k} + i gamma_{2k+1}) / 2; h hermitian, Delta antisymmetric)."""
    n = h.shape[0]
    C = np.zeros((2 * n, 2 * n), complex)

    def add(p, q, val):
        C[p, q] += val

    for j in range(n):
        for k in range(n):
            v = h[j, k]
            if v != 0:
                for p, cp in ((2 * j, 1.0), (2 * j + 1, -1j)):
                    for q, cq in ((2 * k, 1.0), (2 * k + 1, 1j)):
                        add(p, q, v * cp * cq / 4)
            d = Delta[j, k]
            if d != 0:
                for p, cp in ((2 * j, 1.0), (2 * j + 1, -1j)):
                    for q, cq in ((2 * k, 1.0), (2 * k + 1, -1j)):
                        add(p, q, 0.5 * d * cp * cq / 4)
                for p, cp in ((2 * k, 1.0), (2 * k + 1, 1j)):
                    for q, cq in ((2 * j, 1.0), (2 * j + 1, 1j)):
                        add(p, q, 0.5 * np.conj(d) * cp * cq / 4)
    Cas = C - C.T
    A = -2j * Cas
    assert np.allclose(A.imag, 0), "Majorana matrix not real: operator not hermitian?"
    return A.real


def bdg_band(nsites, bonds, t=1.0):
    n = NF * nsites
    h = np.zeros((n, n), complex)
    for s, u in bonds:
        for a in range(NF):
            h[s * NF + a, u * NF + a] += -t
            h[u * NF + a, s * NF + a] += -t
    return h, np.zeros((n, n), complex)


def _pair(Delta, j, k, c):
    """Add c xi_j^+ xi_k^+ (j != k) to Delta (1/2 sum Delta xi^+ xi^+ convention)."""
    Delta[j, k] += c
    Delta[k, j] -= c


def bdg_eps_XY(nsites, site):
    """X_s = D_s + D_s^+ and Y_s = i (D_s - D_s^+), D_s = xi1 xi2 + xi3 xi4."""
    n = NF * nsites
    hX = np.zeros((n, n), complex)
    DX = np.zeros((n, n), complex)
    hY = np.zeros((n, n), complex)
    DY = np.zeros((n, n), complex)
    o = site * NF
    _pair(DX, o + 0, o + 1, -1.0)
    _pair(DX, o + 2, o + 3, -1.0)
    _pair(DY, o + 0, o + 1, 1j)
    _pair(DY, o + 2, o + 3, 1j)
    return (hX, DX), (hY, DY)


def bdg_phi_pairing(nsites, s, u, a, b, z):
    """z Phi^ab(s, u)^+ + conj(z) Phi^ab(s, u)."""
    n = NF * nsites
    h = np.zeros((n, n), complex)
    D = np.zeros((n, n), complex)
    _pair(D, s * NF + a, u * NF + b, -z)
    _pair(D, s * NF + b, u * NF + a, -z)
    return h, D


def bdg_phi_commutator(nsites, s, u, a, b):
    """[Phi^ab(s, u), Phi^ab(s, u)^+] up to a constant: -(1 + delta_ab) [n_a(s) + n_b(s) + n_a(u) + n_b(u)]."""
    n = NF * nsites
    h = np.zeros((n, n), complex)
    for site in (s, u):
        for f in {a, b}:
            h[site * NF + f, site * NF + f] += -4.0 if a == b else -1.0
    return h, np.zeros((n, n), complex)


def bdg_density(nsites, site, coeff=1.0):
    n = NF * nsites
    h = np.zeros((n, n), complex)
    for a in range(NF):
        h[site * NF + a, site * NF + a] = coeff
    return h, np.zeros((n, n), complex)


def bdg_flavour_generator(nsites, site, T):
    """sum_ab T_ab xi_a^+(site) xi_b(site) for a hermitian 4 x 4 T."""
    n = NF * nsites
    h = np.zeros((n, n), complex)
    h[site * NF : (site + 1) * NF, site * NF : (site + 1) * NF] = T
    return h, np.zeros((n, n), complex)


def bdg_bond_hopping(nsites, s, u, z):
    """z K + conj(z) K^+ with K = sum_a xi_a^+(s) xi_a(u)."""
    n = NF * nsites
    h = np.zeros((n, n), complex)
    for a in range(NF):
        h[s * NF + a, u * NF + a] += z
        h[u * NF + a, s * NF + a] += np.conj(z)
    return h, np.zeros((n, n), complex)


# ---- MTR classification -----------------------------------------------------------------------------------------
def _null_space_gram(rows, n2, tol=1e-9):
    Gm = np.zeros((n2, n2))
    for R in rows:
        Gm += R.T @ R
    w, v = np.linalg.eigh(Gm)
    return v[:, w < tol * max(1.0, w.max())]


def mtr_module(ensemble, tol=1e-9):
    """(basis of L, basis of C) as lists of n x n matrices for an ensemble [(A, "herm" | "anti")]."""
    n = ensemble[0][0].shape[0]
    I = np.eye(n)
    rowsL, rowsC = [], []
    for A, kind in ensemble:
        sigma = -1.0 if kind == "herm" else 1.0
        rowsL.append(np.kron(A.T, I) - sigma * np.kron(I, A))  # vec(X A - sigma A X), column-major vec
        rowsC.append(np.kron(A.T, I) - np.kron(I, A))
    NL = _null_space_gram(rowsL, n * n, tol)
    NC = _null_space_gram(rowsC, n * n, tol)
    toM = lambda N: [N[:, k].reshape(n, n, order="F") for k in range(N.shape[1])]
    return toM(NL), toM(NC)


def _sym_anti_split(basis):
    sym = [(B + B.T) / 2 for B in basis]
    anti = [(B - B.T) / 2 for B in basis]

    def indep(mats):
        if not mats:
            return []
        A = np.array([m.ravel() for m in mats])
        _, s, vt = np.linalg.svd(A, full_matrices=False)
        r = int(np.sum(s > 1e-9 * max(1.0, s.max())))
        n = mats[0].shape[0]
        return [vt[k].reshape(n, n) for k in range(r)]

    return indep(sym), indep(anti)


def _sqrtm_spd(P):
    w, v = np.linalg.eigh((P + P.T) / 2)
    return (v * np.sqrt(np.maximum(w, 0))) @ v.T


def _orthogonal_element(basis, rng, trials=20):
    """Polar part of a random element of a space of (anti)symmetric matrices, if a generic element is invertible."""
    if not basis:
        return None
    n = basis[0].shape[0]
    for _ in range(trials):
        X = sum(rng.normal() * B for B in basis)
        s = np.linalg.svd(X, compute_uv=False)
        if s.min() < 1e-8 * s.max():
            continue
        U = X @ np.linalg.inv(_sqrtm_spd(X.T @ X))
        if np.allclose(U @ U.T, np.eye(n), atol=1e-8):
            return U
    return None


def mtr_classify(ensemble, rng=None, tol=1e-9):
    """dims of L_sym, L_anti, C; a T+ and a T- if they exist; the sign-free class ("Majorana", "Kramers") or None.

    When a class is found, ``U1`` (the T-) and ``Q`` (orthogonal in C, anticommuting with U1) are returned too."""
    rng = rng or np.random.default_rng(0)
    Lb, Cb = mtr_module(ensemble, tol)
    Ls, La = _sym_anti_split(Lb)
    Cs, Ca = _sym_anti_split(Cb)
    res = {"dimL_sym": len(Ls), "dimL_anti": len(La), "dimC": len(Cb), "Tplus": None, "Tminus": None, "cls": None}
    res["Tplus"] = _orthogonal_element(Ls, rng)
    res["Tminus"] = _orthogonal_element(La, rng)
    if res["Tminus"] is None:
        return res
    for _ in range(8):
        U1 = _orthogonal_element(La, rng)
        if U1 is None:
            break

        def anticomm_subspace(basis, U1=U1):
            if not basis:
                return []
            A = np.array([(B @ U1 + U1 @ B).ravel() for B in basis]).T
            _, ss, vv = np.linalg.svd(A, full_matrices=True)
            r = int(np.sum(ss > 1e-8 * max(1.0, ss.max() if ss.size else 1.0)))
            return [sum(c[k] * basis[k] for k in range(len(basis))) for c in vv[r:]]

        for name, mats in (("Majorana", Cs), ("Kramers", Ca)):
            Q = _orthogonal_element(anticomm_subspace(mats), rng)
            if Q is not None:
                res.update(cls=name, Q=Q, U1=U1)
                return res
    return res


# ---- HS ensembles at the single-particle level ---------------------------------------------------------------
def hs_ensembles(nsites, bonds, t=1.0):
    """Majorana matrices (A, kind) of the band term and of the bilinears of every HS scheme: dict of lists.

    band; eps_pos / eps_neg (He et al. 2016: V > 0 -> i s X, s Y; V < 0 -> s X, i s Y); pairing (scheme A: real
    couplings to Phi + Phi^+ and i(Phi^+ - Phi), plus the fixed commutator bilinear of the product HS);
    dens_exch (scheme B: s(n + n'), i s(n - n'), i s X_K, i s Y_K, fixed [K, K^+] = n - n'); spin (scheme C:
    s(S^A + S'^A), i s(S^A - S'^A) for the 16 u(4) generators)."""
    mm = lambda hD: majorana_matrix_from_bdg(*hD)
    band = [(mm(bdg_band(nsites, bonds, t)), "herm")]
    eps_pos, eps_neg = [], []
    for s in range(nsites):
        (hX, DX), (hY, DY) = bdg_eps_XY(nsites, s)
        AX, AY = mm((hX, DX)), mm((hY, DY))
        eps_pos += [(AX, "anti"), (AY, "herm")]
        eps_neg += [(AX, "herm"), (AY, "anti")]
    pairing = []
    for s, u in bonds:
        for a in range(NF):
            for b in range(a, NF):
                pairing.append((mm(bdg_phi_pairing(nsites, s, u, a, b, 1.0)), "herm"))
                pairing.append((mm(bdg_phi_pairing(nsites, s, u, a, b, 1j)), "herm"))
                pairing.append((mm(bdg_phi_commutator(nsites, s, u, a, b)), "herm"))
    dens_exch = []
    for s, u in bonds:
        hn = bdg_density(nsites, s)[0] + bdg_density(nsites, u)[0]
        hd = bdg_density(nsites, s)[0] - bdg_density(nsites, u)[0]
        z = np.zeros_like(hn)
        dens_exch.append((mm((hn, z)), "herm"))
        dens_exch.append((mm((hd, z)), "anti"))
        dens_exch.append((mm(bdg_bond_hopping(nsites, s, u, 1.0)), "anti"))
        dens_exch.append((mm(bdg_bond_hopping(nsites, s, u, 1j)), "anti"))
        dens_exch.append((mm((hd, z)), "herm"))
    spin = []
    gens = su4_generators() + [np.eye(4)]
    for s, u in bonds:
        for T in gens:
            hp = bdg_flavour_generator(nsites, s, T)[0] + bdg_flavour_generator(nsites, u, T)[0]
            hm = bdg_flavour_generator(nsites, s, T)[0] - bdg_flavour_generator(nsites, u, T)[0]
            z = np.zeros_like(hp)
            spin.append((mm((hp, z)), "herm"))
            spin.append((mm((hm, z)), "anti"))
    return dict(band=band, eps_pos=eps_pos, eps_neg=eps_neg, pairing=pairing, dens_exch=dens_exch, spin=spin)
