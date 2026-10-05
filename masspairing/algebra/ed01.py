"""0+1D exact diagonalisation of 8 light (a) + 8 heavy (b) Majoranas with Fidkowski-Kitaev (Cayley 4-form) quartics, a
light-heavy quartic X, a heavy mass M and the Higgs-like symmetry-breaking term Phi (claim K5.10):

    H = U_a W(a) + U_b W(b) + lambda X(a, b) + M sum_{j=1..4} i b_{2j-1} b_{2j} + Phi sum_k i a_k b_k,
    X = sum_{k<l, k = l mod 2} (i a_k a_l)(i b_k b_l).

Symmetries (verified as operator identities in ``symmetry_checks``): T (antiunitary, every Majorana even) forbids every
bilinear; R_b (b_even -> -b_even) commutes with W and X; M breaks T and R_b and keeps T' = R_b T; Phi breaks T'.

Pre-registered scans and clauses (fixed before the code existed):
  Z   U_a = U_b = 1, lambda in {-1, -0.5, 0, 0.5, 1}, M in {0 ... 100}, Phi = 0: unique ground state, every light
      bilinear and the mass-type (omega-even) part of the light Green's function below 1e-10, Delta_L(100)/Delta_L(10)
      > 0.5;
  Z'  U_a = 0 (no light quartic), lambda in {0.5, 1}: the same zero clauses;
  S1  Phi = 0.3, lambda in {0, 0.5}: the mass-type part must exceed 1e-3 at every M > 0 (the zero-sabotage control);
  S2  Phi = 0.3, U_a = lambda = 0: a seesaw light gap, d ln Delta_L / d ln M in [-1.2, -0.8] over M = 10 ... 100.
  The verdict PASS needs all four; Z/Z' are void unless S1 and S2 fire.
"""

from __future__ import annotations

import numpy as np

X2 = np.array([[0, 1], [1, 0]], complex)
Y2 = np.array([[0, -1j], [1j, 0]], complex)
Z2 = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)

# Cayley 4-form (Spin(7)); 1-based indices, signs as listed (checked by the unique ground state of W)
CAYLEY = [
    ((1, 2, 3, 4), +1),
    ((1, 2, 5, 6), +1),
    ((1, 2, 7, 8), +1),
    ((1, 3, 5, 7), +1),
    ((1, 3, 6, 8), -1),
    ((1, 4, 5, 8), -1),
    ((1, 4, 6, 7), -1),
    ((2, 3, 5, 8), -1),
    ((2, 3, 6, 7), -1),
    ((2, 4, 5, 7), -1),
    ((2, 4, 6, 8), +1),
    ((3, 4, 5, 6), +1),
    ((3, 4, 7, 8), +1),
    ((5, 6, 7, 8), +1),
]
MS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LAMS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def majoranas(nmodes):
    """Jordan-Wigner Majoranas gamma_1 ... gamma_{2n} (hermitian, {gamma_i, gamma_j} = 2 delta_ij) on 2^n states."""
    out = []
    for m in range(nmodes):
        for P in (X2, Y2):
            ops = [Z2] * m + [P] + [I2] * (nmodes - m - 1)
            M = ops[0]
            for o in ops[1:]:
                M = np.kron(M, o)
            out.append(M)
    return out


def fk(g, sign=1.0):
    """W = sign sum Omega_ijkl gamma_i gamma_j gamma_k gamma_l over the Cayley terms (g: 8 Majoranas)."""
    W = 0
    for (i, j, k, l), s in CAYLEY:
        W = W + s * (g[i - 1] @ g[j - 1] @ g[k - 1] @ g[l - 1])
    return sign * W


G16 = majoranas(8)
A = G16[:8]
B = G16[8:]
_W_SIGN = None
_CACHE = {}


def w_sign():
    """The sign of W for which 8 Majoranas alone have a unique ground state (checked, not assumed)."""
    global _W_SIGN
    if _W_SIGN is None:
        g8 = majoranas(4)
        best = None
        for s in (+1.0, -1.0):
            e = np.linalg.eigvalsh(fk(g8, s))
            if e[1] - e[0] > 1e-9:
                best = s if best is None else best
        assert best is not None, "no sign of the Cayley quartic gives a unique ground state"
        _W_SIGN = best
    return _W_SIGN


def x_term():
    X = 0
    for k in range(8):
        for l in range(k + 1, 8):
            if (k - l) % 2 == 0:
                X = X + (1j * A[k] @ A[l]) @ (1j * B[k] @ B[l])
    return X


def pieces():
    if not _CACHE:
        s = w_sign()
        _CACHE["Wa"] = fk(A, s)
        _CACHE["Wb"] = fk(B, s)
        _CACHE["X"] = x_term()
        _CACHE["Mb"] = sum(1j * B[2 * j] @ B[2 * j + 1] for j in range(4))
        _CACHE["Phi"] = sum(1j * A[k] @ B[k] for k in range(8))
    return _CACHE


def hamiltonian(Ua=1.0, Ub=1.0, lam=0.0, M=0.0, Phi=0.0):
    P = pieces()
    H = Ua * P["Wa"] + Ub * P["Wb"] + lam * P["X"] + M * P["Mb"] + Phi * P["Phi"]
    assert np.allclose(H, H.conj().T)
    return H


def solve(H, omegas=(0.01, 0.1, 1.0)):
    """Ground-state split, light gap, largest light bilinear and the mass-type (omega-even) part of the light Green's
    function, G_kl(i omega) = sum_n [M_kl/(i omega - E_n) + M_lk/(i omega + E_n)], M_kl = <0|a_k|n><n|a_l|0>; its even
    part is sum_n E_n (M_lk - M_kl)/(omega^2 + E_n^2)."""
    E, V = np.linalg.eigh(H)
    g0 = V[:, 0]
    out = dict(E0=float(E[0]), split=float(E[1] - E[0]))
    amp = np.array([V.conj().T @ (a @ g0) for a in A])
    w = np.sum(np.abs(amp) ** 2, axis=0)
    En = E - E[0]
    sel = (w > 1e-10) & (En > 1e-12)
    out["gap_L"] = float(En[sel].min()) if sel.any() else 0.0
    out["bil_L"] = float(max(abs(np.vdot(g0, 1j * A[k] @ A[l] @ g0)) for k in range(8) for l in range(k + 1, 8)))
    Mkl = np.einsum("kn,ln->kln", amp.conj(), amp)
    ev = {}
    for om in omegas:
        f = En / (om**2 + En**2)
        Ge = np.einsum("kln,n->kl", Mkl.transpose(1, 0, 2) - Mkl, f)
        ev[str(om)] = float(np.max(np.abs(Ge)))
    out["even_L"] = ev
    return out


def even_odd_ratio(H, om=0.01):
    """max |omega-even part| / max |omega-odd part| of the light Green's function at frequency om."""
    Ev, V = np.linalg.eigh(H)
    g0 = V[:, 0]
    amp = np.array([V.conj().T @ (a @ g0) for a in A])
    En = Ev - Ev[0]
    Mkl = np.einsum("kn,ln->kln", amp.conj(), amp)
    Ge = np.einsum("kln,n->kl", Mkl.transpose(1, 0, 2) - Mkl, En / (om**2 + En**2))
    Go = np.einsum("kln,n->kl", Mkl + Mkl.transpose(1, 0, 2), om / (om**2 + En**2))
    return float(np.abs(Ge).max() / np.abs(Go).max())


def symmetry_checks():
    """T = U K with U gamma* U^dag = gamma for every Majorana; R_b the product of the four b_even; T' = R_b T."""
    P = pieces()
    R = np.eye(2**8, dtype=complex)
    for j in range(4):
        R = R @ B[2 * j + 1]
    U = np.eye(2**8, dtype=complex)
    for g in G16[1::2]:
        U = U @ g
    T_maj = all(np.allclose(U @ g.conj() @ U.conj().T, g) for g in G16)

    def T(Hm):
        return U @ Hm.conj() @ U.conj().T

    def Tp(Hm):
        return R @ T(Hm) @ R.conj().T

    return dict(
        T_majoranas_even=bool(T_maj),
        T_Wa=bool(np.allclose(T(P["Wa"]), P["Wa"])),
        T_X=bool(np.allclose(T(P["X"]), P["X"])),
        T_Mb_odd=bool(np.allclose(T(P["Mb"]), -P["Mb"])),
        Tp_Mb_even=bool(np.allclose(Tp(P["Mb"]), P["Mb"])),
        Tp_Phi_broken=bool(not np.allclose(Tp(P["Phi"]), P["Phi"])),
        Tp_light_bilinears_odd=bool(
            all(np.allclose(Tp(1j * A[k] @ A[l]), -1j * A[k] @ A[l]) for k in range(8) for l in range(k + 1, 8))
        ),
        R_comm_Wb=bool(np.allclose(R @ P["Wb"], P["Wb"] @ R)),
        R_comm_X=bool(np.allclose(R @ P["X"], P["X"] @ R)),
        R_comm_Wa=bool(np.allclose(R @ P["Wa"], P["Wa"] @ R)),
        R_anti_Mb=bool(np.allclose(R @ P["Mb"], -P["Mb"] @ R)),
    )


# ------------------------------------------------------------ the pre-registered scans
def scan(Ua, lams, Phi, Ms=MS):
    return {str(lam): {str(M): solve(hamiltonian(Ua=Ua, Ub=1.0, lam=lam, M=M, Phi=Phi)) for M in Ms} for lam in lams}


def slope(rows, Ms=(10.0, 30.0, 100.0)):
    g = np.array([rows[str(M)]["gap_L"] for M in Ms])
    return float(np.polyfit(np.log(Ms), np.log(g), 1)[0])


def zero_clauses(sc):
    worst = dict(split=np.inf, bil=0.0, even=0.0)
    for rows in sc.values():
        for r in rows.values():
            worst["split"] = min(worst["split"], r["split"])
            worst["bil"] = max(worst["bil"], r["bil_L"])
            worst["even"] = max(worst["even"], max(r["even_L"].values()))
    ok = worst["split"] > 1e-8 and worst["bil"] < 1e-10 and worst["even"] < 1e-10
    return ok, worst


def analyse():
    res = dict(sym=symmetry_checks(), w_sign=w_sign())
    res["Z"] = scan(1.0, LAMS, 0.0)
    res["Zp"] = scan(0.0, (0.5, 1.0), 0.0)
    res["S1"] = scan(1.0, (0.0, 0.5), 0.3, Ms=(0.0, 1.0, 10.0, 100.0))
    res["S2"] = scan(0.0, (0.0,), 0.3, Ms=(10.0, 30.0, 100.0))
    v = {}
    okZ, wZ = zero_clauses(res["Z"])
    ratios = {lam: res["Z"][lam]["100.0"]["gap_L"] / res["Z"][lam]["10.0"]["gap_L"] for lam in res["Z"]}
    v["Z"] = dict(ok=bool(okZ and min(ratios.values()) > 0.5), worst=wZ, ratio_100_10=ratios)
    okZp, wZp = zero_clauses(res["Zp"])
    v["Zp"] = dict(ok=bool(okZp), worst=wZp, slopes={lam: slope(rows) for lam, rows in res["Zp"].items()})
    s1all = {lam: min(r["even_L"]["0.01"] for M, r in rows.items() if float(M) > 0) for lam, rows in res["S1"].items()}
    v["S1"] = dict(ok=bool(min(s1all.values()) > 1e-3), min_even_M_pos=s1all)
    sl = slope(res["S2"]["0.0"])
    v["S2"] = dict(ok=bool(-1.2 <= sl <= -0.8), slope=sl)
    v["direction"] = {lam: {M: rows[M]["gap_L"] / rows["0.0"]["gap_L"] for M in rows} for lam, rows in res["Z"].items()}
    v["PASS"] = bool(v["Z"]["ok"] and v["Zp"]["ok"] and v["S1"]["ok"] and v["S2"]["ok"])
    res["verdict"] = v
    return res


def random_tprime(n_trials=12, seed=20261004):
    """Random T'-invariant Hamiltonians (40 random real quartics with an even number of b_even factors, random T'-even
    bilinears i b_odd b_even and i a_k b_even), and the same plus the T'-odd bilinear 0.3 i a_1 b_1. Returns
    (number with a unique ground state, worst light bilinear / mass-type part among them, smallest mass-type part with
    the T'-odd term)."""
    import itertools

    rng = np.random.default_rng(seed)
    G = G16
    idx_beven = {8 + 1, 8 + 3, 8 + 5, 8 + 7}
    quads = [q for q in itertools.combinations(range(16), 4) if len(idx_beven.intersection(q)) % 2 == 0]
    worst_sym, best_brk, n_unique = 0.0, np.inf, 0
    for _ in range(n_trials):
        H = 0
        for q in rng.choice(len(quads), 40, replace=False):
            i, j, k, l = quads[q]
            H = H + rng.normal() * (G[i] @ G[j] @ G[k] @ G[l])
        for i, j in [(8 + 2 * m, 8 + 2 * m + 1) for m in range(4)] + [
            (k, 8 + 2 * m + 1) for k in range(8) for m in range(4)
        ]:
            H = H + rng.normal() * 1j * G[i] @ G[j]
        r = solve(H)
        if r["split"] > 1e-8:
            n_unique += 1
            worst_sym = max(worst_sym, r["bil_L"], max(r["even_L"].values()))
        rb = solve(H + 0.3 * 1j * G[0] @ G[8])
        best_brk = min(best_brk, max(rb["even_L"].values()))
    return n_unique, worst_sym, best_brk
