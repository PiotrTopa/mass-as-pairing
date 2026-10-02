"""K2.2: the pure Sym^2 (10-channel) term has no sign-free formulation found -- every natural Hamiltonian HS decoupling
is in a sign-problematic MTR class, the fermion-bag weights are positive only empirically (link monomials go negative),
real flavour-blind HS fields are excluded by a moment obstruction and the 10-plet Yukawa Pfaffian is indefinite --
while the completed term T_Q + sum_{l in Q} K_l^2 is a sum of squares with weight Pf(M - A(s))^4 = det^2 >= 0."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import itertools

import numpy as np
from scipy.linalg import expm

from masspairing.algebra.bags import bag_weights, kmonomial_table, kmonomial_weight, link_pfaffians
from masspairing.algebra.grassmann import Grass, berezin_integral, pfaffian, substitute, wick
from masspairing.algebra.hamiltonian import (
    ExpOp,
    bdg_band,
    bdg_bond_hopping,
    bdg_eps_XY,
    bdg_flavour_generator,
    bdg_phi_commutator,
    bdg_phi_pairing,
    cluster_ops,
    h10_operator,
    hs_ensembles,
    majorana_matrix_from_bdg,
    majorana_ops,
    mtr_classify,
    parity_sectors,
    quadratic_from_A,
    trace_weights,
)
from masspairing.algebra.staggered import (
    FieldAlgebra,
    WickCache,
    line_quads,
    oriented_links,
    plaquette_quads,
    random_su4,
    staggered_matrix,
    su4_generators,
    sym2_yukawa_matrix,
)
from masspairing.claimcheck import Check

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K2.2")

# =============================================================================================================
# A. 2+1d Hamiltonian: MTR classes and exact HS weights
# =============================================================================================================
rng = np.random.default_rng(21)
ops = cluster_ops(2, [(0, 1)])
cs, xi, dim = ops["cs"], ops["xi"], ops["dim"]
gam = majorana_ops(cs)


def upto_const(op, A):
    diff = op - quadratic_from_A(A, gam)
    return np.abs(diff - np.trace(diff) / dim * np.eye(dim)).max()


D0 = ops["Dops"][0]
(hX, DX), (hY, DY) = bdg_eps_XY(2, 0)
z = 0.7 + 0.3j
K = sum(xi(0, a).conj().T @ xi(1, a) for a in range(4))
T3 = su4_generators()[3]
conv = [
    upto_const(ops["Hband"], majorana_matrix_from_bdg(*bdg_band(2, [(0, 1)]))),
    upto_const(D0 + D0.conj().T, majorana_matrix_from_bdg(hX, DX)),
    upto_const(1j * (D0 - D0.conj().T), majorana_matrix_from_bdg(hY, DY)),
    upto_const(
        z * ops["Phis"][(0, 1, 1, 2)].conj().T + np.conj(z) * ops["Phis"][(0, 1, 1, 2)],
        majorana_matrix_from_bdg(*bdg_phi_pairing(2, 0, 1, 1, 2, z)),
    ),
    upto_const(z * K + np.conj(z) * K.conj().T, majorana_matrix_from_bdg(*bdg_bond_hopping(2, 0, 1, z))),
    upto_const(
        sum(T3[a, b] * xi(0, a).conj().T @ xi(0, b) for a in range(4) for b in range(4)),
        majorana_matrix_from_bdg(*bdg_flavour_generator(2, 0, T3)),
    ),
]
for k_ in ((0, 1, 0, 0), (0, 1, 1, 2)):
    P_ = ops["Phis"][k_]
    conv.append(upto_const(P_ @ P_.conj().T - P_.conj().T @ P_, majorana_matrix_from_bdg(*bdg_phi_commutator(2, *k_))))
c.item(
    "A0 BdG -> Majorana converters reproduce the dense operators (max deviation, 8 operators)",
    max(conv),
    max(conv) < 1e-12,
    fmt="{:.1e}",
)


def commutes(U, ens):
    return all(np.allclose(U @ A @ U.T, (-1.0 if kind == "herm" else 1.0) * A, atol=1e-7) for A, kind in ens)


for nsites, bonds, lab in [(2, [(0, 1)], "2-site"), (4, [(0, 1), (1, 2), (2, 3), (3, 0)], "2x2")]:
    E = hs_ensembles(nsites, bonds)
    Tz = np.diag([1.0, -1.0, 0, 0])
    zeeman = [(majorana_matrix_from_bdg(*bdg_flavour_generator(nsites, s, Tz)), "herm") for s in range(nsites)]
    cases = {
        "eps V>0": E["eps_pos"],
        "eps V<0": E["eps_neg"],
        "Sym2 A (pairing)": E["pairing"],
        "Sym2 B (density+exchange)": E["dens_exch"],
        "Sym2 C (density+spin)": E["spin"],
        "eps + Sym2 A": E["eps_pos"] + E["pairing"],
        "eps + Zeeman control": E["eps_pos"] + zeeman,
    }
    for name, ens in cases.items():
        r = mtr_classify(E["band"] + ens, rng)
        verified = True
        if r["cls"] is not None:  # verify the symmetries explicitly
            U1, Q = r["U1"], r["Q"]
            U2 = U1 @ Q
            verified = np.allclose(U1.T, -U1) and np.allclose(U1 @ U1.T, np.eye(len(U1)), atol=1e-8)
            verified &= commutes(U1, E["band"] + ens) and commutes(U2, E["band"] + ens)
            verified &= np.allclose(U1 @ U2, -U2 @ U1, atol=1e-7)
            verified &= (r["cls"] == "Majorana") == np.allclose(U2.T, U2, atol=1e-8)
        val = (r["dimL_sym"], r["dimL_anti"], r["Tplus"] is not None, r["Tminus"] is not None, r["cls"])
        if name.startswith("eps V"):
            ok = r["cls"] == "Majorana" and verified
        elif name == "eps + Zeeman control":
            ok = r["cls"] is None
        else:
            ok = r["cls"] is None and r["Tminus"] is None
            if name in ("Sym2 B (density+exchange)", "Sym2 C (density+spin)"):
                ok &= r["dimL_sym"] + r["dimL_anti"] == 0
        c.item(f"A1 {lab} {name}: (dim L_sym, dim L_anti, T+, T-, class)", val, ok)

# exact dense weights on the 2-site cluster (256 states, 6 Trotter slices)
Hband, Dops, Phis = ops["Hband"], ops["Dops"], ops["Phis"]
dtau, ntau, N = 0.25, 6, 120
eB = expm(-dtau * Hband)
sec = parity_sectors(dim)
Exp = lambda O: ExpOp(O, sec)  # exp(c O), blockwise in the two fermion-parity sectors


def report(label, ws):
    im = float(np.max(np.abs(ws.imag) / np.abs(ws)))
    neg = float(np.mean(ws.real < -1e-9 * np.abs(ws)))
    c.record(
        f"{label}: n, max|Im|/|w|, fraction Re < 0, min Re, max Re",
        (len(ws), f"{im:.1e}", round(neg, 2), f"{ws.real.min():.3g}", f"{ws.real.max():.3g}"),
    )
    return im, neg


a = dtau * 1.0 / 4
EX = [Exp(D + D.conj().T) for D in Dops]
EY = [Exp(1j * (D - D.conj().T)) for D in Dops]


def eps_pos(a=a):  # e^{-a X^2} e^{+a Y^2} = E[e^{i s X}] E[e^{s Y}]
    return [(EX[s], 1j * rng.normal() * np.sqrt(2 * a)) for s in range(2)] + [
        (EY[s], rng.normal() * np.sqrt(2 * a)) for s in range(2)
    ]


def eps_neg():
    return [(EX[s], rng.normal() * np.sqrt(2 * a)) for s in range(2)] + [
        (EY[s], 1j * rng.normal() * np.sqrt(2 * a)) for s in range(2)
    ]


w_pos = report("eps V>0 (He et al. decoupling)", trace_weights(eps_pos, N, ntau, eB, sec))
w_neg = report("eps V<0", trace_weights(eps_neg, N, ntau, eB, sec))
c.item(
    "A2 eps term (V = +-1): weights real and positive (max|Im|/|w|, fraction negative)",
    (w_pos, w_neg),
    w_pos[0] < 1e-9 and w_pos[1] == 0 and w_neg[0] < 1e-9 and w_neg[1] == 0,
)
aV = dtau * 3.0 / 4
w3 = report("eps V=3 alone", trace_weights(lambda: eps_pos(aV), N, ntau, eB, sec))
nz = sum(xi(s, 0).conj().T @ xi(s, 0) - xi(s, 1).conj().T @ xi(s, 1) for s in range(2))
wz = report(
    "eps V=3 + Zeeman(1,2) h=2 control",
    trace_weights(lambda: eps_pos(aV), N, ntau, expm(-dtau * (Hband + 2.0 * nz)), sec),
)
c.item(
    "A3 sabotage: eps V = 3 alone positive; + flavour Zeeman h(n1 - n2), h = 2: fraction negative",
    (w3[1], wz[1]),
    w3[0] < 1e-8 and w3[1] == 0 and wz[1] > 0.1,
)
g = 1.0
keys = list(Phis)
EPr = {k: Exp(Phis[k] + Phis[k].conj().T) for k in keys}
EPi = {k: Exp(1j * (Phis[k].conj().T - Phis[k])) for k in keys}
EPc = {k: Exp(Phis[k] @ Phis[k].conj().T - Phis[k].conj().T @ Phis[k]) for k in keys}


def symA():  # e^{l Phi^+ Phi} = E_s[e^{s1 X}] E_s[e^{s2 Y}] e^{-l [Phi, Phi^+]/2} + O(l^2)
    out = []
    for k in keys:
        lam = dtau * g * (1.0 if k[2] == k[3] else 2.0)
        s1, s2 = rng.normal(size=2) * np.sqrt(lam / 2)
        out += [(EPr[k], s1), (EPi[k], s2), (EPc[k], -lam / 2)]
    return out


wA = report("Sym2 scheme A (complex pairing HS), g > 0", trace_weights(symA, N, ntau, eB, sec))
wAE = report("Sym2 scheme A + eps", trace_weights(lambda: symA() + eps_pos(), N, ntau, eB, sec))
n_ = lambda s: sum(xi(s, b).conj().T @ xi(s, b) for b in range(4))
nx, nxp = n_(0), n_(1)
X = K + K.conj().T
Y = 1j * (K - K.conj().T)
comm = K @ K.conj().T - K.conj().T @ K
Enp, Enm, EXK, EYK, Efix = Exp(nx + nxp), Exp(nx - nxp), Exp(X), Exp(Y), Exp(comm / 2 + nxp)
assert np.abs(comm - (nx - nxp)).max() < 1e-12
lamB = 2 * g * dtau


def symB():  # -g|Phi|^2 = -2g[n n' - K^+ K + n']
    return [
        (Enp, rng.normal() * np.sqrt(lamB / 2)),
        (Enm, 1j * rng.normal() * np.sqrt(lamB / 2)),
        (EXK, 1j * rng.normal() * np.sqrt(lamB / 2)),
        (EYK, 1j * rng.normal() * np.sqrt(lamB / 2)),
        (Efix, lamB),
    ]


wB = report("Sym2 scheme B (density + exchange HS)", trace_weights(symB, N, ntau, eB, sec))
gens = su4_generators() + [np.eye(4)]
Sop = lambda s, T: sum(T[a, b] * xi(s, a).conj().T @ xi(s, b) for a in range(4) for b in range(4))
E4 = np.eye(4)
TrSS = sum(Sop(0, np.outer(E4[a], E4[b])) @ Sop(1, np.outer(E4[b], E4[a])) for a in range(4) for b in range(4))
cA = [1.0 / np.trace(T @ T).real for T in gens]
e_ss = np.abs(sum(cc * Sop(0, T) @ Sop(1, T) for cc, T in zip(cA, gens, strict=True)) - TrSS).max()
H10 = h10_operator(ops, g=1.0)
e_h10 = np.abs(H10 - (-2.0) * (nx @ nxp + TrSS)).max()
c.item(
    "A4 identities Tr S S' = sum_A c_A S^A S'^A and sum_ab Phi^+ Phi = 2 (n n' + Tr S S')",
    (e_ss, e_h10),
    e_ss < 1e-12 and e_h10 < 1e-12,
)
ESp = [Exp(Sop(0, T) + Sop(1, T)) for T in gens]
ESm = [Exp(Sop(0, T) - Sop(1, T)) for T in gens]


def symC():  # e^{+l Tr SS'} = prod_A e^{l c_A/4 (S^A + S'^A)^2} e^{-l c_A/4 (S^A - S'^A)^2}, plus the density part
    out = []
    for cc, Ep, Em in zip(cA, ESp, ESm, strict=True):
        lam = lamB * cc
        out += [(Ep, rng.normal() * np.sqrt(lam / 2)), (Em, 1j * rng.normal() * np.sqrt(lam / 2))]
    return out + [(Enp, rng.normal() * np.sqrt(lamB / 2)), (Enm, 1j * rng.normal() * np.sqrt(lamB / 2))]


wC = report("Sym2 scheme C (density + SU(4) spin-channel HS)", trace_weights(symC, N, ntau, eB, sec))
c.item(
    "A5 fraction of negative weights: scheme A, A + eps, B, C (g = 1); max|Im|/|w| of scheme A",
    (wA[1], wAE[1], wB[1], wC[1], round(wA[0], 2)),
    min(wA[1], wAE[1], wB[1], wC[1]) > 0.1,
)
nodes, wts = np.polynomial.hermite_e.hermegauss(40)
wts = wts / np.sqrt(2 * np.pi)
lam = 0.1
lhs = expm(lam * (nx + nxp) @ (nx + nxp))
rhs = sum(w * Enp(np.sqrt(2 * lam) * sn) for sn, w in zip(nodes, wts, strict=True))
e_single = np.abs(lhs - rhs).max() / np.abs(lhs).max()
Pk = keys[1]
P_ = Phis[Pk]


def pairing_err(lam):
    ex = expm(lam * P_.conj().T @ P_)
    av = sum(
        w1 * w2 * (EPr[Pk](np.sqrt(lam / 2) * s1) @ EPi[Pk](np.sqrt(lam / 2) * s2))
        for s1, w1 in zip(nodes, wts, strict=True)
        for s2, w2 in zip(nodes, wts, strict=True)
    ) @ EPc[Pk](-lam / 2)
    return np.abs(av - ex).max() / np.abs(ex).max()


e1, e2 = pairing_err(0.02), pairing_err(0.04)
c.item(
    "A6 HS identities: single-operator e^{l O^2} = E[e^{s O}] (rel. error); pairing HS with the commutator factor "
    "at l = 0.02, 0.04",
    (f"{e_single:.1e}", f"{e1:.1e}", f"{e2:.1e}"),
    e_single < 1e-10 and e2 < 1e-5,
)

# =============================================================================================================
# B. Fermion-bag weights of the plaquette term (Euclidean)
# =============================================================================================================
M1, _, _ = staggered_matrix(4, 1)
n = 4
fa1 = FieldAlgebra(n)
S0 = Grass()
for i in range(n):
    for j in range(n):
        if M1[i, j] != 0:
            S0 = S0 + Grass.gen(i) * Grass.gen(j) * (0.5 * M1[i, j])
w0 = (S0 * (-1.0)).exp_nilpotent(n)
Z0 = berezin_integral(w0, n)
G1 = np.linalg.inv(M1)
dev = abs(Z0 - pfaffian(-M1).real)
for I in [(0, 1), (1, 0), (0, 3), (2, 1), (0, 1, 2, 3), (1, 0, 2, 3), (3, 2, 1, 0)]:
    mono = Grass.one()
    for i in I:
        mono = mono * Grass.gen(i)
    dev = max(dev, abs(berezin_integral(w0 * mono, n) / Z0 - wick(G1, list(I)).real))
c.item(
    "B0 Wick / Pfaffian conventions against brute-force Berezin integration (1d, L = 4)", dev, dev < 1e-12, fmt="{:.1e}"
)

M, S, idx = staggered_matrix(2, 3)
fa = FieldAlgebra(M.shape[0])
W = WickCache(M)
minw = np.inf
for r_ in range(1, 9):
    for sub in itertools.combinations(range(8), r_):
        poly = Grass.one()
        for s in sub:
            poly = poly * fa.eps_term(s)
        minw = min(minw, W.expect(poly))
c.item(
    "B1 control: eps-term monomers on 2^3, all 255 subsets: min weight Pf(G[S])^4", minw, minw >= -1e-12, fmt="{:.3g}"
)
quads = plaquette_quads(2, 3, S, idx)
res = bag_weights(M, quads, fa.sym2_term, ksum=8, normalise=False)
ws = np.array([w for _, w in res])
single = [w for cfg, w in res if sum(cfg) == 1]
c.item(
    "B2 2^3 Sym^2 plaquette term, all orders: nonzero configurations, min, max",
    (len(res), round(ws.min(), 2), f"{ws.max():.3g}"),
    len(quads) == 6 and len(res) == 552 and ws.min() > 0,
)
c.item(
    "  single insertions uniform (shift symmetry)",
    single[0],
    np.allclose(single, single[0]) and single[0] > 0,
    fmt="{:.4g}",
)
res6 = bag_weights(M, quads, fa.lam2_term, ksum=3, normalise=True)
neg6 = [(cfg, w) for cfg, w in res6 if w < 0]
cfg, w = min(neg6, key=lambda t: sum(t[0]))
c.item(
    "B3 sabotage: two-site Lambda^2 term T_6, sum k <= 3: negative / total, first negative (order, config, w)",
    (f"{len(neg6)}/{len(res6)}", sum(cfg), cfg, round(w, 2)),
    bool(neg6) and sum(cfg) == 2,
)
c.item(
    "B4 g < 0 flips every odd order: sign(g^k w) = (-1)^k",
    True,
    all(((-1) ** sum(cfg)) * w < 0 for cfg, w in res if sum(cfg) % 2 == 1),
)
M, S, idx = staggered_matrix(4, 2)
fa = FieldAlgebra(M.shape[0])
res = bag_weights(M, plaquette_quads(4, 2, S, idx), fa.sym2_term, ksum=3, normalise=False)
ws = np.array([w for _, w in res])
c.item(
    "B5 4^2 plaquette term, sum k <= 3: configurations, min weight",
    (len(res), round(ws.min(), 2)),
    len(res) == 164 and ws.min() > 0,
)
resl = bag_weights(M, line_quads(4, 2, S, idx), fa.sym2_term, ksum=3, normalise=False)
naive_neg = any(w < 0 for cfg, w in resl if sum(cfg) == 1)
resl = bag_weights(M, line_quads(4, 2, S, idx), fa.sym2_term, ksum=3, normalise=True)
c.item(
    "  collinear geometry: naive orientation has negative single insertions; oriented: configurations, min",
    (naive_neg, len(resl), round(min(w for _, w in resl), 2)),
    naive_neg and len(resl) == 164 and min(w for _, w in resl) > 0,
)
M, S, idx = staggered_matrix(4, 3)
fa = FieldAlgebra(M.shape[0])
quads = [q for q in plaquette_quads(4, 3, S, idx) if all(S[i][0] in (0, 1) and S[i][1] in (0, 1) for i in q)]
res = bag_weights(M, quads, fa.sym2_term, ksum=3, normalise=False)
ws = np.array([w for _, w in res])
c.item(
    "B6 4^3 (2x2x4 block, 10 quads), sum k <= 3: configurations, min weight",
    (len(res), round(ws.min(), 2)),
    len(quads) == 10 and len(res) == 285 and ws.min() > 0,
)
M, S, idx = staggered_matrix(4, 2)
fa = FieldAlgebra(M.shape[0])
W = WickCache(M)
ls = oriented_links(M, S, idx, 4, 2)
lkeys = sorted(ls)
Kl = {link: fa.K(*link) * ls[link] for link in lkeys}
flux = all(
    ls[(x, y)] * ls[(xp, yp)] == -1 and ls[(x, yp)] * ls[(xp, y)] == +1
    for x, xp, y, yp in plaquette_quads(4, 2, S, idx)[:6]
)
c.item("B7 pi flux of the link orientation: s_mu s_mu' = -1, s_nu s_nu' = +1", flux, flux)
G = np.linalg.inv(M)
signs = [ls[link] for link in lkeys]
nzc, minv = 0, np.inf
for k in (1, 2, 3):
    for sel in itertools.combinations(range(len(lkeys)), k):
        lk = [lkeys[i] for i in sel]
        if len({s for link in lk for s in link}) < 2 * k:
            continue
        T4 = kmonomial_table(link_pfaffians(G, lk, [signs[i] for i in sel]), k, 4)[(slice(1, None),) * k]
        vals = T4[np.abs(T4) > 1e-12]
        nzc += vals.size
        minv = min(minv, vals.min())
c.item("  4^2: oriented K-monomials on <= 3 links: nonzero count, min (all positive)", (nzc, round(minv, 4)), minv > 0)
neg4 = {(0, 1): 2, (2, 6): 2, (5, 9): 3, (8, 12): 3}
neg5 = {(2, 1): 2, (8, 11): 2, (10, 6): 2, (13, 9): 2, (15, 14): 2}
for label, mdict, expect in (("4-link", neg4, -21 / 256), ("5-link", neg5, None)):
    lk = list(mdict)
    gf = kmonomial_table(link_pfaffians(G, lk, [ls[link] for link in lk]), len(lk), 4)[tuple(mdict[x] for x in lk)]
    wk = kmonomial_weight(W, Kl, mdict)
    ok = abs(gf - wk) < 1e-9 and wk < -1e-3 and (expect is None or abs(wk - expect) < 1e-9)
    c.item(
        f"  4^2: negative {label} monomial {[(S[u], S[v], m) for (u, v), m in mdict.items()]} (generating function "
        "= Wick)",
        round(wk, 5),
        ok,
    )

# =============================================================================================================
# C. Sign-free reformulation
# =============================================================================================================
rng = np.random.default_rng(23)
fa = FieldAlgebra(4)
par = [1, 1, -1, -1]
n = fa.ngen
pairs = list(itertools.combinations(range(n), 2))
basis = [Grass.gen(i) * Grass.gen(j) for i, j in pairs]
pidx = {(1 << i) | (1 << j): k for k, (i, j) in enumerate(pairs)}


def rep_matrix(V):
    R = np.zeros((len(pairs), len(pairs)), complex)
    for k, b in enumerate(basis):
        for m, cf in substitute(b, V, n).t.items():
            R[pidx[m], k] = cf
    return R


rows = [rep_matrix(fa.map_su4(random_su4(rng), par)) - np.eye(len(pairs)) for _ in range(3)]
rows += [rep_matrix(fa.map_u1eps(0.7, par)) - np.eye(len(pairs))]
ninv = int(np.sum(np.linalg.svd(np.vstack(rows), compute_uv=False) < 1e-9))
c.item("C1 SU(4) x U(1)_eps-invariant bilinears on a quad (the four even-odd link K's)", ninv, ninv == 4)
x, xp, y, yp = 0, 1, 2, 3
E = fa.sym2_term((x, xp, y, yp)).exp_nilpotent(8)
moment = True
for link in [(x, y), (x, yp), (xp, y), (xp, yp)]:
    Kl_ = fa.K(*link)
    mono = fa.chi(link[0], 0) * fa.chi(link[1], 0) * fa.chi(link[0], 1) * fa.chi(link[1], 1)
    ((mask, cf),) = mono.t.items()
    moment &= abs((Kl_ * Kl_).coeff(mask) / cf - 2.0) < 1e-12 and abs(E.coeff(mask)) < 1e-12
c.item(
    "C2 moment obstruction: chi1_u chi1_v chi2_u chi2_v has coefficient 2 in K_l^2 and 0 in e^{T_Q} (all 4 links)",
    moment,
    moment,
)
M, S, idx = staggered_matrix(4, 2)
quads = plaquette_quads(4, 2, S, idx)
found = {}
for kind, amp in (("real", 1.0), ("imag", 1.0), ("complex", 0.5)):
    vals = []
    for _ in range(40):
        phi = []
        for _q in quads:
            aa = rng.normal(size=(4, 4))
            bb = rng.normal(size=(4, 4))
            ph = (aa + aa.T) / 2 * amp
            if kind == "imag":
                ph = 1j * ph
            if kind == "complex":
                ph = ph + 1j * (bb + bb.T) / 2 * amp
            phi.append(ph)
        vals.append(pfaffian(sym2_yukawa_matrix(M, quads, phi)))
    found[kind] = np.array(vals)
fr = {
    k: (round(float(np.max(np.abs(v.imag) / np.abs(v))), 2), round(float(np.mean(v.real < 0)), 2))
    for k, v in found.items()
}
ok = fr["real"][0] < 1e-8 and fr["real"][1] > 0.1 and fr["imag"][0] < 1e-8 and fr["imag"][1] > 0.1
ok &= fr["complex"][0] > 0.3
c.item(
    "C3 4^2 10-plet Yukawa Pfaffian, 40 draws each: (max|Im|/|Pf|, fraction Re < 0) for real / imag / complex", fr, ok
)
worst = 0.0
for L, d in ((4, 2), (2, 3)):
    M, S, idx = staggered_matrix(L, d)
    fa = FieldAlgebra(M.shape[0])
    ls = oriented_links(M, S, idx, L, d)
    for q in plaquette_quads(L, d, S, idx):
        x, xp, y, yp = q
        s = {link: ls[link] for link in [(x, y), (xp, yp), (x, yp), (xp, y)]}
        Kt = {link: fa.K(*link) * s[link] for link in s}
        B1 = Kt[(x, yp)] + Kt[(xp, y)]
        B2 = Kt[(x, y)] + Kt[(xp, yp)]
        lhs = fa.sym2_term(q) + sum((fa.K(*link) * fa.K(*link) for link in s), Grass())
        worst = max(worst, (lhs - (B1 * B1 + B2 * B2)).maxabs())
        worst = max(worst, (fa.sym2_term(q) - (Kt[(x, yp)] * Kt[(xp, y)] + Kt[(x, y)] * Kt[(xp, yp)]) * 2.0).maxabs())
c.item(
    "C4 T_Q + sum_{l in Q} K_l^2 = (Kt_nu + Kt_nu')^2 + (Kt_mu + Kt_mu')^2 on every quad of 4^2 and 2^3",
    worst,
    worst < 1e-12,
    fmt="{:.1e}",
)
M, S, idx = staggered_matrix(2, 3)
fa = FieldAlgebra(M.shape[0])
ls = oriented_links(M, S, idx, 2, 3)
x, xp, y, yp = plaquette_quads(2, 3, S, idx)[0]
B = fa.K(x, yp) * ls[(x, yp)] + fa.K(xp, y) * ls[(xp, y)]
g = 0.37
lhs = (B * B * g).exp_nilpotent(8)
nodes, wts = np.polynomial.hermite_e.hermegauss(12)
rhs = Grass()
for sn, wn in zip(nodes, wts, strict=True):
    rhs = rhs + (B * (sn * np.sqrt(2 * g))).exp_nilpotent(16) * (wn / np.sqrt(2 * np.pi))
c.item(
    "C5 e^{g B^2} = E_s[e^{s B}] exactly (12-point Gauss-Hermite on the nilpotent series)",
    (lhs - rhs).maxabs(),
    (lhs - rhs).maxabs() < 1e-10,
    fmt="{:.1e}",
)
worst_pf, minpf = 0.0, np.inf
for L, d in ((4, 2), (4, 3)):
    M, S, idx = staggered_matrix(L, d)
    ls = oriented_links(M, S, idx, L, d)
    quads = plaquette_quads(L, d, S, idx)
    Nn = M.shape[0]
    for amp in (0.5, 2.0, 8.0):
        for _ in range(6):
            A = np.zeros((Nn, Nn))
            for x, xp, y, yp in quads:
                s1, s2 = rng.normal(size=2) * amp
                for (u, v), cf in (((x, yp), s1), ((xp, y), s1), ((x, y), s2), ((xp, yp), s2)):
                    A[u, v] += cf * ls[(u, v)]
                    A[v, u] -= cf * ls[(u, v)]
            D1 = M - A
            pf1 = pfaffian(D1)
            pf4 = pfaffian(np.kron(np.eye(4), D1))
            det1 = np.linalg.det(D1)
            assert abs(pf1.imag) < 1e-9 * abs(pf1) and abs(pf4.imag) < 1e-9 * abs(pf4)
            worst_pf = max(worst_pf, abs(pf4.real - pf1.real**4) / abs(pf4), abs(det1**2 - pf4.real) / abs(pf4))
            minpf = min(minpf, pf4.real)
c.item(
    "C6 4^2 and 4^3, random real link fields (amp 0.5, 2, 8): Pf(D(s)) = Pf(M - A)^4 = det(M - A)^2 > 0 "
    "(max rel. deviation)",
    worst_pf,
    worst_pf < 1e-8 and minpf > 0,
    fmt="{:.1e}",
)
M2, S2, i2 = staggered_matrix(2, 3)
fa = FieldAlgebra(M2.shape[0])
ls2 = oriented_links(M2, S2, i2, 2, 3)
q2 = plaquette_quads(2, 3, S2, i2)[0]
x, xp, y, yp = q2
Kt = {link: fa.K(*link) * ls2[link] for link in [(x, y), (xp, yp), (x, yp), (xp, y)]}
worst = 0.0
for g6 in (0.5, -0.5, 1.0, -1.0):
    lhs = fa.sym2_term(q2) + fa.lam2_term(q2) * g6
    a1, a2 = 1.0 - g6, 1.0 + g6
    sos = (Kt[(x, yp)] + Kt[(xp, y)]) ** 2 * a1 + (Kt[(x, y)] + Kt[(xp, yp)]) ** 2 * a2
    extra = (fa.K(x, yp) ** 2 + fa.K(xp, y) ** 2) * a1 + (fa.K(x, y) ** 2 + fa.K(xp, yp) ** 2) * a2
    worst = max(worst, (lhs + extra - sos).maxabs())
c.item(
    "C7 g10 T_Q + g6 T_6 + (g10 - g6)(K_nu^2 + K_nu'^2) + (g10 + g6)(K_mu^2 + K_mu'^2) = sum of squares "
    "(g10 = 1, g6 = +-0.5, +-1)",
    worst,
    worst < 1e-12,
    fmt="{:.1e}",
)
c.done()
