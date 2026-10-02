"""K2.5: C0 (the same-parity source pattern) is a nodal pairing -- no corner mass in the free theory, and none induced
by the interaction at odd order <= 5 in h -- while C_chi in the eps_{mu nu rho sigma} convention is the taste-chiral
Majorana mass (one taste doublet gapped at h, the other exactly massless); the tau_2 K sign check is pattern-blind."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.algebra.orbits import group_representations, orbit_span, perm_matrix, power_traces, sym_characters
from masspairing.claimcheck import Check
from masspairing.data import derived
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.operator import DoubletOperator
from masspairing.patterns import (
    DUAL,
    PH,
    all_planes_mass,
    chiral_mass,
    eps_sign,
    kinetic_gammas,
    plane_op,
    scalar_mass_strings,
    singular_values,
    source_pattern,
)
from masspairing.symmetry import sign_field, site_map, translation

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K2.5")
PERIODIC, AX, AAAA = (1, 1, 1, 1), (-1, 1, 1, 1), (-1, -1, -1, -1)


def zero_modes(K):
    ev, U = np.linalg.eigh(K.T @ K)
    return U[:, ev < 1e-9]


# =============================================================================================================
# 1. C0 is nodal
# =============================================================================================================
for L in (4, 6, 8):
    lat = Lattice((L,) * 4, bc=PERIODIC)
    K, C0 = kinetic_matrix(lat).toarray(), source_pattern(lat).toarray()
    P0 = zero_modes(K)
    nz = [int((singular_values(K - h * C0, 24) < 1e-6).sum()) for h in (0.1, 0.5, 1.0, 2.0)]
    c.item(
        f"L={L} periodic: zero modes of K; |C0 P0|; zero modes of K - h C0 at h = 0.1, 0.5, 1, 2",
        (P0.shape[1], f"{abs(C0 @ P0).max():.1e}", nz),
        P0.shape[1] == 16 and abs(C0 @ P0).max() < 1e-12 and nz == [16] * 4,
    )
hs = (0.1, 0.3, 0.5, 1.0, 2.0)
for L in (4, 6):
    res = {}
    for bc, lab in ((None, "antiperiodic t"), (AX, "antiperiodic x"), (AAAA, "all antiperiodic")):
        lat = Lattice((L,) * 4, bc=bc)
        K, C0 = kinetic_matrix(lat).toarray(), source_pattern(lat).toarray()
        s0 = singular_values(K, 1)[0]
        s = np.array([singular_values(K - h * C0, 1)[0] for h in hs])
        res[lab] = (s0, s)
        c.record(f"L={L} {lab}: s_min free, with C0 at h = {hs}", (round(float(s0), 4), np.round(s, 4)))
    (s0t, st), (s0x, sx), (s0a, sa) = res["antiperiodic t"], res["antiperiodic x"], res["all antiperiodic"]
    c.item(
        f"L={L}: s_min(K - h C0) h-independent with antiperiodic t (max shift), h-dependent with antiperiodic x "
        "(max shift), lowered with all-antiperiodic bc (min / free)",
        (f"{abs(st - s0t).max():.1e}", round(float(abs(sx - s0x).max()), 3), round(float(sa.min() / s0a), 3)),
        abs(st - s0t).max() < 1e-10 and abs(sx - s0x).max() > 0.05 and sa.min() < 0.6 * s0a,
    )

# interaction-induced corner masses: character sums over the lattice symmetry group (4^4, production bc)
lat = Lattice((4,) * 4)
V = lat.V
K = kinetic_matrix(lat)
gens = [translation(lat, mu) for mu in range(4)]
gens += [site_map(lat, p, (1, 1, 1, 1)) for p in ((1, 0, 2, 3), (0, 2, 1, 3))]
for mu in range(4):
    r = [1, 1, 1, 1]
    r[mu] = -1
    gens.append(site_map(lat, (0, 1, 2, 3), tuple(r)))
gens = [(g, sign_field(K, g)) for g in gens]
assert all(s is not None for _, s in gens)
Ts = [perm_matrix(g, s) for g, s in gens]
QW = orbit_span([source_pattern(lat)], Ts)
QM = orbit_span([plane_op(lat, *D) for D in PH], Ts)
QX = orbit_span([chiral_mass(lat)], Ts)
elems = group_representations(gens, [QW, QM, QX])
n = len(elems)
pW = power_traces([e[0] for e in elems], 5)
cM = np.array([np.trace(e[1]) for e in elems])
cX = np.array([np.trace(e[2]) for e in elems])
m1, m3, m5 = (np.dot(ch, cM) / n for ch in sym_characters(pW))
c.item(
    "|G| (shifts x spatial permutations x reflections x time reflection, mod the global sign); dim of the orbit "
    "span of C0; dim of the plane-mass span",
    (n, QW.shape[1], QM.shape[1]),
    n == 24576 and QW.shape[1] == 24 and QM.shape[1] == 6,
)
c.item(
    "multiplicity of the plane-mass irreps in Sym^1, Sym^3, Sym^5 of the C0 orbit span",
    (f"{m1:.1e}", f"{m3:.1e}", f"{m5:.1e}"),
    abs(m1) < 1e-6 and abs(m3) < 1e-6 and abs(m5) < 1e-6,
)
cXM, cXW = np.dot(cX, cM) / n, np.dot(cX, pW[0]) / n
c.item(
    "<chi_orbit(C_chi), chi_planes> (both taste triplets), <chi_orbit(C_chi), chi_orbit(C0)>",
    (round(float(cXM), 9), round(float(cXW), 9)),
    abs(cXM - 2) < 1e-9 and abs(cXW) < 1e-9,
)

# =============================================================================================================
# 2. corner space: the scalar-mass strings, the plane operators, the eps convention
# =============================================================================================================
G = kinetic_gammas()
c.item(
    "the four corner-space kinetic gammas anticommute",
    True,
    all(np.allclose(G[i] @ G[j] + G[j] @ G[i], 0) for i in range(4) for j in range(i + 1, 4)),
)
strings = scalar_mass_strings()
two_link = sorted((D, S) for S, D, anti in strings if len(D) == 2 and anti)
one_link = sorted((D, S) for S, D, anti in strings if len(D) == 1 and anti)
c.item("number of scalar-mass strings X_S Z_D", len(strings), len(strings) == 16)
c.item(
    "antisymmetric two-link scalar masses: one per plane with the phases PH",
    two_link,
    two_link == sorted((D, PH[D]) for D in PH),
)
c.item("antisymmetric one-link (Dirac-type) scalar masses", one_link, len(one_link) == 4)
c.item(
    "eps_{mu nu rho sigma} signs of (plane, dual plane)",
    {k: eps_sign(k) for k in PH},
    {k: eps_sign(k) for k in PH} == {(0, 1): 1, (0, 2): -1, (0, 3): 1, (1, 2): 1, (1, 3): -1, (2, 3): 1},
)
lat = Lattice((4,) * 4, bc=PERIODIC)
K = kinetic_matrix(lat).toarray()
P0 = zero_modes(K)
blk = {D: P0.T @ plane_op(lat, *D).toarray() @ P0 for D in PH}
anti = all(abs(plane_op(lat, *D).toarray() + plane_op(lat, *D).toarray().T).max() == 0 for D in PH)
sv4 = all(np.allclose(np.linalg.svd(blk[D], compute_uv=False), 4.0) for D in PH)
c.item("plane operators antisymmetric, corner blocks 16-fold degenerate at 4", (anti, sv4), anti and sv4)
alg = True
for D in PH:
    for F in PH:
        if F == DUAL[D]:
            alg &= np.allclose(blk[D] @ blk[F] - blk[F] @ blk[D], 0)
        elif F != D:
            alg &= np.allclose(blk[D] @ blk[F] + blk[F] @ blk[D], 0)
c.item("a plane commutes with its dual plane and anticommutes with the other four", alg, alg)
ranks = {}
for D, sgn in (((0, 3), 1), ((0, 1), 1), ((0, 2), 1), ((0, 3), -1)):
    sv = np.linalg.svd(blk[D] + sgn * eps_sign(D) * blk[DUAL[D]], compute_uv=False)
    ranks[f"{D}{sgn:+d}"] = (int((sv > 1e-9).sum()), np.unique(np.round(sv[sv > 1e-9], 6)).tolist())
c.item(
    "(anti-)self-dual sums C(plane) +- eps C(dual): corner rank, nonzero value",
    ranks,
    all(r == (8, [8.0]) for r in ranks.values()),
)


def light(C):
    T = P0.T @ C @ P0
    w, v = np.linalg.eigh(T.T @ T)
    return v[:, w < 1e-9]


cos = lambda A, B: np.linalg.svd(A.T @ B, compute_uv=False)
ref = light(chiral_mass(lat, (0, 3), 1).toarray())
c01 = cos(ref, light(chiral_mass(lat, (0, 1), 1).toarray()))
c02 = cos(ref, light(chiral_mass(lat, (0, 2), 1).toarray()))
c02m = cos(ref, light(chiral_mass(lat, (0, 2), -1).toarray()))
c.item(
    "light corner subspaces: principal cosines of (0,1)+ and (0,2)+ against (0,3)+ (min), of (0,2)- (max)",
    (round(float(c01.min()), 6), round(float(c02.min()), 6), f"{c02m.max():.1e}"),
    ref.shape[1] == 8 and np.allclose(c01, 1) and np.allclose(c02, 1) and np.allclose(c02m, 0),
)
old02 = (plane_op(lat, 0, 2) + plane_op(lat, 1, 3)) / 8.0
c.item(
    "the combination without eps, [C(0,2) + C(1,3)]/8, is (0,2)- (gaps the other doublet)",
    abs(old02 - chiral_mass(lat, (0, 2), -1)).max(),
    abs(old02 - chiral_mass(lat, (0, 2), -1)).max() < 1e-15,
    fmt="{:.1e}",
)
Qold = {D: blk[D] @ blk[DUAL[D]] / 16.0 for D in ((0, 1), (0, 2), (0, 3))}
Qeps = {D: eps_sign(D) * blk[D] @ blk[DUAL[D]] / 16.0 for D in ((0, 1), (0, 2), (0, 3))}
e_old = (abs(Qold[(0, 1)] - Qold[(0, 3)]).max(), abs(Qold[(0, 2)] + Qold[(0, 3)]).max())
e_eps = max(abs(Qeps[(0, 1)] - Qeps[(0, 3)]).max(), abs(Qeps[(0, 2)] - Qeps[(0, 3)]).max())
c.item(
    "corner Q_D = J_D J_dual / 16: Q(0,1) = Q(0,3) = -Q(0,2); with eps, Q_D identical for the three components",
    (f"{max(e_old):.1e}", f"{e_eps:.1e}"),
    max(e_old) < 1e-12 and e_eps < 1e-12,
)
T03, T02m = blk[(0, 3)] + blk[(1, 2)], blk[(0, 2)] - blk[(1, 3)]
T01, T02p = blk[(0, 1)] + blk[(2, 3)], blk[(0, 2)] + blk[(1, 3)]
trip = max(abs(A @ B + B @ A).max() for A, B in ((T03, T02m), (T03, T01), (T01, T02m)))
c.item(
    "heavy-doublet triplet {P01+P23, P02-P13, P03+P12} mutually anticommutes; [P03+P12, P02+P13] = 0",
    (f"{trip:.1e}", f"{abs(T03 @ T02p - T02p @ T03).max():.1e}"),
    trip < 1e-12 and abs(T03 @ T02p - T02p @ T03).max() < 1e-12,
)

# =============================================================================================================
# 3. the chiral mass C_chi
# =============================================================================================================
for L in (4, 6, 8):
    lat = Lattice((L,) * 4, bc=PERIODIC)
    K, C = kinetic_matrix(lat).toarray(), chiral_mass(lat).toarray()
    devs = []
    for h in (0.1, 0.5):
        s = singular_values(K - h * C, 16)
        devs.append((int((s < 1e-6).sum()), float(abs(s[8:16] - h).max())))
    c.item(
        f"L={L} periodic, h = 0.1, 0.5: (zero modes, max |s - h| of the next 8)",
        [(z, f"{e:.1e}") for z, e in devs],
        all(z == 8 and e < 1e-8 for z, e in devs),
    )
    latA = Lattice((L,) * 4)
    KA, CA = kinetic_matrix(latA).toarray(), chiral_mass(latA).toarray()
    s0 = singular_values(KA, 1)[0]
    prev, rows, ok = 0.0, [], True
    for h in (0.1, 0.5, 2.0):
        s = singular_values(KA - h * CA, 32)
        light_, heavy = s[:16], s[16:32]
        hb = np.sqrt(s0**2 + (0.8 * min(h, 1.0)) ** 2)
        ok &= abs(light_ - s0).max() < 0.05 * h * h + 1e-9 and heavy.min() >= hb and heavy.min() >= prev
        rows.append((h, round(float(abs(light_ - s0).max()), 4), round(float(heavy.min()), 4), round(float(hb), 4)))
        prev = heavy.min()
    c.item(
        f"L={L} production bc (s0 = {s0:.4f}): (h, max light deviation (< 0.05 h^2), heavy min, bound "
        "sqrt(s0^2 + (0.8 min(h,1))^2)), monotone",
        rows,
        ok,
    )
    if L == 8:
        Cn = 8.0 * all_planes_mass(lat, phase="eta_nu").toarray()
        sm = {h: singular_values(K - h * Cn, 1)[0] for h in (0.1, 0.3)}
        c.item(
            "contrast: naive eta_nu-phased plane sum on periodic 8^4 has a level crossing: s_min at h = 0.1, 0.3",
            (round(float(sm[0.1]), 4), round(float(sm[0.3]), 4)),
            sm[0.1] > 0.25 and sm[0.3] < 0.05,
        )
lat = Lattice((4,) * 4)
latx = Lattice((4,) * 4, bc=(-1, 1, 1, 1))
K, Kx = kinetic_matrix(lat).toarray(), kinetic_matrix(latx).toarray()
iso, comp = 0.0, 0.0
for h in (0.3, 1.0):
    s_t = np.sort(np.linalg.svd(K - h * chiral_mass(lat).toarray(), compute_uv=False))
    s_x = np.sort(np.linalg.svd(Kx - h * chiral_mass(latx).toarray(), compute_uv=False))
    iso = max(iso, abs(s_t - s_x).max())
    for cmp, sgn in (((0, 1), 1), ((0, 2), 1), ((0, 3), -1)):
        s = np.sort(np.linalg.svd(K - h * chiral_mass(lat, cmp, sgn).toarray(), compute_uv=False))
        comp = max(comp, abs(s - s_t).max())
c.item(
    "isotropy (antiperiodic t vs antiperiodic x); components (0,1)+, (0,2)+, (0,3)- vs (0,3)+: max spectral "
    "difference, h = 0.3, 1",
    (f"{iso:.1e}", f"{comp:.1e}"),
    iso < 1e-12 and comp < 1e-12,
)
Ce = chiral_mass(lat, parity=+1).toarray()
ld0 = np.linalg.slogdet(K)[1]
dd = [(np.linalg.slogdet(K - h * Ce)[0], np.linalg.slogdet(K - h * Ce)[1] - ld0) for h in (0.5, 2.0)]
c.item(
    "even-only C_chi leaves det(K - h C) = det K (h = 0.5, 2: sign, log det difference)",
    [(float(sg), f"{x:.1e}") for sg, x in dd],
    all(sg > 0 and abs(x) < 1e-8 for sg, x in dd),
)

# =============================================================================================================
# 4. sign: tau_2 K holds for any real flavour-blind pattern; the mass test detects a wrong one
# =============================================================================================================
lat = Lattice((4,) * 4)
C = chiral_mass(lat).toarray()
rng = np.random.default_rng(147)
configs = [("random", rng.normal(size=3 * lat.V)) for _ in range(2)]
configs.append(("stored 4^4 P_c", np.load(derived("algebra", "eps_L4_y2.41_k-0.01_final.npz"))["fields"]))
D = DoubletOperator(lat, 2.41)
for name, f in configs:
    Dc = D.dense(f)
    sg = [np.linalg.slogdet(Dc - h * np.kron(C, np.eye(2)))[0] for h in (0.3, 1.0)]
    c.item(
        f"4^4 {name}, y = 2.41: sign of det(D_c - h C_chi (x) 1) at h = 0.3, 1",
        [np.round(x, 10) for x in sg],
        all(abs(x - 1.0) < 1e-10 for x in sg),
    )


def flipped(lat, comp=(0, 3)):
    """Control: plane (0,3) without its phase zeta."""
    return ((plane_op(lat, *comp, phase=()) + plane_op(lat, *DUAL[comp])) / 8.0).tocsr()


lat6 = Lattice((6,) * 4)
D6 = DoubletOperator(lat6, 2.41).dense(np.load(derived("algebra", "eps_L6_y2.41_k-0.01_final.npz"))["fields"])
pats = {"C_chi": chiral_mass(lat6).toarray(), "C_flip": flipped(lat6).toarray(), "C0": source_pattern(lat6).toarray()}
for name, Cp in pats.items():
    sg = [np.linalg.slogdet(D6 - h * np.kron(Cp, np.eye(2)))[0] for h in (0.3, 1.0)]
    c.item(
        f"stored 6^4 P_c, y = 2.41: sign of det(D_c - h {name} (x) 1) at h = 0.3, 1",
        [np.round(x, 10) for x in sg],
        all(abs(x - 1.0) < 1e-10 for x in sg),
    )
lat4 = Lattice((4,) * 4, bc=PERIODIC)
K4 = kinetic_matrix(lat4).toarray()
sf = singular_values(K4 - 0.5 * flipped(lat4).toarray(), 16)
sc = singular_values(K4 - 0.5 * chiral_mass(lat4).toarray(), 16)
c.item(
    "mass test (periodic 4^4, h = 0.5): lowest 16 singular values, flipped pattern vs C_chi",
    (np.unique(np.round(sf, 4)).tolist(), np.unique(np.round(sc, 4)).tolist()),
    int((sc < 1e-6).sum()) == 8 and int((sf < 1e-6).sum()) == 0 and np.allclose(sf, 0.25, atol=1e-6),
)
c.done()
