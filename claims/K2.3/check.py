"""K2.3: anatomy of the completed (wedge) term -- edge identities, the wedge as the boundary of every positive real
flavour-blind HS, the symmetry patterns of the columnar dimer and of the Sym^2 condensate, free-theory weights on 4^4,
and a sign-free same-parity source."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import itertools

import numpy as np

from masspairing.action import build_model
from masspairing.algebra.grassmann import Grass, substitute
from masspairing.algebra.staggered import NF, FieldAlgebra, oriented_links, parity, random_su4, staggered_matrix
from masspairing.claimcheck import Check
from masspairing.flavour import real_flavour_block
from masspairing.lattice import Lattice

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K2.3")
L, d = 4, 2
Ma, S, idx = staggered_matrix(L, d)  # antiperiodic (Wick reference)
Mp, _, _ = staggered_matrix(L, d, bc="periodic")  # periodic (symmetry table: shifts need no boundary sign)
n = Ma.shape[0]
fa = FieldAlgebra(n)
NG = fa.ngen
par = np.array([parity(x) for x in S])
links = oriented_links(Ma, S, idx, L, d)
links_p = {link: -float(np.sign(Mp[link])) for link in links}  # local rule s_l = -sign M_uv on the periodic lattice
sh = lambda x, mu, k=1: tuple((x[i] + (k if i == mu else 0)) % L for i in range(d))
Kt = lambda u, v: fa.K(u, v) * links[(u, v)]
Ktp = lambda u, v: fa.K(u, v) * links_p[(u, v)]
c.item(
    "orientation s_l = sign G_uv equals the local rule -sign M_uv (antiperiodic 4^2)",
    True,
    all(links[link] == -float(np.sign(Ma[link])) for link in links),
)

# ---- (1) edge identities on every quad (both orientations)
quads = []
for x in S:
    if parity(x) != 1:
        continue
    for mu, nu in itertools.combinations(range(d), 2):
        for sg in (+1, -1):
            quads.append((idx[x], idx[sh(sh(x, mu), nu, sg)], idx[sh(x, mu)], idx[sh(x, nu, sg)]))
worst = 0.0
for q in quads:
    x, xp, y, yp = q
    TQ, T6 = fa.sym2_term(q), fa.lam2_term(q)
    worst = max(
        worst, ((TQ + T6) - Kt(x, y) * Kt(xp, yp) * 4.0).maxabs(), ((TQ - T6) - Kt(x, yp) * Kt(xp, y) * 4.0).maxabs()
    )
c.item(
    f"T_Q + T_6 = 4 Kt_mu Kt_mu' and T_Q - T_6 = 4 Kt_nu Kt_nu' on all {len(quads)} quads of 4^2 (max coefficient)",
    worst,
    worst < 1e-12,
    fmt="{:.1e}",
)

# ---- (2) moment bookkeeping
x, xp, y, yp = quads[0]
lk = [(x, yp), (xp, y), (x, y), (xp, yp)]  # [nu, nu', mu, mu']
a = np.array([0.37, -0.81, 0.55, 1.13])
E = sum((Kt(*lk[i]) * a[i] for i in range(4)), Grass()).exp_nilpotent(8)
mono = lambda u, v, up, vp: fa.chi(u, 0) * fa.chi(v, 0) * fa.chi(up, 1) * fa.chi(vp, 1)
ok = True
for i in range(4):
    ((m, cf),) = mono(*lk[i], *lk[i]).t.items()
    ok &= abs(E.coeff(m) / cf - a[i] ** 2) < 1e-12
    for j in range(4):
        if j != i:
            ((m2, c2),) = mono(*lk[i], *lk[j]).t.items()
            ok &= abs(E.coeff(m2) / c2 - a[i] * a[j] * links[lk[i]] * links[lk[j]]) < 1e-12
c.item("in e^{sum a_l Kt_l}: same-link two-flavour monomial a_l^2, two-link monomial a_l a_l' s_l s_l'", ok, ok)
g, dl = 0.3, np.array([0.2, 0.45, 0.31, 0.29])
Eg = (fa.sym2_term(quads[0]) * g + sum((fa.K(*lk[i]) * fa.K(*lk[i]) * dl[i] for i in range(4)), Grass())).exp_nilpotent(
    8
)
ok = True
for i in range(4):
    ((m, cf),) = mono(*lk[i], *lk[i]).t.items()
    ok &= abs(Eg.coeff(m) / cf - 2 * dl[i]) < 1e-12
for i, j, expect in (
    (0, 1, 2 * g * links[lk[0]] * links[lk[1]]),
    (2, 3, 2 * g * links[lk[2]] * links[lk[3]]),
    (0, 2, 0.0),
    (1, 3, 0.0),
    (0, 3, 0.0),
    (1, 2, 0.0),
):
    ((m2, c2),) = mono(*lk[i], *lk[j]).t.items()
    ok &= abs(Eg.coeff(m2) / c2 - expect) < 1e-12
c.item(
    "in e^{g T_Q + sum d_l K_l^2}: 2 d_l, and 2 g s s' (nu and mu pairs), 0 (nu-mu pairs) => d_nu d_nu' >= g^2, "
    "d_mu d_mu' >= g^2 (Cauchy-Schwarz)",
    ok,
    ok,
)

# ---- (3) symmetry table on periodic 4^2
S0 = fa.kinetic(Mp)
image = lambda poly, V: substitute(poly, V, NG)


def shift_map(mu):
    """chi(x) -> zeta_mu(x) chi(x + mu), zeta_mu(x) = (-1)^{sum_{nu > mu} x_nu}."""
    V = np.zeros((NG, NG))
    for x in S:
        zeta = (-1.0) ** sum(x[nu] for nu in range(mu + 1, d))
        for aa in range(NF):
            V[aa * n + idx[x], aa * n + idx[sh(x, mu)]] = zeta
    return V


rng = np.random.default_rng(3)
maps = {
    "U(1)_eps": fa.map_u1eps(0.7, par),
    "Z4": fa.map_u1eps(np.pi / 2, par),
    "SU(4)": fa.map_su4(random_su4(rng), par),
    "shift_0": shift_map(0),
    "shift_1": shift_map(1),
}
kin = max((image(S0, V) - S0).maxabs() for V in maps.values())
c.item("kinetic term invariant under U(1)_eps, Z4, SU(4), shift_0, shift_1", kin, kin < 1e-10, fmt="{:.1e}")


def D_col(mu, nu):
    r = Grass()
    for x in S:
        xn = sh(x, mu)
        u, v = (idx[x], idx[xn]) if parity(x) == 1 else (idx[xn], idx[x])
        r = r + Ktp(u, v) * ((-1.0) ** x[nu])
    return r * (1.0 / n)


def Phi10(delta, a, b, odd=False):
    r = Grass()
    for x in S:
        if (parity(x) == 1) == odd:
            continue
        xd = x
        for mu, dm in enumerate(delta):
            xd = sh(xd, mu, dm)
        r = r + fa.Phi(idx[x], idx[xd], a, b)
    return r


def factor(img, cand):
    if not cand.t:
        return None
    m0, c0 = next(iter(cand.t.items()))
    cf = img.coeff(m0) / c0
    return cf if (img - cand * cf).maxabs() < 1e-10 else None


D00, D11 = D_col(0, 0), D_col(1, 1)
Ph = {(a, b): Phi10((1, 1), a, b) for a in range(4) for b in range(a, 4)}
Phb = {(a, b): Phi10((1, 1), a, b, odd=True) for a in range(4) for b in range(a, 4)}
t = {name: factor(image(D00, maps[name]), D00) for name in maps}
t["D11 shift_0"] = factor(image(D11, maps["shift_0"]), D11)
t["D11 shift_1"] = factor(image(D11, maps["shift_1"]), D11)
near = lambda v, cc: v is not None and abs(v - cc) < 1e-10
dimer = [t[k] for k in ("U(1)_eps", "Z4", "SU(4)", "shift_0", "shift_1", "D11 shift_0", "D11 shift_1")]
c.item(
    "columnar dimer D_00: factor under U(1)_eps, Z4, SU(4), shift_0, shift_1; D_11 under shift_0, shift_1",
    [None if v is None else round(float(np.real(v)), 4) for v in dimer],
    all(near(v, cc) for v, cc in zip(dimer, (1, 1, 1, -1, 1, 1, -1), strict=True)),
)
fU = factor(image(Ph[(0, 1)], maps["U(1)_eps"]), Ph[(0, 1)])
fZ = factor(image(Ph[(0, 1)], maps["Z4"]), Ph[(0, 1)])
fS = factor(image(Ph[(0, 1)], maps["shift_0"]), Phb[(0, 1)])
c.item(
    "Sym^2 Phi^01(x, x + (1,1)): factor under U(1)_eps (alpha = 0.7: e^{1.4 i}), under Z4; shift_0 -> Phibar^01",
    (np.round(fU, 4), np.round(fZ, 4), None if fS is None else round(float(abs(fS)), 4)),
    near(fU, np.exp(2j * 0.7)) and near(fZ, -1) and fS is not None and abs(abs(fS) - 1) < 1e-10,
)
inv10 = sum((Phb[k] * Ph[k] * (1.0 if k[0] == k[1] else 2.0) for k in Ph), Grass())
e_inv = (image(inv10, maps["SU(4)"]) - inv10).maxabs()
imgP = image(Ph[(0, 1)], maps["SU(4)"])
A = np.array([[Ph[k].coeff(m) for k in Ph] for m in imgP.t])
b = np.array([imgP.t[m] for m in imgP.t])
coef, *_ = np.linalg.lstsq(A, b, rcond=None)
e_span = np.linalg.norm(A @ coef - b)
c.item(
    "SU(4): sum Phibar Phi invariant; the image of Phi^01 lies in span(10) (residuals)",
    (f"{e_inv:.1e}", f"{e_span:.1e}"),
    e_inv < 1e-10 and e_span < 1e-10,
)

# ---- (4) free-theory weights on 4^4 (exact doublet propagator)
lat = Lattice((4, 4, 4, 4))
model = build_model(lat, 0.0, -0.01, 1.0, g10=0.1, g6=0.0, quads="all")
V = lat.V
Sd = np.linalg.inv(model.D.dense(np.zeros(model.nfields))).reshape(V, 2, V, 2)
Gab = lambda src, dst: real_flavour_block(Sd[src, :, dst, :])


def KK(u, v, up, vp):
    bp = np.einsum("naa,nbb->n", Gab(u, v), Gab(up, vp))
    ex = -np.einsum("nab,nab->n", Gab(u, up), Gab(v, vp)) + np.einsum("nab,nab->n", Gab(u, vp), Gab(v, up))
    return bp, ex


geom = model.geom
ql, lu, lv, sg = geom.qlinks, geom.lu, geom.lv, geom.lsign
parts = {}
for name, (i, j) in (("nu", (0, 1)), ("mu", (2, 3))):
    A_, B_ = ql[:, i], ql[:, j]
    bp, ex = KK(lu[A_], lv[A_], lu[B_], lv[B_])
    s = sg[A_] * sg[B_]
    parts[name] = (s * bp, s * ex)
TQ = 2 * sum(parts[k][0] + parts[k][1] for k in parts)
bond = 2 * sum(parts[k][0] for k in parts)
exch = 2 * sum(parts[k][1] for k in parts)
T6 = -2 * (parts["nu"][0] + parts["nu"][1] - parts["mu"][0] - parts["mu"][1])
comp = sum(
    KK(lu[ql[:, i]], lv[ql[:, i]], lu[ql[:, i]], lv[ql[:, i]])[0]
    + KK(lu[ql[:, i]], lv[ql[:, i]], lu[ql[:, i]], lv[ql[:, i]])[1]
    for i in range(4)
)
Kt_link = (sg * np.einsum("naa->n", Gab(lu, lv))).mean()
v = {k: float(np.mean(val)) for k, val in (("TQ", TQ), ("bond", bond), ("exch", exch), ("T6", T6), ("comp", comp))}
c.item("free 4^4 per quad: <Kt>_0 (= 1/2 exactly)", Kt_link, abs(Kt_link - 0.5) < 1e-10, fmt="{:.10f}")
c.item(
    "  <T_Q>_0 = bond product + exchange; 4<Kt>^2",
    (round(v["TQ"], 4), round(v["bond"], 4), round(v["exch"], 4), round(4 * Kt_link**2, 4)),
    abs(v["TQ"] - v["bond"] - v["exch"]) < 1e-12 and abs(v["bond"] - 4 * Kt_link**2) < 0.06,
)
c.item("  <T_6>_0, completion sum_l <K_l^2>_0", (round(v["T6"], 4), round(v["comp"], 4)), v["comp"] < v["TQ"])
c.item(
    "  exchange / T_Q, completion / exchange",
    (round(v["exch"] / v["TQ"], 3), round(v["comp"] / v["exch"], 3)),
    v["exch"] < 0.25 * v["TQ"] and v["comp"] > 2.5 * v["exch"],
)

# ---- (5) a sign-free same-parity source
model = build_model(lat, 2.41, -0.01, 1.0, g10=0.1, g6=0.0, quads="all")
rng = np.random.default_rng(5)
eps = lat.eps_np.reshape(V)
idx4 = np.arange(V).reshape(lat.shape)
worst = 0.0
for _ in range(10):
    Dc = model.D.dense(np.asarray(model.start(rng, "hot+hot")))
    C = np.zeros((V, V))
    for mu in range(4):
        for nu in range(mu + 1, 4):
            for sgn in (+1, -1):
                j = np.roll(np.roll(idx4, -1, axis=mu), -sgn, axis=nu).reshape(V)
                amp = rng.normal(size=V) * 0.3
                C[np.arange(V), j] += amp
                C[j, np.arange(V)] -= amp
    assert np.allclose(C, -C.T) and np.all(eps[np.nonzero(C)[0]] == eps[np.nonzero(C)[1]])
    sign, _ = np.linalg.slogdet(Dc + np.kron(C, np.eye(2)))
    worst = max(worst, abs(sign - 1.0))
c.item(
    "det D_c(sigma, s; hC) with a random same-parity flavour-blind real antisymmetric C (amp 0.3), 10 hot 4^4 "
    "configurations: max |sign - 1|",
    worst,
    worst < 1e-10,
    fmt="{:.1e}",
)
c.done()
