"""I.4: the link-field (wedge) HMC is exact, and the per-link completion model is the wedge model without the plaquette
term, exactly.

Model: D_c(sigma, s) = (K - A(s)) (x) 1_2 + i y sigma.tau, weight det D_c = det(D_c^dagger D_c)^(1/2) > 0, link fields
s_{Q,j} with prior s^2/(4 g_j), g_1 = g10 - g6 (nu pairs), g_2 = g10 + g6 (mu pairs), on every square ("all") or the
checkerboard set; per-link model ("links"): one field per link, prior s^2/(4 c_l g), c_l = number of squares through l.
(1) Geometry: orientation s_l = -sign K[u, v] = sign of the free propagator on every link (2^3, 4^2, 4^3, 2^4, 4^4);
    checkerboard quads = plaquette quads; dense operator at y = 0 = K - A(s) built independently; the sum-of-squares
    identity T_Q + sum_l K_l^2 = (Kt_nu + Kt_nu')^2 + (Kt_mu + Kt_mu')^2 (and its g6 version) as Grassmann identities on
    quads of both orientations; per-link model: fields = links, c_l, incidence diag(lsign), E e^{s Kt} = e^{c g K^2}
    to all orders, sum_Q [squares - T_Q] = sum_l c_l K_l^2.
(2) Exact 2^3 (y = 0): HS partition function by polynomial algebra = all-orders bag enumeration of the completed vertex
    (interior and edge points), and of c g K_l^2 for the per-link model; controls (wrong sign in one Kt term, the
    uncompleted vertex, the plaquette term kept, width 2g) differ.
(3) HMC on 2^3 reproduces the exact <sum s^2>, <sum term>, <sum s> and the fermionic bond estimator.
(4) 4^4: forces vs finite differences, reversibility, <|dH|> ~ dtau^2, Hasenbusch terms, nested integrator; per-link
    model: det D_c > 0 on hot configurations, forces, reversibility.
(5) 4^4: stochastic estimators of the channel measure vs dense on a thermalised configuration.
(6) 2^4 (all antiperiodic): exact-determinant Metropolis vs RHMC for the wedge model and for the per-link model
    (trajectory-length jitter 0.3), 10 observables; the controls (wrong sign in one Kt term; prior with c_l = 1) are
    rejected. Chains frozen in data/derived/k3/chains/, prefixes regenerated bit-identically here.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.action import build_model
from masspairing.algebra.grassmann import Grass
from masspairing.algebra.staggered import FieldAlgebra, oriented_links, plaquette_quads, staggered_matrix
from masspairing.analysis import k3_exact as kx
from masspairing.claimcheck import Check
from masspairing.data import DERIVED
from masspairing.hmc import HMC, MTSHMC
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.measure import ChannelMeasure
from masspairing.operator import DoubletOperator
from masspairing.wedge import WedgeGeometry

CH = DERIVED / "k3" / "chains"
KEYS6 = ("sigma2", "O4", "Sigma_stag_abs", "phi_stag_sq", "s2", "E_bond", "dimer_sq_par", "chi6", "E10", "E6")
PREFIX = {
    "wedge_2x3_hmc": 20,
    "wedge_2x4_metropolis": 200,
    "wedge_2x4_rhmc": 10,
    "wedge_2x4_rhmc_flip": 10,
    "links_2x4_metropolis": 200,
    "links_2x4_rhmc": 10,
    "links_2x4_rhmc_prior": 10,
}
c = Check("I.4")
T0 = time.time()

# ================================================================ (1) geometry
rng = np.random.default_rng(70)
ok = True
worst_A = 0.0
for L, d in ((2, 3), (4, 2), (4, 3), (2, 4), (4, 4)):
    lat = Lattice((L,) * d, bc=(-1,) * d)
    M, S, idx = staggered_matrix(L, d)
    K0 = kinetic_matrix(lat).toarray()
    ok &= np.allclose(K0, 2 * M)
    ls = oriented_links(M, S, idx, L, d)
    g = WedgeGeometry(lat, "checkerboard", 0.3, 0.1)
    agree = {ls[(int(u), int(v))] * g.lsign[i] for i, (u, v) in enumerate(zip(g.lu, g.lv, strict=True))}
    ok &= agree == {1.0} and g.n_links == len(ls)
    ok &= {frozenset(q) for q in plaquette_quads(L, d, S, idx)} == {frozenset(q) for q in g.qsites}
    s = rng.normal(size=g.n_fields)
    R = DoubletOperator(lat, 0.0, geom=g).dense_real(np.concatenate([np.zeros(3 * lat.V), s]))
    A = np.zeros((lat.V,) * 2)  # A[u, v] += s * sign G_uv per link of each pair
    eps = lat.eps_np.reshape(-1)
    for q, (x, xp_, y, yp) in enumerate(g.qsites):
        for j, links in ((0, ((x, yp), (xp_, y))), (1, ((x, y), (xp_, yp)))):
            f = np.where((g.f_quad == q) & (g.f_pair == j))[0]
            for u, v in links:
                u, v = (u, v) if eps[u] > 0 else (v, u)
                A[u, v] += s[f[0]] * ls[(u, v)]
                A[v, u] -= s[f[0]] * ls[(u, v)]
    worst_A = max(worst_A, abs(R - (2 * M - A)).max())
c.item(
    "(1) 2^3, 4^2, 4^3, 2^4, 4^4: orientation = sign of the free propagator on every link, checkerboard quads = "
    "plaquette quads, dense K - A(s) vs independent construction",
    worst_A,
    ok and worst_A < 1e-12,
    fmt="{:.1e}",
)
worst = 0.0
for L, d, nq in ((4, 2, None), (4, 3, 24)):
    lat = Lattice((L,) * d, bc=(-1,) * d)
    g = WedgeGeometry(lat, "all", 1.0, 0.0)
    fa = FieldAlgebra(lat.V)
    qs = range(g.n_quads) if nq is None else rng.choice(g.n_quads, nq, replace=False)
    for q in qs:
        x, xp_, y, yp = map(int, g.qsites[q])
        ln = g.qlinks[q]
        K = [fa.K(int(g.lu[i]), int(g.lv[i])) for i in ln]
        Kt = [K[i] * float(g.lsign[l]) for i, l in enumerate(ln)]
        B1, B2 = Kt[0] + Kt[1], Kt[2] + Kt[3]
        lhs = fa.sym2_term((x, xp_, y, yp)) + sum((k * k for k in K), Grass())
        worst = max(worst, max([abs(v) for v in (lhs - (B1 * B1 + B2 * B2)).t.values()] + [0]))
        g6 = 0.4
        lhs6 = (
            fa.sym2_term((x, xp_, y, yp))
            + fa.lam2_term((x, xp_, y, yp)) * g6
            + (K[0] * K[0] + K[1] * K[1]) * (1 - g6)
            + (K[2] * K[2] + K[3] * K[3]) * (1 + g6)
        )
        worst = max(worst, max([abs(v) for v in (lhs6 - (B1 * B1 * (1 - g6) + B2 * B2 * (1 + g6))).t.values()] + [0]))
    ok &= {o[2] for o in g.qorient} == {1, -1}
c.item(
    "(1) sum-of-squares identity (g6 = 0 and 0.4) on all 16 quads of 4^2 and 24 random quads of 4^3, both "
    "orientations: largest coefficient of the difference",
    worst,
    worst < 1e-12 and ok,
    fmt="{:.1e}",
)
ok1 = True
for shape, cnt in (((4, 4, 4, 4), 6), ((2, 2, 2, 2), 3), ((2, 2, 2), 2), ((4, 4), 2)):
    lat = Lattice(shape, bc=(-1,) * len(shape))
    g, ga = WedgeGeometry(lat, "links", 0.2, 0.0), WedgeGeometry(lat, "all", 0.2, 0.0)
    ok1 &= np.all(g.link_count == cnt) and g.n_fields == g.n_links == ga.n_links
    ok1 &= np.allclose(g.inv4g, 1 / (4 * cnt * 0.2)) and abs(g.I - np.diag(g.lsign)).max() < 1e-15
    ok1 &= np.array_equal(g.lsign, ga.lsign)
fa = FieldAlgebra(2)
K = fa.K(0, 1)
cg = 6 * 0.17
sig2 = 2 * cg
mom = {2: sig2, 4: 3 * sig2**2, 6: 15 * sig2**3, 8: 105 * sig2**4}
lhs, Kn, fact = Grass.one(), Grass.one(), 1.0
for n in range(1, 9):
    Kn = Kn * K
    fact *= n
    if n % 2 == 0 and Kn.t:
        lhs = lhs + Kn * (mom[n] / fact)
e_id = max([abs(v) for v in (lhs - (K * K * cg).exp_nilpotent(8)).t.values()] + [0.0])
lat2d = Lattice((4, 4), bc=(-1, -1))
ga, gl = WedgeGeometry(lat2d, "all", 1.0, 0.0), WedgeGeometry(lat2d, "links", 1.0, 0.0)
fa = FieldAlgebra(lat2d.V)
tot = Grass()
for q in range(ga.n_quads):
    ln = ga.qlinks[q]
    Kt = [fa.K(int(ga.lu[i]), int(ga.lv[i])) * float(ga.lsign[i]) for i in ln]
    tot = (
        tot
        + (Kt[0] + Kt[1]) * (Kt[0] + Kt[1])
        + (Kt[2] + Kt[3]) * (Kt[2] + Kt[3])
        - fa.sym2_term(tuple(map(int, ga.qsites[q])))
    )
comp = Grass()
for i in range(gl.n_links):
    Kl = fa.K(int(gl.lu[i]), int(gl.lv[i]))
    comp = comp + Kl * Kl * float(gl.link_count[i])
e_c = max([abs(v) for v in (tot - comp).t.values()] + [0.0])
c.item(
    "(1) per-link model: fields = links, c_l = 6/3/2/2 on 4^4/2^4/2^3/4^2, prior 1/(4 c_l g), incidence diag(lsign);"
    " E e^{s Kt} = e^{c g K^2} to all orders; sum_Q [squares - T_Q] = sum_l c_l K_l^2 on 4^2",
    (e_id, e_c, len(comp.t)),
    ok1 and e_id < 1e-12 and e_c < 1e-12 and len(comp.t) > 0,
    fmt="{0[0]:.1e}, {0[1]:.1e} ({0[2]} monomials)",
)

# ================================================================ (2) exact 2^3
bags = json.loads((DERIVED / "k3" / "bags_2x2x2.json").read_text())
exact = {}
for name, (g10, g6) in (("interior", (0.4, 0.1)), ("edge+", (0.5, 0.5))):
    geom = kx.model_2x3(g10, g6).geom
    r = exact[name] = kx.hs_exact(geom, geom.K0.toarray())
    rb = bags[name]
    dz = abs(r["Z_over_Z0"] - rb["Z_over_Z0"]) / rb["Z_over_Z0"]
    dt = abs(r["sum_term"] - rb["sum_term"]) / abs(rb["sum_term"])
    c.item(
        f"(2) 2^3 {name} (g10, g6) = ({g10}, {g6}): Z/Z0 HS-exact vs bag enumeration ({rb['n_configs']:.0f} "
        "configurations, all orders), <sum term>: relative differences",
        (r["Z_over_Z0"], dz, r["sum_term"], dt),
        dz < 1e-9 and dt < 1e-8,
        fmt="{0[0]:.10f} ({0[1]:.1e}); {0[2]:.8f} ({0[3]:.1e})",
    )
geom = kx.flip_nu_sign(kx.model_2x3(0.4, 0.1).geom)
zA = kx.hs_exact(geom, geom.K0.toarray())["Z_over_Z0"]
dA = abs(zA - exact["interior"]["Z_over_Z0"]) / exact["interior"]["Z_over_Z0"]
dB = abs(bags["pure"]["Z_over_Z0"] - exact["edge+"]["Z_over_Z0"]) / exact["edge+"]["Z_over_Z0"]
c.item(
    "(2) controls: wrong sign in one Kt term of every nu pair (interior); the uncompleted vertex g10 T_Q + g6 T_6 "
    "(edge): Z/Z0 and relative difference (rejected if > 1e-3, > 1e-2)",
    (zA, dA, bags["pure"]["Z_over_Z0"], dB),
    dA > 1e-3 and dB > 1e-2,
    fmt="{0[0]:.6f} ({0[1]:.2e}); {0[2]:.6f} ({0[3]:.2e})",
)
gl3 = kx.model_2x3(0.3, 0.0, quads="links").geom
zl = kx.hs_exact(gl3, gl3.K0.toarray(), sigma=np.sqrt(1.0 / (2.0 * gl3.inv4g)))["Z_over_Z0"]
zw = kx.hs_exact(gl3, gl3.K0.toarray(), sigma=np.sqrt(2.0 * 0.3) * np.ones(gl3.n_fields))["Z_over_Z0"]
za = kx.hs_exact(kx.model_2x3(0.3, 0.0).geom, gl3.K0.toarray())["Z_over_Z0"]
bl = kx.bag_reference(gl3, "links")
dl = abs(zl - bl["Z_over_Z0"]) / bl["Z_over_Z0"]
c.item(
    "(2) per-link model 2^3, g = 0.3: Z/Z0 HS-exact vs live bag enumeration of c g K_l^2 (configurations); the "
    "'all' model with g T_Q and the width 2g (c_l = 1) differ",
    (zl, bl["n_configs"], dl, za, zw),
    dl < 1e-9 and abs(za - zl) / zl > 1e-2 and abs(zw - zl) / zl > 1e-2,
    fmt="{0[0]:.10f} ({0[1]} configurations, rel {0[2]:.1e}); all-set {0[3]:.6f}, width 2g {0[4]:.6f}",
)

# ================================================================ frozen chains: live prefixes
chains = {name: kx.load_chain(CH / f"{name}.npz") for name in PREFIX}
for name, n in PREFIX.items():
    live = kx.run_chain(name, n)
    same = all(np.array_equal(live[k], chains[name][k][: len(live[k])]) for k in chains[name])
    c.item(f"live prefix of {name} (n = {n}) bit-identical to the frozen chain", same, same)

# ================================================================ (3) HMC on 2^3 vs exact
r = exact["interior"]
h3 = chains["wedge_2x3_hmc"]
pulls = []
for k, ex in (
    ("s2", r["sum_s2"]),
    ("term", r["sum_term"]),
    ("s1", float(r["s1"].sum())),
    ("bond", float(r["s1"].sum())),
):
    v, e = kx.blocked(h3[k])
    pulls.append((v - ex) / e)
    c.record(f"    2^3 HMC {k}", f"{v:.5f}({e:.5f}) vs exact {ex:.5f}: {pulls[-1]:+.2f} sigma")
c.item(
    "(3) HMC on 2^3 (3000 trajectories after 300, 5 steps) vs exact <sum s^2>, <sum term>, <sum s>, fermionic bond "
    "estimator: pulls (< 3.5)",
    pulls,
    max(abs(p) for p in pulls) < 3.5,
    fmt="{0[0]:+.2f}, {0[1]:+.2f}, {0[2]:+.2f}, " "{0[3]:+.2f}",
)

# ================================================================ (4) 4^4 forces, reversibility, dH, Hasenbusch
lat4 = Lattice((4, 4, 4, 4))
m = build_model(lat4, 1.7, -0.01, 1.0, g10=0.25, g6=0.05, cg_tol=1e-12, lanczos_tol=1e-12, n_rational=20)
rng = np.random.default_rng(74)
F = m.start(rng, "hot")
phi, _ = m.heatbath_phi(F, lat4.random_psi(rng))
w = np.linalg.eigvalsh(m.D.dense(F).conj().T @ m.D.dense(F))
m.set_window(w.min(), w.max())
fpf, _ = m.f_pf(F, phi)
fb = m.f_boson(F)
eps_fd = 1e-3


def fd4(fun, i, F0):
    out = []
    for h in (2 * eps_fd, eps_fd, -eps_fd, -2 * eps_fd):
        Fp = F0.copy()
        Fp[i] += h
        out.append(fun(Fp))
    return -(-out[0] + 8 * out[1] - 8 * out[2] + out[3]) / (12 * eps_fd)


worst = 0.0
for i in list(rng.integers(0, m.nsig, size=2)) + list(rng.integers(m.nsig, m.nfields, size=3)):
    a, b = fd4(lambda X: m.s_pf(X, phi), i, F), fd4(m.s_boson, i, F)
    worst = max(worst, abs(a - fpf[i]) / (abs(a) + 1e-3), abs(b - fb[i]) / (abs(b) + 1e-3))
c.item(
    "(4) 4^4 (1.7, -0.01, 1), g10 = 0.25, g6 = 0.05: sigma and s forces vs 4-point finite differences of the "
    "Lanczos action (worst relative)",
    worst,
    worst < 1e-5,
    fmt="{:.1e}",
)
h = HMC(m, tau=1.0, nsteps=8, seed=74)
rv = {}
for tol in (1e-12, 1e-8):
    m.cg_tol = tol
    rv[tol] = h.reversibility(F)
c.item(
    "(4) reversibility |d fields|/|fields|, |d pi|/|pi| at cg_tol 1e-12 and 1e-8",
    (rv[1e-12]["dsigma"], rv[1e-12]["dpi"], rv[1e-8]["dsigma"], rv[1e-8]["dpi"]),
    rv[1e-12]["dsigma"] < 1e-12 and rv[1e-12]["dpi"] < 1e-12 and rv[1e-8]["dsigma"] < 1e-7 and rv[1e-8]["dpi"] < 1e-7,
    fmt="{0[0]:.1e} / {0[1]:.1e}; {0[2]:.1e} / {0[3]:.1e}",
)
m.cg_tol, m.lanczos_tol, m.n_rational = 1e-8, 1e-10, 12
F = m.start(rng, "hot+zero")
for _ in range(20):
    F, info = h.trajectory(F, nsteps=16)
res = {}
for n in (6, 12, 24):
    res[n] = float(np.mean([abs(h.trajectory(F, nsteps=n)[1]["dH"]) for _ in range(6)]))
r1, r2 = res[6] / res[12], res[12] / res[24]
c.item(
    "(4) <|dH|> at 6 / 12 / 24 steps on a thermalised configuration; ratios (expect 4)",
    (res[6], res[12], res[24], r1, r2),
    2.5 < r1 < 9.0 and 2.5 < r2 < 9.0,
    fmt="{0[0]:.4f} / {0[1]:.4f} / {0[2]:.4f}; {0[3]:.2f}, {0[4]:.2f}",
)
mh = build_model(
    lat4, 1.7, -0.01, 1.0, g10=0.25, g6=0.05, hasenbusch=[0.1, 1.0], cg_tol=1e-12, lanczos_tol=1e-12, n_rational=20
)
phis = [mh.heatbath_term(k, F, lat4.random_psi(rng))[0] for k in range(mh.nterms)]
w = np.linalg.eigvalsh(mh.D.dense(F).conj().T @ mh.D.dense(F))
mh.set_window(w.min(), w.max())
wk = []
for k in range(mh.nterms):
    fk, _ = mh.f_term(k, F, phis[k])
    wk.append(0.0)
    for i in (int(rng.integers(0, mh.nsig)), int(rng.integers(mh.nsig, mh.nfields))):
        a = fd4(lambda X, k=k: mh.s_term(k, X, phis[k]), i, F)
        wk[-1] = max(wk[-1], abs(a - fk[i]) / (abs(a) + 1e-3))
rvm = MTSHMC(mh, tau=1.0, levels=(4, 2, 2), seed=75).reversibility(F)
c.item(
    "(4) Hasenbusch terms (0, 0.1), (0.1, 1), heavy(1): forces vs finite differences; nested (4, 2, 2) "
    "reversibility",
    (*wk, rvm["dsigma"], rvm["dpi"]),
    max(wk) < 1e-5 and rvm["dsigma"] < 1e-12 and rvm["dpi"] < 1e-11,
    fmt="{0[0]:.1e}, {0[1]:.1e}, {0[2]:.1e}; {0[3]:.1e} / {0[4]:.1e}",
)
ml = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1, quads="links", cg_tol=1e-12, lanczos_tol=1e-12, n_rational=20)
rl = np.random.default_rng(111)
lds, ok3 = [], True
for _ in range(10):
    D = ml.D.dense(np.asarray(ml.start(rl, "hot+hot")))
    sgn, ld = np.linalg.slogdet(D)
    lds.append(ld)
    ok3 &= abs(sgn - 1.0) < 1e-10 and abs(D + D.conj().T).max() < 1e-13
Fl = ml.start(rl, "hot+hot")
phil, _ = ml.heatbath_phi(Fl, lat4.random_psi(rl))
w = np.linalg.eigvalsh(ml.D.dense(Fl).conj().T @ ml.D.dense(Fl))
ml.set_window(w.min(), w.max())
fpl, _ = ml.f_pf(Fl, phil)
fbl = ml.f_boson(Fl)
worst = 0.0
for i in list(rl.integers(0, ml.nsig, size=2)) + list(rl.integers(ml.nsig, ml.nfields, size=3)):
    a, b = fd4(lambda X: ml.s_pf(X, phil), i, Fl), fd4(ml.s_boson, i, Fl)
    worst = max(worst, abs(a - fpl[i]) / (abs(a) + 1e-3), abs(b - fbl[i]) / (abs(b) + 1e-3))
rvl = HMC(ml, tau=1.0, nsteps=8, seed=111).reversibility(Fl)
c.item(
    "(4) per-link model 4^4 (y = 2.41, g = 0.1): det D_c > 0 and anti-hermitian on 10 hot configurations (log det "
    "range); forces vs finite differences; reversibility",
    (min(lds), max(lds), worst, rvl["dsigma"], rvl["dpi"]),
    ok3 and worst < 1e-5 and rvl["dsigma"] < 1e-12 and rvl["dpi"] < 1e-11,
    fmt="{0[0]:.1f} .. {0[1]:.1f}; {0[2]:.1e}; {0[3]:.1e} / {0[4]:.1e}",
)

# ================================================================ (5) estimators on the thermalised 4^4 configuration
ex = ChannelMeasure(m, exact=True).bilinears(F)
K5 = ("O4", "phi_stag_sq", "phi_sq", "chi10", "chi6", "E10", "E6", "dimer_sq_par", "dimer_sq_perp")
vals = {k: [] for k in K5}
Eb = []
for seed in range(4):
    o = ChannelMeasure(m, n_noise=24, seed=100 + seed, cg_tol=1e-10).bilinears(F)
    for k in K5:
        vals[k].append(o[k])
    Eb.append(o["E_bond"])
pz = []
for k in K5:
    v = np.asarray(vals[k])
    pz.append((v.mean() - ex[k]) / (v.std(ddof=1) / 2))
    c.record(
        f"    4^4 {k}", f"dense {ex[k]:+.5f}, stochastic {v.mean():+.5f}({v.std(ddof=1) / 2:.5f}): {pz[-1]:+.2f} sigma"
    )
Eb = np.asarray(Eb)
zb = float(np.abs((Eb.mean(0) - ex["E_bond"]) / (Eb.std(0, ddof=1) / 2)).max())
c.item(
    "(5) 4^4: stochastic estimators (24 noise vectors x 4 seeds) vs dense on 9 observables and the 4 bond energies:"
    " largest |pull| (< 4)",
    max(max(abs(p) for p in pz), zb),
    max(max(abs(p) for p in pz), zb) < 4.0,
    fmt="{:.2f}",
)

# ================================================================ (6) 2^4 Metropolis vs RHMC
for tag, label in (("wedge", "wedge model g10 = 0.4, g6 = 0.1"), ("links", "per-link model g = 0.15")):
    met, rh = chains[f"{tag}_2x4_metropolis"], chains[f"{tag}_2x4_rhmc"]
    sab = chains["wedge_2x4_rhmc_flip" if tag == "wedge" else "links_2x4_rhmc_prior"]
    worst = worst_sab = 0.0
    for k in KEYS6:
        vm, em = kx.blocked(met[k])
        vh, eh = kx.blocked(rh[k])
        vx, exx = kx.blocked(sab[k])
        z, zx = (vm - vh) / np.hypot(em, eh), (vm - vx) / np.hypot(em, exx)
        worst, worst_sab = max(worst, abs(z)), max(worst_sab, abs(zx))
        c.record(
            f"    2^4 {tag} {k}",
            f"Metropolis {vm:+.5f}({em:.5f}) RHMC {vh:+.5f}({eh:.5f}) {z:+.2f} | control "
            f"{vx:+.5f}({exx:.5f}) {zx:+.1f}",
        )
    chi10max = max(np.abs(met["chi10"]).max(), np.abs(rh["chi10"]).max())
    c.item(
        f"(6) 2^4 {label}: Metropolis (10000 sweeps) vs RHMC (2000 trajectories, acceptance "
        f"{np.mean(rh['accepted']):.2f}): largest pull over {len(KEYS6)} observables (< 3.5); control largest pull "
        "(> 5); chi10 = 0 identically on L = 2",
        (worst, worst_sab, chi10max),
        worst < 3.5 and worst_sab > 5 and chi10max < 1e-12,
        fmt="{0[0]:.2f}; {0[1]:.1f}; {0[2]:.0e}",
    )
rh = chains["links_2x4_rhmc"]
c.record("    per-link model <s^2>/(6 g) on 2^4 (c_l = 3)", f"{np.mean(rh['s2']) / (6 * 0.15):.3f}")
c.record("    runtime", f"{time.time() - T0:.0f} s")
c.done()
