"""I.8: the flavour-selective source in the real 4V basis -- operator, |Pf| RHMC and real-basis observables exact.

(1) operator (4^4): M(h) = Re[W D_c W^dagger] - h C_0 (x) diag(0,0,0,1) is real antisymmetric; the packed apply equals
    M chi (single vectors and batches); apply_dag = -apply; the all-flavour mask reproduces the flavour-blind operator
    D_c - h C_0 (x) 1_2; Pf M(0) = det D_c > 0 (Householder and Parlett-Reid).
(2) 4^4 wedge (g10, g6) = (0.25, 0.05), h = 0.1 on flavour 4: Lanczos S_pf = dense phi^T (M^T M)^(-1/2) phi; forces vs
    4-point finite differences; reversibility.
(3) 2^3 x 4: exact Metropolis on |Pf M| e^(-S_B) vs the RHMC on det(M^T M)^(1/4), 17 sign-reweighted observables within
    3 sigma; the sabotage (C_0 with the wrong sign on the odd sublattice) rejected (> 5 sigma). Frozen chains of
    scripts/derive_n1inst.py i8; the first sweeps / trajectories are regenerated live and must be bit-identical.
(4) observables: at h = 0 the real-basis estimators equal the doublet channel measure; chi_f_conn = d<O^(a)>/dh_a
    (sabotage fails); free theory with the flavour-4 source; stochastic = dense.
About 8 min.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.action import build_model
from masspairing.analysis import n1inst as X
from masspairing.claimcheck import Check
from masspairing.data import derived
from masspairing.flavour import flavour_mask, realify, unpack
from masspairing.hmc import HMC
from masspairing.lattice import Lattice
from masspairing.measure import ChannelMeasure, N1Measure
from masspairing.measure.n1 import _pf_small
from masspairing.patterns import source_pattern
from masspairing.pfaffian import pfaffian_householder, pfaffian_logpr

c = Check("I.8")
MASK = np.array([0.0, 0.0, 0.0, 1.0])
lat4 = Lattice((4, 4, 4, 4))
V = lat4.V

# ---- (1) operator
rng = np.random.default_rng(140)
m = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1, h_sel=0.3, mask=MASK)
mb = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1, h_sel=0.3, mask=(1, 1, 1, 1))
mw = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1, h=0.3)
m0 = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1)
C0 = source_pattern(lat4).toarray()
e_anti = e_ap = e_bl = e_pf = 0.0
for _ in range(4):
    F = np.asarray(m.start(rng, "hot+hot"))
    M = m.D.dense_real4(F)
    Y = m.D.prepare(F)
    e_anti = max(
        e_anti, abs(M + M.T).max(), abs(M - (realify(m.D.dense(F), V) - 0.3 * np.kron(C0, np.diag(MASK)))).max()
    )
    psi = lat4.random_psi(rng)
    B = lat4.random_psi(rng, n=3).transpose(1, 0, 2)
    e_ap = max(
        e_ap,
        abs(unpack(m.D.apply(psi, Y=Y)) - (M @ unpack(psi).reshape(-1)).reshape(V, 4)).max(),
        abs(m.D.apply_dag(psi, Y=Y) + m.D.apply(psi, Y=Y)).max(),
        max(
            abs(unpack(m.D.apply(B, Y=Y)[:, i, :]) - (M @ unpack(B[:, i, :]).reshape(-1)).reshape(V, 4)).max()
            for i in range(3)
        ),
    )
    e_bl = max(
        e_bl,
        abs(mb.D.dense_real4(F) - realify(mw.D.dense(F), V)).max(),
        abs(mb.D.apply(psi, Y=mb.D.prepare(F)) - mw.D.apply(psi, Y=mw.D.prepare(F))).max(),
    )
    Dc = m0.D.dense(F)
    s, ld = np.linalg.slogdet(Dc)
    M0 = realify(Dc, V)
    sh, lh = pfaffian_householder(M0)
    sp, lp = pfaffian_logpr(M0)
    e_pf = max(e_pf, abs(sh - 1.0), abs(sp - 1.0), abs(lh - ld) / ld, abs(lp - ld) / ld)
c.item("4^4: M(h) antisymmetric and = Re[W D_c W^dagger] - h C_0 (x) P", e_anti, e_anti < 1e-13, "{:.1e}")
c.item("packed apply = M chi (single, batch); apply_dag = -apply", e_ap, e_ap < 1e-12, "{:.1e}")
c.item("all-flavour mask = flavour-blind operator D_c - h C_0 (x) 1_2", e_bl, e_bl < 1e-12, "{:.1e}")
c.item("Pf M(0) = det D_c > 0 (Householder, Parlett-Reid; relative log)", e_pf, e_pf < 1e-10, "{:.1e}")

# ---- (2) action, forces, reversibility
m = build_model(
    lat4, 1.7, -0.01, 1.0, g10=0.25, g6=0.05, h_sel=0.1, mask=MASK, cg_tol=1e-12, lanczos_tol=1e-12, n_rational=20
)
F = m.start(rng, "hot")
phi, _ = m.heatbath_phi(F, lat4.random_psi(rng))
M = m.D.dense_real4(F)
w, U = np.linalg.eigh(M.T @ M)
m.set_window(w.min(), w.max())
ph = unpack(phi).reshape(-1)
s_dense = float(ph @ (U @ ((U.T @ ph) / np.sqrt(w))))
e_act = abs(m.s_pf(F, phi) - s_dense) / abs(s_dense)
fpf, _ = m.f_pf(F, phi)
eps_ = 1e-3


def fd4(i):
    out = []
    for hh in (2 * eps_, eps_, -eps_, -2 * eps_):
        Fp = F.copy()
        Fp[i] += hh
        out.append(m.s_pf(Fp, phi))
    return -(-out[0] + 8 * out[1] - 8 * out[2] + out[3]) / (12 * eps_)


worst = 0.0
for i in list(rng.integers(0, m.nsig, size=3)) + list(rng.integers(m.nsig, m.nfields, size=3)):
    a = fd4(i)
    worst = max(worst, abs(a - fpf[i]) / (abs(a) + 1e-3))
rv = HMC(m, tau=1.0, nsteps=8, seed=140).reversibility(F)
c.item("4^4 wedge (0.25, 0.05), h = 0.1 on flavour 4: Lanczos S_pf vs dense", e_act, e_act < 1e-9, "{:.1e}")
c.item("sigma- and s-forces vs 4-point finite differences (6 components)", worst, worst < 1e-5, "{:.1e}")
c.item("reversibility |d fields|, |d pi|", (rv["dsigma"], rv["dpi"]), rv["dsigma"] < 1e-12 and rv["dpi"] < 1e-11)

# ---- (3) Metropolis vs RHMC (frozen chains, live prefixes)
z = np.load(derived("n1inst", "I8_chains.npz"))
part = {}
for k in z.files:
    name, key = k.split("__", 1)
    part.setdefault(name, {})[key] = z[k]
live = X.i8_metropolis(206, X.I8_MET[0][1], record_trace=True)
e_m = max(
    abs(live["trace_sb"] - part["met0"]["trace_sb"][:206]).max(), abs(live["sign"] - part["met0"]["sign"][:2]).max()
)
e_m = max([e_m] + [abs(live[k] - part["met0"][k][:2]).max() for k in X.SELECTIVE_KEYS])
c.item("live Metropolis prefix (206 sweeps, 2 measurements) = frozen chain", e_m, e_m == 0.0, "{:.0e}")
for name, (_n, seed, sab) in zip(("rhmc", "sabotage"), X.I8_RHMC, strict=True):
    lv = X.i8_rhmc(2, seed, sab)
    e_r = max(
        [abs(lv["dH"] - part[name]["dH"][:202]).max()]
        + [abs(lv[k] - part[name][k][:2]).max() for k in X.SELECTIVE_KEYS]
    )
    c.item(f"live {name} prefix (202 trajectories, 2 measurements) = frozen chain", e_r, e_r == 0.0, "{:.0e}")
met = {k: np.concatenate([part["met0"][k], part["met1"][k]]) for k in X.SELECTIVE_KEYS + ["sign"]}
sh, sx = part["rhmc"], part["sabotage"]
c.record(
    "Metropolis 2 x 3000 sweeps (acc), RHMC 1200 trajectories (acc), sabotage 400 (acc)",
    [float(part["met0"]["acc"]), float(sh["acc"]), float(sx["acc"])],
)
c.item(
    "negative Pfaffians: Metropolis, RHMC (counts of measured configurations)",
    [f"{int((met['sign'] < 0).sum())}/{len(met['sign'])}", f"{int((sh['sign'] < 0).sum())}/{len(sh['sign'])}"],
    (met["sign"] > 0).all() and (sh["sign"] > 0).all(),
)
worst = worst_sab = 0.0
for k in X.SELECTIVE_KEYS:
    vm, em = X.reweighted(met[k], met["sign"])
    vh, eh = X.reweighted(sh[k], sh["sign"])
    vx, ex = X.reweighted(sx[k], sx["sign"])
    p, px = X.pull(vm, em, vh, eh), X.pull(vm, em, vx, ex)
    worst, worst_sab = max(worst, abs(p)), max(worst_sab, abs(px))
    c.record(
        f"{k:16s} Metropolis / RHMC / sabotage",
        f"{vm:+.5f}({em:.5f}) {vh:+.5f}({eh:.5f}) pull {p:+.2f} | {vx:+.5f}({ex:.5f}) pull {px:+.1f}",
    )
c.item("largest Metropolis-RHMC pull over 17 observables (<= 3)", worst, worst <= 3.0, "{:.2f}")
c.item("sabotage rejected: largest pull (> 5)", worst_sab, worst_sab > 5.0, "{:.1f}")

# ---- (4) observables
rng = np.random.default_rng(142)
m0 = build_model(lat4, 2.41, -0.01, 1.0, g10=0.05, cg_tol=1e-11)
F = np.asarray(m0.start(rng, "hot+hot"))
b1 = N1Measure(m0, exact=True).bilinears(F)
b2 = ChannelMeasure(m0, exact=True).bilinears(F)
e1 = max(abs(np.asarray(b1[k]) - np.asarray(b2[k])).max() for k in ("phi", "phi_stag", "O4", "E_bond", "dimer_sq"))
e1 = max(
    e1,
    abs(b1["phi_f"].sum() - b2["phi_src"]),
    abs(b1["phi_f_even"].sum() - b2["phi_src_even"]),
    abs(b1["phi_f_odd"].sum() - b2["phi_src_odd"]),
)
x0 = (1, 2, 0, 3)
c1 = N1Measure(m0, exact=False).correlator(F, x0=x0)
c2 = ChannelMeasure(m0, exact=False).correlator(F, x0=x0)
e1c = max(abs(c1["Cf_f"].sum(0) / 2 - c2["Cf"]).max(), abs(c1["Gf_f"].sum(0) / 4 - c2["Gf"].real).max())
A = rng.normal(size=(6, 6, 6))
A = A - A.transpose(0, 2, 1)
e1p = max(
    abs(_pf_small(A)[i] - pfaffian_householder(A[i])[0] * np.exp(pfaffian_householder(A[i])[1])) for i in range(6)
)
c.item(
    "h = 0: phi, phi_stag, O4, E_bond, dimer_sq, sum_a phi_f(_even/_odd) = channel measure", e1, e1 < 1e-12, "{:.1e}"
)
c.item("h = 0: sum_a Cf_f / 2 = Cf, sum_a Gf_f / 4 = Re Gf (same point source)", e1c, e1c < 1e-8, "{:.1e}")
c.item("6 x 6 Pfaffian of the composite channel = Householder Pfaffian", e1p, e1p < 1e-12, "{:.1e}")

worst2 = sab2 = 0.0
for fl in ("4", "2"):
    a = int(fl) - 1

    def mk(h, fl=fl):
        return build_model(lat4, 2.41, -0.01, 1.0, g10=0.05, h_sel=h, mask=flavour_mask(fl))

    mm = mk(0.1)
    bx = N1Measure(mm, exact=True, six_fermion=False).bilinears(F)
    d = 1e-3
    fd = (
        N1Measure(mk(0.1 + d), exact=True, six_fermion=False).bilinears(F)["phi_f"][a]
        - N1Measure(mk(0.1 - d), exact=True, six_fermion=False).bilinears(F)["phi_f"][a]
    ) / (2 * d)
    worst2 = max(worst2, abs(bx["chi_f_conn"][a] - fd) / abs(fd))
    A_ = mm.D.dense_G(F)[:, a, :, a]
    CA, AC = C0 @ A_, A_ @ C0
    sab = (0.25 * A_ * (CA @ C0) + 0.25 * CA * AC).sum() / V  # second exchange term with the wrong sign
    sab2 = max(sab2, abs(sab - fd) / abs(fd))
c.item("chi_f_conn = d<O^(a)>/dh_a (flavours 4 and 2, h = 0.1; relative)", worst2, worst2 < 1e-4, "{:.1e}")
c.item("sabotage (second exchange term sign flipped) off by", sab2, sab2 > 0.05, "{:.0%}")

ok3 = True
for L, h in ((4, 0.01), (4, 0.2), (6, 0.2)):
    latL = Lattice((L,) * 4)
    VL = latL.V
    mm = build_model(latL, 0.0, -0.01, 1.0, g10=0.1, h_sel=h, mask=MASK)
    f0 = np.zeros(mm.nfields)
    b = N1Measure(mm, exact=True).bilinears(f0)
    ps = ChannelMeasure(build_model(latL, 0.0, -0.01, 1.0, g10=0.1, h=h), exact=True).bilinears(f0)["phi_src"]
    m00 = build_model(latL, 0.0, -0.01, 1.0, g10=0.1)
    b0 = N1Measure(m00, exact=True, six_fermion=False).bilinears(f0)
    G = m00.D.dense_G(f0)[:, 0, :, 0]
    t_of = latL.coords.reshape(4, VL)[-1]
    CT = np.zeros(L)
    for x0 in range(VL):
        dt = (t_of - t_of[x0]) % L
        wrap = np.where(t_of < t_of[x0], -1.0, 1.0)
        np.add.at(CT, dt, -wrap * G[:, x0] ** 3)
    CT /= VL
    e_light = max(abs(b["phi_f"][:3]).max(), abs(b["chi_f_conn"][:3] - b0["chi_f_conn"][:3]).max(), abs(b["phi_T"]))
    e_heavy = abs(b["phi_f"][3] - ps / 4)
    e_ct = abs(b["C_T"] - CT).max()
    c.item(
        f"free L={L} h={h}: light untouched, phi_T = 0, heavy = phi_src/4 (phi_f[3]/h = {b['phi_f'][3] / h:.3f}); "
        "C_T = -sum wrap G^3",
        (e_light, e_heavy, e_ct),
        e_light < 1e-10 and e_heavy < 1e-10 and e_ct < 1e-12,
    )

m = build_model(lat4, 2.41, -0.01, 1.0, g10=0.05, h_sel=0.1, mask=MASK, cg_tol=1e-10)
F = np.asarray(m.start(np.random.default_rng(7), "hot+zero"))
bx = N1Measure(m, exact=True).bilinears(F)
keys = ["phi_abs", "phi_stag_abs", "O4", "phi_sq", "phi_stag_sq", "dimer_sq_par", "phi_f", "phi_f_even", "phi_f_odd"]
keys += ["chi_f", "chi_f_conn", "chi_f_pmin", "phi_f_corners", "chi_f_corners"]
vals = {k: [] for k in keys}
for sd in range(16):
    b = N1Measure(m, n_noise=4, seed=sd).bilinears(F)
    for k in keys:
        vals[k].append(np.asarray(b[k], float).reshape(-1))
pulls = []
for k in keys:
    v = np.asarray(vals[k])
    mu, err = v.mean(0), v.std(0, ddof=1) / 4.0
    ref = np.asarray(bx[k], float).reshape(-1)
    sel = err > 1e-12
    pulls += list((mu[sel] - ref[sel]) / err[sel])
pulls = np.asarray(pulls)
c.item(
    f"stochastic (4 noise x 16 seeds) vs dense on {len(pulls)} quantities: largest pull",
    abs(pulls).max(),
    abs(pulls).max() <= 4.0,
    "{:.2f}",
)
rms = float(np.sqrt(np.mean(pulls**2)))
c.item("rms pull", rms, rms <= 1.3, "{:.2f}")
x0 = (2, 1, 3, 0)
i0 = int(np.ravel_multi_index(x0, lat4.shape))
cs = N1Measure(m, exact=False, cg_tol=1e-11).correlator(F, x0=x0)
G = m.D.dense_G(F)
t_of = lat4.coords.reshape(4, V)[-1]
eta4 = np.asarray(lat4.eta[-1]).reshape(V)
Gf, Cf = np.zeros((4, 4)), np.zeros((4, 4))
dt = (t_of - t_of[i0]) % 4
wrap = np.where(t_of < t_of[i0], -1.0, 1.0)
for a in range(4):
    np.add.at(Gf[a], dt, wrap * eta4[i0] * G[:, a, i0, a])
    np.add.at(Cf[a], dt, (G[:, :, i0, a] ** 2).sum(1))
e4c = max(abs(cs["Gf_f"] - Gf).max(), abs(cs["Cf_f"] - Cf).max())
c.item("point-source Gf_f, Cf_f (CG) vs dense", e4c, e4c < 1e-8, "{:.1e}")
c.record(
    "dense phi_f, chi_f, phi_T",
    [np.round(bx["phi_f"], 4).tolist(), np.round(bx["chi_f"], 3).tolist(), round(bx["phi_T"], 5)],
)
c.done()
