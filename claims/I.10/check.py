"""I.10: the N1 instrument -- the flavour-blind taste-chiral mass h C_chi in the RHMC -- is exact; the phase-flipped
pattern is rejected.

(1) operator (4^4, production and all-antiperiodic bc, epsilon model and wedge g10 = 0.1): D_c(h) = D_c(0) - h C_chi
    (x) 1_2 (C_chi: 8 entries per row, merged with K into 16 neighbours), D^dagger = -D; the merged CSR applies D and
    D^dagger as the dense matrix; the flavour-selective path with the all-flavour mask is the same operator; det D_c > 0
    on hot configurations and on the stored 4^4 P_c configuration (h = 1).
(2) 4^4 wedge (0.25, 0.05), h = 0.3 chiral: Lanczos S_pf = dense; forces vs finite differences; reversibility.
(3) 4^4 all-antiperiodic, epsilon model (2, -0.01, 1), h = 0.3: exact-determinant Metropolis (Woodbury) vs RHMC on 20 N1
    observables within 3 sigma; the RHMC with the phase-flipped pattern rejected (> 5 sigma). Frozen chains of
    scripts/derive_n1inst.py i10; live: the first sweeps / trajectories are regenerated bit-identically and the last
    measured configuration of every chain is re-measured.
About 3 min.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.action import build_model
from masspairing.analysis import n1inst as X
from masspairing.claimcheck import Check
from masspairing.data import derived
from masspairing.flavour import realify
from masspairing.hmc import HMC
from masspairing.lattice import Lattice
from masspairing.measure import N1Measure
from masspairing.patterns import chiral_mass

c = Check("I.10")

# ---- (1) operator
rng = np.random.default_rng(161)
e_op = e_ap = e_m = 0.0
lds = []
nn_ok = True
pos = True
for bc in ((1, 1, 1, -1), (-1, -1, -1, -1)):
    lat = Lattice((4, 4, 4, 4), bc=bc)
    V = lat.V
    C = chiral_mass(lat)
    for g10 in (0.0, 0.1):
        m = build_model(lat, 2.41, -0.01, 1.0, g10=g10, h=0.3, pattern="chiral")
        m0 = build_model(lat, 2.41, -0.01, 1.0, g10=g10)
        mS = build_model(lat, 2.41, -0.01, 1.0, g10=g10, h_sel=0.3, mask=(1, 1, 1, 1), pattern="chiral")
        t = m.D.merged_tables()
        nn_ok &= bool(np.all(np.diff(t["indptr"]) == 16) and np.all(np.diff(C.indptr) == 8))
        for _ in range(3):
            F = np.asarray(m.start(rng, "hot+hot"))
            D = m.D.dense(F)
            e_op = max(
                e_op,
                abs(D - (m0.D.dense(F) - 0.3 * np.kron(C.toarray(), np.eye(2)))).max(),
                abs(D + D.conj().T).max(),
            )
            psi = lat.random_psi(rng)
            B = lat.random_psi(rng, n=3).transpose(1, 0, 2)
            Y = m.D.prepare(F)
            e_ap = max(
                e_ap,
                abs(m.D.apply(psi, Y=Y).reshape(-1) - D @ psi.reshape(-1)).max(),
                abs(m.D.apply_dag(psi, Y=Y).reshape(-1) + D @ psi.reshape(-1)).max(),
                max(abs(m.D.apply(B, Y=Y)[:, i, :].reshape(-1) - D @ B[:, i, :].reshape(-1)).max() for i in range(3)),
            )
            e_m = max(
                e_m,
                abs(mS.D.apply(psi, Y=mS.D.prepare(F)) - m.D.apply(psi, Y=Y)).max(),
                abs(mS.D.dense_real4(F) - realify(D, V)).max(),
            )
            sign, ld = np.linalg.slogdet(D)
            pos &= abs(sign - 1.0) < 1e-10
            lds.append(ld)
lat, meta, f = X.stored_config("L4_y2.41_eps_pppa")
ms = build_model(lat, meta["y"], meta["kappa"], 1.0, h=1.0, pattern="chiral")
sign, ld = np.linalg.slogdet(ms.D.dense(f[: 3 * lat.V]))
pos &= abs(sign - 1.0) < 1e-10
lds.append(ld)
c.item("C_chi has 8 entries per row; merged with K: 16 neighbours", nn_ok, nn_ok)
c.item("D_c(h) = D_c(0) - h C_chi (x) 1_2 and D^dagger = -D", e_op, e_op < 1e-13, "{:.1e}")
c.item("merged CSR D, D^dagger (single, batch) vs dense", e_ap, e_ap < 1e-12, "{:.1e}")
c.item("flavour-selective path with the all-flavour mask = the same operator", e_m, e_m < 1e-12, "{:.1e}")
c.item(
    "det D_c > 0 on 12 hot configurations and the stored 4^4 P_c one (log det range)",
    f"{min(lds):.1f} ... {max(lds):.1f}",
    pos,
)

# ---- (2) action, forces, reversibility
lat4 = Lattice((4, 4, 4, 4))
m = build_model(
    lat4, 1.7, -0.01, 1.0, g10=0.25, g6=0.05, h=0.3, pattern="chiral", cg_tol=1e-12, lanczos_tol=1e-12, n_rational=20
)
F = m.start(rng, "hot")
phi, _ = m.heatbath_phi(F, lat4.random_psi(rng))
Dd = m.D.dense(F)
w, U = np.linalg.eigh(Dd.conj().T @ Dd)
m.set_window(w.min(), w.max())
ph = phi.reshape(-1)
s_dense = float((ph.conj() @ (U @ ((U.conj().T @ ph) / np.sqrt(w)))).real)
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
rv = HMC(m, tau=1.0, nsteps=8, seed=161).reversibility(F)
c.item("4^4 wedge (0.25, 0.05), h = 0.3 chiral: Lanczos S_pf vs dense", e_act, e_act < 1e-9, "{:.1e}")
c.item("sigma- and s-forces vs 4-point finite differences (6 components)", worst, worst < 1e-5, "{:.1e}")
c.item("reversibility |d fields|, |d pi|", (rv["dsigma"], rv["dpi"]), rv["dsigma"] < 1e-12 and rv["dpi"] < 1e-11)

# ---- (3) Metropolis vs RHMC
part = {}
for name in ("met0", "met1", "rhmc", "flipped"):
    z = np.load(derived("n1inst", f"I10_{name}.npz"))
    part[name] = {k.split("__", 1)[1]: z[k] for k in z.files}
live = X.i10_metropolis(20, X.I10_MET[0][1], record_trace=True, measure=False)
e_l = abs(live["trace_sb"] - part["met0"]["trace_sb"][:20]).max()
c.item("live Metropolis prefix (20 sweeps, boson action after every sweep) = frozen chain", e_l, e_l == 0.0, "{:.0e}")
for name, (_, seed, flip) in zip(("rhmc", "flipped"), X.I10_RHMC, strict=True):
    lv = X.i10_rhmc(0, seed, flip, ntherm=10, measure=False)
    e_r = abs(lv["dH"] - part[name]["dH"][:10]).max()
    c.item(f"live {name} prefix (10 trajectories, dH) = frozen chain", e_r, e_r == 0.0, "{:.0e}")
lat = Lattice(X.I10["shape"], bc=X.I10["bc"])
e_rm = 0.0
for name in ("met0", "met1", "rhmc", "flipped"):
    model = X.i10_model(flipped=name == "flipped")
    row = X.taste_row(lat, model, N1Measure(model, exact=True, taste=True), part[name]["fields_last"])
    e_rm = max([e_rm] + [abs(row[k] - part[name][k][-1]) for k in X.TASTE_KEYS])
c.item("re-measurement of the last measured configuration of each chain = stored row", e_rm, e_rm < 1e-12, "{:.0e}")

met = {k: np.concatenate([part["met0"][k], part["met1"][k]]) for k in X.TASTE_KEYS}
sh, sx = part["rhmc"], part["flipped"]
c.record(
    "Metropolis acc / max |Im det ratio|; RHMC acc; flipped acc",
    [float(part["met0"]["acc"]), max(float(part["met0"]["worst_im"]), float(part["met1"]["worst_im"]))]
    + [float(sh["acc"]), float(sx["acc"])],
)
c.item(
    "Woodbury det ratios real (max relative imaginary part)",
    max(float(part["met0"]["worst_im"]), float(part["met1"]["worst_im"])),
    max(float(part["met0"]["worst_im"]), float(part["met1"]["worst_im"])) < 1e-8,
    "{:.1e}",
)
nmeas = [len(part["met0"]["O4"]), len(part["met1"]["O4"]), len(sh["O4"]), len(sx["O4"])]
n_met = len(range(304, 5000, 8))
c.item(f"measurements: Metropolis 2 x {n_met}, RHMC 2000, flipped 500", nmeas, nmeas == [n_met, n_met, 2000, 500])
worst = worst_x = 0.0
nontrivial = True
for k in X.TASTE_KEYS:
    vm, em = X.blocked(met[k])
    vh, eh = X.blocked(sh[k])
    vx, ex = X.blocked(sx[k])
    nontrivial &= em > 0 and eh > 0
    p, px = X.pull(vm, em, vh, eh), X.pull(vm, em, vx, ex)
    worst, worst_x = max(worst, abs(p)), max(worst_x, abs(px))
    c.record(
        f"{k:14s} Metropolis / RHMC / flipped",
        f"{vm:+.5f}({em:.5f}) {vh:+.5f}({eh:.5f}) pull {p:+.2f} | {vx:+.5f}({ex:.5f}) pull {px:+.1f}",
    )
c.item("every observable has variance in both chains (non-trivial on this box)", nontrivial, nontrivial)
c.item("largest Metropolis-RHMC pull over 20 observables (<= 3)", worst, worst <= 3.0, "{:.2f}")
c.item("phase-flipped pattern rejected: largest pull (> 5)", worst_x, worst_x > 5.0, "{:.1f}")
c.done()
