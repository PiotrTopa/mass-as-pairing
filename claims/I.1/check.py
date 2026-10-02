"""I.1: the RHMC simulates the model's measure exactly.

(a) Free limit on 4^4 (y = 0): log det(D^dagger D) = 2 sum_p log(4 sum_mu sin^2 p_mu); the CG propagator D^-1 e_0 equals
    the momentum-space propagator [K box^-1](x, 0); the free on-site block vanishes (|phi| = O4 = 0); the Lanczos
    action phi^dagger (D^dagger D)^-1/2 phi, the Lanczos vector and the Zolotarev / multishift-CG solve agree with
    dense linear algebra at y = 0 and y = 1.
(b) RHMC on 4^4 at (y, kappa, lambda) = (1.7, -0.01, 1): forces = finite differences of the exact action;
    reversibility; <|dH|> ~ dtau^2; <exp(-dH)> = 1; acceptance.
(c) Brute force on 2^4 (all directions antiperiodic, kappa = 0, lambda = 1): exact-determinant single-site Metropolis
    vs RHMC at y = 1.2 (five observables) and at y = 0 against the exact single-site integral. The long runs are frozen
    in data/derived/k7/mc (scripts/derive_k7.py mc); a prefix of every run is regenerated here and must be
    bit-identical.
(d) 4^4 at kappa = -0.01: y = 1.0 (symmetric) vs y = 2.5 (SMG): O4 grows > 5x, sigma^2 grows.
"""

import importlib.util
import os
import pathlib
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import quad

from masspairing.action import build_model
from masspairing.claimcheck import Check
from masspairing.data import derived, load_chain
from masspairing.hmc import HMC
from masspairing.lattice import Lattice, to_numpy
from masspairing.measure import FermionMeasure, scalar_observables
from masspairing.rational import invsqrt_partial_fractions
from masspairing.solvers import lanczos, multishift_cg

spec = importlib.util.spec_from_file_location("derive_k7", ROOT / "scripts" / "derive_k7.py")
DK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DK)

c = Check("I.1")


def blocked(x, nb=20):
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


def free_logdet_AA(lat):
    """log det(D^dagger D) at y = 0: D^2 = box (x) 1_2 with eigenvalues -4 sum_mu sin^2 p_mu (twisted momenta)."""
    s2 = sum(np.sin(p) ** 2 for p in lat.momenta())
    return 2.0 * np.sum(np.log(4.0 * s2))


def free_propagator_column(lat, x0=(0, 0, 0, 0)):
    """[D^-1](x, x0) = [K box^-1](x, x0) at y = 0 over x; box^-1 from the twisted momentum sum at unwrapped
    coordinates (exp(i p_mu L) = bc_mu)."""
    pv = np.stack([p.reshape(-1) for p in lat.momenta()], 1)
    denom = -4.0 * np.sum(np.sin(pv) ** 2, 1)
    coords = np.stack([x.reshape(-1) for x in lat.coords], 1).astype(float)

    def B(xs):
        return (np.exp(1j * (xs - np.asarray(x0, float)) @ pv.T) / denom).sum(1) / lat.V

    G = np.zeros(lat.V, complex)
    eta = to_numpy(lat.eta).reshape(4, -1)
    for mu in range(4):
        e = np.zeros(4)
        e[mu] = 1
        G += eta[mu] * (B(coords + e) - B(coords - e))
    return G


# ---------------------------------------------------------------------------------------------- (a) free limit, 4^4
rng = np.random.default_rng(40)
lat = Lattice((4, 4, 4, 4))
f0 = np.zeros(3 * lat.V)
m0 = build_model(lat, 0.0, 0.0, 1.0, cg_tol=1e-12)
M0 = m0.D.dense(f0)
ev0 = np.linalg.eigvals(M0)
ld_dense, ld_exact = float(np.sum(np.log(np.abs(ev0) ** 2))), free_logdet_AA(lat)
c.item(
    "(a) y = 0: log det D^dagger D, dense vs analytic momentum sum",
    f"{ld_dense:.10f} vs {ld_exact:.10f}",
    abs(ld_dense - ld_exact) < 1e-8 * abs(ld_exact),
)
G = free_propagator_column(lat)
e = np.zeros((lat.V, 2), complex)
e[0, 0] = 1.0
z = m0.solve_Dinv(f0, e)
err = np.abs(z[:, 0] - G).max()
c.item(
    "(a) CG propagator D^-1 e_0 vs momentum-space propagator: max error (max |G|), off-doublet component",
    f"{err:.2e} ({np.abs(G).max():.4f}), {np.abs(z[:, 1]).max():.1e}",
    err < 1e-10 and np.abs(z[:, 1]).max() == 0.0,
)
b = FermionMeasure(m0, exact=True).bilinears(f0)
c.item(
    "(a) free on-site block: |phi|, O4", f"{b['phi_abs']:.1e}, {b['O4']:.1e}", b["phi_abs"] < 1e-12 and b["O4"] < 1e-12
)
sig1 = lat.random_sigma(rng).reshape(-1)
for y, fields in ((0.0, f0), (1.0, sig1)):
    D = build_model(lat, y, 0.0, 1.0).D
    M = D.dense(fields)
    Y = D.prepare(fields)
    v = lat.random_psi(rng)
    assert np.abs((M @ v.reshape(-1)).reshape(lat.V, 2) - D.apply(v, Y=Y)).max() < 1e-12
    w, U = np.linalg.eigh(M.conj().T @ M)
    Amh = (U * w**-0.5) @ U.conj().T
    vf = v.reshape(-1)
    A = lambda u: D.AA(u, Y=Y)
    r = lanczos(A, v, lambda t: t**-0.5, tol=1e-12, want_vector=True)
    e_val = abs(r["value"] - (vf.conj() @ Amh @ vf)) / abs(r["value"])
    e_vec = np.linalg.norm(r["vec"].reshape(-1) - Amh @ vf) / np.linalg.norm(vf)
    pf = invsqrt_partial_fractions(12, w.min() * 0.5, w.max() * 1.3)
    xs, it = multishift_cg(A, v, pf["poles"], tol=1e-12)
    approx = pf["c0"] * v + np.tensordot(pf["rho"], xs, axes=(0, 0))
    e_rat = np.linalg.norm(approx.reshape(-1) - Amh @ vf) / np.linalg.norm(vf)
    c.item(
        f"(a) y = {y}: Lanczos action rel. error, Lanczos vector error, rational (n = 12, {it} CG its) A^-1/2 v error;"
        " spec(D^dagger D)",
        f"{e_val:.1e}, {e_vec:.1e}, {e_rat:.1e}; [{w.min():.3f}, {w.max():.2f}]",
        e_val < 1e-10 and e_vec < 1e-7 and e_rat < 1e-9,
    )

# ---------------------------------------------------------------------------------------------- (b) RHMC on 4^4
t0 = time.time()
rng = np.random.default_rng(41)
m = build_model(lat, 1.7, -0.01, 1.0, cg_tol=1e-12, lanczos_tol=1e-12)
fields = lat.random_sigma(rng, 0.5).reshape(-1)
eps = 1e-5
eta = lat.random_psi(rng)
phi, info = m.heatbath_phi(fields, eta)
m.set_window(info["ritz_min"], info["ritz_max"])
f_pf, _ = m.f_pf(fields, phi)
f_b = m.f_boson(fields)
worst = 0.0
for _ in range(4):
    a, idx = int(rng.integers(3)), tuple(int(i) for i in rng.integers(4, size=4))
    j = a * lat.V + int(np.ravel_multi_index(idx, lat.shape))
    fp, fm_ = fields.copy(), fields.copy()
    fp[j] += eps
    fm_[j] -= eps
    fd_pf = -(m.s_pf(fp, phi) - m.s_pf(fm_, phi)) / (2 * eps)
    fd_b = -(m.s_boson(fp) - m.s_boson(fm_)) / (2 * eps)
    worst = max(worst, abs(fd_pf - f_pf[j]) / (abs(fd_pf) + 1e-3), abs(fd_b - f_b[j]) / (abs(fd_b) + 1e-3))
c.item(
    "(b) pseudofermion and boson forces vs central finite differences (eps = 1e-5): worst relative",
    worst,
    worst < 1e-5,
    "{:.1e}",
)
h = HMC(m, tau=1.0, nsteps=8, seed=41)
for tol, lim in ((1e-12, 1e-12), (1e-8, 1e-7)):
    m.cg_tol = tol
    r = h.reversibility(fields)
    c.item(
        f"(b) reversibility (8 steps) at cg_tol {tol:.0e}: |d sigma|/|sigma|, |d pi|/|pi|",
        f"{r['dsigma']:.1e}, {r['dpi']:.1e}",
        r["dsigma"] < lim and r["dpi"] < lim,
    )
m.cg_tol, m.lanczos_tol = 1e-8, 1e-10  # production settings
for i in range(30):
    fields, info = h.trajectory(fields, nsteps=16 if i < 10 else 10)
c.record(
    "(b) spec(D^dagger D) after 30 thermalisation trajectories", f"[{info['ritz_min']:.3f}, {info['ritz_max']:.1f}]"
)
res = {}
for n in (6, 12, 24):
    vals = [abs(h.trajectory(fields, nsteps=n, force_accept=False)[1]["dH"]) for _ in range(10)]
    res[n] = (np.mean(vals), np.std(vals) / np.sqrt(len(vals)))
    c.record(f"(b) <|dH|> at {n} steps (10 trajectories)", f"{res[n][0]:.4f} +- {res[n][1]:.4f}")
r1, r2 = res[6][0] / res[12][0], res[12][0] / res[24][0]
c.item(
    "(b) <|dH|> ratios 6 -> 12 and 12 -> 24 steps (4 for dtau^2; accepted 2.5-7)",
    f"{r1:.2f}, {r2:.2f}",
    2.5 < r1 < 7.0 and 2.5 < r2 < 7.0,
)
dHs, acc = [], []
for _ in range(40):
    fields, info = h.trajectory(fields, nsteps=10)
    dHs.append(info["dH"])
    acc.append(info["accepted"])
ex = np.exp(-np.array(dHs))
c.item(
    "(b) 40 trajectories at 10 steps: acceptance, <exp(-dH)>",
    f"{np.mean(acc):.2f}, {ex.mean():.3f} +- {ex.std() / np.sqrt(len(ex)):.3f}",
    np.mean(acc) > 0.7 and abs(ex.mean() - 1.0) < 4 * ex.std() / np.sqrt(len(ex)) + 0.05,
)
c.record("(b) runtime", f"{time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------- (c) brute force, 2^4
def frozen(name):
    s, meta = load_chain(derived("k7", "mc", f"{name}.npz"))
    return {k[3:]: v for k, v in s.items()}, meta


t0 = time.time()
for name in ("metropolis_y0", "rhmc_y0", "metropolis_y1.2", "rhmc_y1.2"):
    s, _ = frozen(name)
    live = DK.mc_run(name, DK.prefix_length(name))
    n = len(live["sigma2"])
    same = all(np.array_equal(live[k], s[k][:n]) for k in live)
    c.item(
        f"(c) {name}: live prefix ({DK.prefix_length(name)} sweeps/trajectories) bit-identical to the frozen run",
        n,
        same,
    )
c.record("(c) prefix runtime", f"{time.time() - t0:.0f} s")
Z = quad(lambda r: r**2 * np.exp(-(r**2) / 2 - r**4 / 4), 0, np.inf)[0]
s2_exact = quad(lambda r: r**4 * np.exp(-(r**2) / 2 - r**4 / 4), 0, np.inf)[0] / Z
sm, sh = frozen("metropolis_y0")[0], frozen("rhmc_y0")[0]
vm, em = blocked(sm["sigma2"])
vh, eh = blocked(sh["sigma2"])
c.item(
    "(c) control y = 0: <|sigma|^2> exact single-site integral | Metropolis 4000 sweeps | RHMC 600 trajectories",
    f"{s2_exact:.5f} | {vm:.5f}({em:.5f}) | {vh:.5f}({eh:.5f})",
    abs(vm - s2_exact) < 4 * em + 1e-3 and abs(vh - s2_exact) < 4 * eh + 1e-3,
)
sm, sh = frozen("metropolis_y1.2")[0], frozen("rhmc_y1.2")[0]
c.record("(c) y = 1.2: RHMC acceptance (6 Omelyan steps)", f"{sh['accepted'].mean():.2f}")
for k in DK.KEYS:
    vm, em = blocked(sm[k])
    vh, eh = blocked(sh[k])
    zz = (vm - vh) / np.sqrt(em**2 + eh**2)
    c.item(
        f"(c) y = 1.2 {k}: Metropolis (40000 sweeps) | RHMC (2500 trajectories) | pull",
        f"{vm:.6f}({em:.6f}) | {vh:.6f}({eh:.6f}) | {zz:+.2f}",
        abs(zz) < 3.5,
    )


# ---------------------------------------------------------------------------------------------- (d) phases on 4^4
def phase_run(y, seed, ntherm=30, ntraj=80):
    m = build_model(lat, y, -0.01, 1.0)
    h = HMC(m, tau=1.0, nsteps=10, seed=seed)
    fm = FermionMeasure(m, exact=True)
    fields = lat.random_sigma(h.rng, 0.5).reshape(-1)
    keys = ("sigma2", "Sigma_abs", "Sigma_stag_abs", "O4", "phi_sq", "phi_stag_sq", "phi_abs", "phi_stag_abs")
    ser = {k: [] for k in keys}
    acc = []
    for it in range(ntherm + ntraj):
        fields, info = h.trajectory(fields, nsteps=20 if it < ntherm // 2 else None)
        if it < ntherm:
            continue
        acc.append(info["accepted"])
        so = scalar_observables(lat, m.split(fields)[0])
        fo = fm.bilinears(fields)
        for k in keys:
            ser[k].append(so[k] if k in so else fo[k])
    out = {k: blocked(v, nb=10) for k, v in ser.items()}
    out["acc"] = float(np.mean(acc))
    return out


t0 = time.time()
ph = {y: phase_run(y, seed) for y, seed in ((1.0, 431), (2.5, 432))}
for y, r in ph.items():
    c.record(
        f"(d) y = {y}: sigma^2, |Sigma|, |Sigma_stag|, O4, |phi|, |phi_stag|, V<|phi|^2>, V<|phi_stag|^2>, acceptance",
        ", ".join(
            f"{r[k][0]:.5f}({r[k][1]:.5f})"
            for k in ("sigma2", "Sigma_abs", "Sigma_stag_abs", "O4", "phi_abs", "phi_stag_abs")
        )
        + f", {lat.V * r['phi_sq'][0]:.3f}, {lat.V * r['phi_stag_sq'][0]:.3f}, {r['acc']:.2f}",
    )
o4s, o4w = ph[2.5]["O4"][0], ph[1.0]["O4"][0]
c.item("(d) O4(y = 2.5) / O4(y = 1.0) > 5", o4s / o4w, o4s > 5 * o4w, "{:.1f}")
c.item(
    "(d) sigma^2(y = 2.5) > sigma^2(y = 1.0)",
    f"{ph[2.5]['sigma2'][0]:.4f} > {ph[1.0]['sigma2'][0]:.4f}",
    ph[2.5]["sigma2"][0] > ph[1.0]["sigma2"][0],
)
c.record("(d) runtime", f"{time.time() - t0:.0f} s")
c.done()
