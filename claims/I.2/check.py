"""I.2: the Hasenbusch-split, multiple-time-scale RHMC (HasenbuschModel + MTSHMC) simulates the same measure as the
plain RHMC.

(A) 4^4, (y, kappa) = (2.41, -0.01), random sigma, shifts (0.05, 0.5): every term's heat-bath vector and Lanczos action
    agree with the dense eigen-decomposition; sum_k log det g_k(A) = -1/2 log det A; each term's force equals central
    finite differences of its action; the nested integrator (4, 2, 2) is reversible.
(B) dH ~ dtau^2 across the levels on a thermalised 4^4 configuration at (1.7, -0.01), shifts (0.1, 1); sabotage: the
    lightest term's force poles x 3 destroy the scaling.
(C) Brute force on 2^4 (all antiperiodic, kappa = 0, lambda = 1): exact-determinant Metropolis vs Hasenbusch/MTS RHMC
    (shifts (0.3, 3), levels (3, 1, 1)) at y = 1.2 and 2.5; sabotage at y = 2.5: the heavy term alone,
    det(D^dagger D + 3)^(1/2), is rejected. Frozen runs in data/derived/k7/mc (scripts/derive_k7.py mc), live prefixes
    bit-identical.
(D) Stored 4^4 chains at P_c = (2.41, -0.01): plain RHMC (20 steps) and Hasenbusch (0.01, 0.1) with levels (5, 2),
    the lightest ratio term on the fine level, agree on five observables.
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

from masspairing.action import build_model
from masspairing.claimcheck import Check
from masspairing.data import derived, load_chain
from masspairing.hmc import MTSHMC
from masspairing.lattice import Lattice

spec = importlib.util.spec_from_file_location("derive_k7", ROOT / "scripts" / "derive_k7.py")
DK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DK)

c = Check("I.2")


def blocked(x, nb=20):
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


# ---------------------------------------------------------------------------------------------- (A) dense checks, 4^4
t0 = time.time()
rng = np.random.default_rng(60)
lat = Lattice((4, 4, 4, 4))
m = build_model(lat, 2.41, -0.01, 1.0, hasenbusch=[0.05, 0.5], cg_tol=1e-12, lanczos_tol=1e-12)
fields = lat.random_sigma(rng, 0.5).reshape(-1)
Ad = m.D.dense(fields)
Ad = Ad.conj().T @ Ad
w, U = np.linalg.eigh(Ad)
c.record("(A) terms; spec(D^dagger D)", f"{m.terms}; [{w.min():.4f}, {w.max():.2f}]")
phis, logdet_terms = [], 0.0
for k in range(m.nterms):
    eta = lat.random_psi(rng)
    phi, info = m.heatbath_term(k, fields, eta)
    phis.append(phi)
    fh, fa = m.term_function(k, "heatbath"), m.term_function(k, "action")
    ex = (U @ (fh(w)[:, None] * (U.conj().T @ eta.reshape(-1, 1)))).reshape(eta.shape)
    e_vec = np.linalg.norm(phi - ex) / np.linalg.norm(ex)
    s_ex = np.vdot(phi.reshape(-1), (U @ (fa(w)[:, None] * (U.conj().T @ phi.reshape(-1, 1)))).reshape(-1)).real
    e_act = abs(m.s_term(k, fields, phi) - s_ex) / abs(s_ex)
    logdet_terms += np.sum(np.log(fa(w)))
    c.item(
        f"(A) term {k}: heat-bath vector vs dense, Lanczos action vs dense (relative)",
        f"{e_vec:.1e}, {e_act:.1e}",
        e_vec < 1e-7 and e_act < 1e-11,
    )
ld = -0.5 * np.sum(np.log(w))
c.item(
    "(A) sum_k log det g_k(A) + 1/2 log det(D^dagger D)",
    abs(logdet_terms - ld),
    abs(logdet_terms - ld) < 1e-8,
    "{:.1e}",
)
m.set_window(w.min(), w.max())
eps = 1e-5
for k in range(m.nterms):
    fk, it = m.f_term(k, fields, phis[k])
    worst = 0.0
    for _ in range(3):
        a, idx = int(rng.integers(3)), tuple(int(i) for i in rng.integers(4, size=4))
        j = a * lat.V + int(np.ravel_multi_index(idx, lat.shape))
        fp, fm_ = fields.copy(), fields.copy()
        fp[j] += eps
        fm_[j] -= eps
        fd = -(m.s_term(k, fp, phis[k]) - m.s_term(k, fm_, phis[k])) / (2 * eps)
        worst = max(worst, abs(fd - fk[j]) / (abs(fd) + 1e-3))
    c.item(f"(A) term {k} force vs finite differences (eps = 1e-5): worst relative", worst, worst < 1e-5, "{:.1e}")
h = MTSHMC(m, tau=1.0, levels=(4, 2, 2), seed=60)
for tol, lim in ((1e-12, 1e-12), (1e-8, 1e-7)):
    m.cg_tol = tol
    r = h.reversibility(fields)
    c.item(
        f"(A) reversibility of levels (4, 2, 2) at cg_tol {tol:.0e}: |d sigma|/|sigma|, |d pi|/|pi|",
        f"{r['dsigma']:.1e}, {r['dpi']:.1e}",
        r["dsigma"] < lim and r["dpi"] < lim,
    )
c.record("(A) runtime", f"{time.time() - t0:.0f} s")

# ---------------------------------------------------------------------------------------------- (B) dH ~ dtau^2
t0 = time.time()
m = build_model(lat, 1.7, -0.01, 1.0, hasenbusch=[0.1, 1.0], cg_tol=1e-8, lanczos_tol=1e-10)
h = MTSHMC(m, tau=1.0, levels=(6, 1, 1), seed=61)
fields = lat.random_sigma(h.rng, 0.5).reshape(-1)
for i in range(20):
    fields, info = h.trajectory(fields, nsteps=10 if i < 8 else 6)


def mean_dH(hh, ntraj=6):
    return float(np.mean([abs(hh.trajectory(fields, force_accept=False)[1]["dH"]) for _ in range(ntraj)]))


res = {}
for lv in ((4, 1, 1), (8, 1, 1), (16, 1, 1), (4, 1, 2)):
    res[lv] = mean_dH(MTSHMC(m, tau=1.0, levels=lv, seed=62))
    c.record(f"(B) <|dH|> at levels {lv} (6 trajectories)", f"{res[lv]:.4f}")
r1, r2 = res[(4, 1, 1)] / res[(8, 1, 1)], res[(8, 1, 1)] / res[(16, 1, 1)]
c.item(
    "(B) <|dH|> ratios 4 -> 8 and 8 -> 16 outer steps (4 for dtau^2; accepted 2.5-9); finer innermost level does not"
    " raise <|dH|>",
    f"{r1:.2f}, {r2:.2f}; {res[(4, 1, 1)]:.4f} -> {res[(4, 1, 2)]:.4f}",
    2.5 < r1 < 9.0 and 2.5 < r2 < 9.0 and res[(4, 1, 2)] < 1.5 * res[(4, 1, 1)],
)
set_window = m.set_window


def bad_window(lo, hi):  # sabotage: wrong poles in the lightest force, which is then not the gradient of its action
    pfs = set_window(lo, hi)
    pfs[0]["poles"] = pfs[0]["poles"] * 3.0
    return pfs


m.set_window = bad_window
bad = mean_dH(MTSHMC(m, tau=1.0, levels=(16, 1, 1), seed=62))
del m.set_window
c.item(
    "(B) sabotage (lightest-term poles x 3) at levels (16, 1, 1): <|dH|> vs correct, rejected if > 10x",
    f"{bad:.4f} vs {res[(16, 1, 1)]:.4f}",
    bad > 10 * res[(16, 1, 1)],
)
c.record("(B) runtime", f"{time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------- (C) brute force, 2^4
def frozen(name):
    s, _ = load_chain(derived("k7", "mc", f"{name}.npz"))
    return {k[3:]: v for k, v in s.items()}


t0 = time.time()
for name in ("hasenbusch_y1.2", "metropolis_y2.5", "hasenbusch_y2.5", "sabotage_heavy_only_y2.5"):
    s = frozen(name)
    live = DK.mc_run(name, DK.prefix_length(name))
    n = len(live["sigma2"])
    same = all(np.array_equal(live[k], s[k][:n]) for k in live)
    c.item(
        f"(C) {name}: live prefix ({DK.prefix_length(name)} sweeps/trajectories) bit-identical to the frozen run",
        n,
        same,
    )
c.record("(C) prefix runtime", f"{time.time() - t0:.0f} s")
for y, ref, run, label in (
    (1.2, "metropolis_y1.2", "hasenbusch_y1.2", "40000 sweeps | 2500 trajectories"),
    (2.5, "metropolis_y2.5", "hasenbusch_y2.5", "20000 sweeps | 1500 trajectories"),
):
    sm, sh = frozen(ref), frozen(run)
    c.record(f"(C) y = {y}: Hasenbusch/MTS acceptance", f"{sh['accepted'].mean():.2f}")
    for k in DK.KEYS:
        vm, em = blocked(sm[k])
        vh, eh = blocked(sh[k])
        zz = (vm - vh) / np.sqrt(em**2 + eh**2)
        c.item(
            f"(C) y = {y} {k}: Metropolis | Hasenbusch/MTS RHMC ({label}) | pull",
            f"{vm:.6f}({em:.6f}) | {vh:.6f}({eh:.6f}) | {zz:+.2f}",
            abs(zz) < 3.5,
        )
sm, sx = frozen("metropolis_y2.5"), frozen("sabotage_heavy_only_y2.5")
pulls = {}
for k in DK.KEYS:
    vm, em = blocked(sm[k])
    vx, ex = blocked(sx[k])
    pulls[k] = (vm - vx) / np.sqrt(em**2 + ex**2)
    c.record(
        f"(C) sabotage y = 2.5 {k}: heavy term only (1000 trajectories, acc {sx['accepted'].mean():.2f})",
        f"{vx:.6f}({ex:.6f}), pull {pulls[k]:+.1f}",
    )
worst = max(abs(v) for v in pulls.values())
c.item("(C) sabotage (ratio terms dropped, y = 2.5): largest pull, rejected if > 5", worst, worst > 5, "{:.1f}")

# ---------------------------------------------------------------------------------------------- (D) stored 4^4 chains
P, Pm = load_chain(derived("k7", "c060", "plain.npz"))
H, Hm = load_chain(derived("k7", "c060", "hasenbusch.npz"))
c.record(
    "(D) plain: steps, trajectories, acceptance, <|dH|>",
    f"{Pm['nsteps']}, {len(P['ts_dH'])}, {P['ts_accepted'].mean():.2f}, {np.abs(P['ts_dH']).mean():.3f}",
)
c.record(
    "(D) Hasenbusch: shifts, levels, assignment, trajectories, acceptance, <|dH|>",
    f"{Hm['hasenbusch']}, {Hm['mts']}, {Hm['mts_assign']}, {len(H['ts_dH'])}, {H['ts_accepted'].mean():.2f},"
    f" {np.abs(H['ts_dH']).mean():.3f}",
)
for k in ("sigma2", "Sigma_abs", "Sigma_stag_abs", "O4", "phi_stag_sq"):
    vp, ep = blocked(P["ts_" + k])
    vh, eh = blocked(H["ts_" + k])
    zz = (vp - vh) / np.sqrt(ep**2 + eh**2)
    c.item(f"(D) {k}: plain | Hasenbusch | pull", f"{vp:.5f}({ep:.5f}) | {vh:.5f}({eh:.5f}) | {zz:+.2f}", abs(zz) < 3.5)
c.done()
