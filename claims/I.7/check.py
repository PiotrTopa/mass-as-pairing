"""I.7: the same-parity pairing source is sign-free, its RHMC is exact, and <Phi_src> responds linearly.

Weight e^{h O_h}, O_h = sum_planes [sum_{x even} w sum_a chi^a(x) chi^a(x + delta) + sum_{y odd} w sum_a chi^a(y)
chi^a(y + delta')] (delta = mu +- nu, delta' = -mu +- nu, w the fermionic boundary sign), i.e. K -> K - h C0 in D_c with
C0 real antisymmetric, same-parity, flavour-blind (the nodal pairing field).
(1) 4^4 operator: C0 antisymmetric, same-parity, 24 entries +-1 per row; D_c(h) = D_c(0) - h C0 (x) 1_2, anti-hermitian;
    sparse application = dense; det D_c real positive on 10 hot configurations at h = 0.3; log det invariant under a
    two-site time translation of sigma.
(2) 4^4, h = 0.1: forces vs 4-point finite differences, reversibility.
(3) 2^3 x 4 (all antiperiodic; on 2^4 the source vanishes identically), (y, kappa, lambda) = (2, 0, 1), g10 = 0.4,
    g6 = 0.1, h = 0.1: exact-determinant Metropolis vs RHMC (trajectory-length jitter 0.3), 13 observables within
    3 sigma; the control with the odd-sublattice half of C0 sign-flipped is rejected (> 5 sigma).
(4) Linear response (Metropolis, h = 0, 0.05, 0.1, 0.2): <Phi_src>_0 = 0; <Phi_src>/h at h = 0.05 and 0.1 agree with
    each other and with the h = 0 susceptibility <phi_src_dh> + V var(Phi_src), within 3 sigma.
The chains of (3)-(4) are frozen in data/derived/k3/chains/ (scripts/derive_k3.py chains); a prefix of every chain is
regenerated here and must be bit-identical.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.action import build_model
from masspairing.analysis import k3_exact as kx
from masspairing.analysis.stats import jackknife
from masspairing.claimcheck import Check
from masspairing.data import DERIVED
from masspairing.hmc import HMC
from masspairing.lattice import Lattice
from masspairing.patterns import source_pattern

CH = DERIVED / "k3" / "chains"
H = 0.1
KEYS = (
    "sigma2",
    "O4",
    "Sigma_stag_abs",
    "phi_stag_sq",
    "s2",
    "E_bond",
    "dimer_sq_par",
    "chi10",
    "chi6",
    "E10",
    "phi_src",
    "phi_src_even",
    "phi_src_odd",
)
PREFIX = {
    "source_metropolis_h0": 260,
    "source_metropolis_h0.1": 260,
    "source_rhmc_h0.1": 20,
    "source_rhmc_h0.1_flip": 20,
}
c = Check("I.7")

# ---------------------------------------------------------------- (1) operator
lat4 = Lattice((4, 4, 4, 4))
V = lat4.V
C0 = source_pattern(lat4)
eps = lat4.eps_np.reshape(V)
co = C0.tocoo()
ok = (
    abs(C0 + C0.T).max() == 0
    and np.all(np.diff(C0.indptr) == 24)
    and set(np.unique(C0.data)) == {-1.0, 1.0}
    and np.all(eps[co.row] == eps[co.col])
)
c.item("C0 on 4^4: antisymmetric, same-parity, 24 entries +-1 per row", ok, ok)
rng = np.random.default_rng(112)
m = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1, g6=0.0, h=0.3)
m0 = build_model(lat4, 2.41, -0.01, 1.0, g10=0.1, g6=0.0)
e_op = e_ap = 0.0
lds, signs = [], []
for _ in range(10):
    F = np.asarray(m.start(rng, "hot+hot"))
    D = m.D.dense(F)
    e_op = max(e_op, abs(D - (m0.D.dense(F) - 0.3 * np.kron(C0.toarray(), np.eye(2)))).max(), abs(D + D.conj().T).max())
    psi = lat4.random_psi(rng)
    Y = m.D.prepare(F)
    e_ap = max(
        e_ap,
        abs(m.D.apply(psi, Y=Y).reshape(-1) - D @ psi.reshape(-1)).max(),
        abs(m.D.apply_dag(psi, Y=Y).reshape(-1) + D @ psi.reshape(-1)).max(),
    )
    s, ld = np.linalg.slogdet(D)
    lds.append(ld)
    signs.append(abs(s - 1.0))
c.item(
    "D_c(h) = D_c(0) - h C0 (x) 1_2 and D_c^dagger = -D_c; sparse D, D^dagger vs dense",
    (e_op, e_ap),
    e_op < 1e-13 and e_ap < 1e-12,
    fmt="{0[0]:.1e}; {0[1]:.1e}",
)
c.item(
    "det D_c > 0 on 10 hot configurations at h = 0.3 (log det range)",
    (min(lds), max(lds), max(signs)),
    max(signs) < 1e-10,
    fmt="{0[0]:.1f} .. {0[1]:.1f}, max |sign - 1| {0[2]:.1e}",
)
ms = build_model(lat4, 2.41, -0.01, 1.0, h=0.3)
F = np.asarray(ms.start(rng, "hot", 0.8))
sig = ms.split(F)[0]
d_ld = abs(
    np.linalg.slogdet(ms.D.dense(F))[1]
    - np.linalg.slogdet(ms.D.dense(ms.pack(np.roll(sig, 2, axis=4), np.zeros(0))))[1]
)
c.item("log det under a two-site time translation of sigma", d_ld, d_ld < 1e-9, fmt="{:.1e}")

# ---------------------------------------------------------------- (2) forces, reversibility
m = build_model(lat4, 1.7, -0.01, 1.0, g10=0.25, g6=0.05, h=0.1, cg_tol=1e-12, lanczos_tol=1e-12, n_rational=20)
F = m.start(rng, "hot")
phi, _ = m.heatbath_phi(F, lat4.random_psi(rng))
w = np.linalg.eigvalsh(m.D.dense(F).conj().T @ m.D.dense(F))
m.set_window(w.min(), w.max())
fpf, _ = m.f_pf(F, phi)
e_ = 1e-3


def fd4(fun, i):
    out = []
    for hh in (2 * e_, e_, -e_, -2 * e_):
        Fp = F.copy()
        Fp[i] += hh
        out.append(fun(Fp))
    return -(-out[0] + 8 * out[1] - 8 * out[2] + out[3]) / (12 * e_)


worst = 0.0
for i in list(rng.integers(0, m.nsig, size=2)) + list(rng.integers(m.nsig, m.nfields, size=3)):
    a = fd4(lambda X: m.s_pf(X, phi), i)
    worst = max(worst, abs(a - fpf[i]) / (abs(a) + 1e-3))
rv = HMC(m, tau=1.0, nsteps=8, seed=112).reversibility(F)
c.item(
    "4^4, h = 0.1: forces vs 4-point finite differences (worst relative); reversibility",
    (worst, rv["dsigma"], rv["dpi"]),
    worst < 1e-5 and rv["dsigma"] < 1e-12 and rv["dpi"] < 1e-11,
    fmt="{0[0]:.1e}; {0[1]:.1e} / {0[2]:.1e}",
)

# ---------------------------------------------------------------- frozen chains and their live prefixes
chains = {name: kx.load_chain(CH / f"{name}.npz") for name in kx.CHAINS if name.startswith("source_")}
for name, n in PREFIX.items():
    live = kx.run_chain(name, n)
    frozen = chains[name]
    same = all(np.array_equal(live[k], frozen[k][: len(live[k])]) for k in frozen)
    c.item(
        f"live prefix of {name} ({n} {'sweeps' if 'metropolis' in name else 'trajectories after 200'}) "
        "bit-identical to the frozen chain",
        same,
        same,
    )
met = {h: chains[f"source_metropolis_h{h:g}"] for h in (0.0, 0.05, 0.1, 0.2)}
rh, sx = chains["source_rhmc_h0.1"], chains["source_rhmc_h0.1_flip"]
worst = worst_sab = 0.0
for k in KEYS:
    vm, em = kx.blocked(met[H][k])
    vh, eh = kx.blocked(rh[k])
    vx, exx = kx.blocked(sx[k])
    z, zx = (vm - vh) / np.hypot(em, eh), (vm - vx) / np.hypot(em, exx)
    worst, worst_sab = max(worst, abs(z)), max(worst_sab, abs(zx))
    c.record(
        f"    {k}: Metropolis, RHMC, pull | control, pull",
        f"{vm:+.5f}({em:.5f}) {vh:+.5f}({eh:.5f}) {z:+.2f} | {vx:+.5f}({exx:.5f}) {zx:+.1f}",
    )
c.item(
    "(3) 2^3 x 4, h = 0.1: Metropolis (8000 sweeps) vs RHMC (2500 trajectories, acceptance "
    f"{np.mean(rh['accepted']):.2f}): largest pull over 13 observables (<= 3)",
    worst,
    worst <= 3.0,
    fmt="{:.2f}",
)
c.item(
    "(3) control (odd half of C0 sign-flipped in the operator): largest pull (> 5, rejected)",
    worst_sab,
    worst_sab > 5,
    fmt="{:.1f}",
)
Vs = 2 * 2 * 2 * 4
ph = {h: kx.blocked(met[h]["phi_src"]) for h in met}
a = np.stack([met[0.0]["phi_src"], met[0.0]["phi_src_dh"]], 1)
nb = len(a) // 20
chi, chie = jackknife(a[: nb * 20], lambda b: b[:, 1].mean() + Vs * b[:, 0].var(), blen=nb)
s1, e1 = ph[0.05][0] / 0.05, ph[0.05][1] / 0.05
s2, e2 = ph[H][0] / H, ph[H][1] / H
z0 = ph[0.0][0] / ph[0.0][1]
z12, z1c, z2c = (s1 - s2) / np.hypot(e1, e2), (s1 - chi) / np.hypot(e1, chie), (s2 - chi) / np.hypot(e2, chie)
c.record("    <Phi_src> at h = 0, 0.05, 0.1, 0.2", " ".join(f"{ph[h][0]:+.5f}({ph[h][1]:.5f})" for h in sorted(ph)))
c.item(
    "(4) <Phi_src>_0 = 0 (pull); <Phi_src>/h at h = 0.05, 0.1 [0.2]; h = 0 susceptibility; pulls (two slopes, "
    "slopes vs susceptibility) within 3",
    (z0, s1, e1, s2, e2, ph[0.2][0] / 0.2, ph[0.2][1] / 0.2, chi, chie, z12, z1c, z2c),
    abs(z0) < 3 and abs(z12) < 3 and abs(z1c) < 3 and abs(z2c) < 3 and s2 / e2 > 5,
    fmt="{0[0]:+.2f}; {0[1]:.4f}({0[2]:.4f}), {0[3]:.4f}({0[4]:.4f}) [{0[5]:.4f}({0[6]:.4f})]; {0[7]:.4f}({0[8]:.4f});"
    " {0[9]:+.2f}, {0[10]:+.2f}, {0[11]:+.2f}",
)
c.done()
