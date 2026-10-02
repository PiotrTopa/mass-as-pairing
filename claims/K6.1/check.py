"""K6.1: no frequency winding in the sign-free class; what the loop carries; its quantisation scale.

(1) the det winding N_det(p_vec = 0) of the corner block around the antiperiodic p_0 loop: every step Arg in {0, pi},
    det G_B is real, N_det is a count of sign changes (half turns), never a winding -- for the doublet operator on free,
    Dirac-massive, coherent AFM and white-noise backgrounds (L = 4, 6), for K - h C_chi and K - C_0 (L = 4, 6, 8) and
    for two stored 6^4 equilibrium configurations; controls: a synthetic unit winding and a synthetic real sign change.
(2) G_B(-p) = tau_2 G_B(p)* tau_2 and the spectral flip of H = i G_B on white-noise sigma (L = 4, 6).
(3) for a coherent Majorana mass m (AFM background, m = y s): f_M = m^2/(m^2 + 4 sin^2 p_1), N_odd = 32 * 4 sin^2 p_1 /
    (m^2 + 4 sin^2 p_1), p_1 = pi/L_t; free massless f_M = 0, N_odd = 32, alpha = -1.
(4) quantisation (N_odd within 1 of 0 or 32 and f_M within 0.03 of 0 or 1) needs m >= sqrt(31) 2 sin(pi/L_t), above the
    staggered bandwidth 4 at every L_t <= 8; no 0.37 <= m <= 4 quantises.
About 1 min.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.analysis.n1inst import stored_config
from masspairing.claimcheck import Check
from masspairing.corner import CornerBlocks, analyse, analyse_blocks, det_winding, doublet_operator, p0_loop, quantised
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.patterns import chiral_mass, source_pattern

c = Check("K6.1")


def loop_test(D, shape, bc, n_int, name):
    blocks, _, _ = CornerBlocks(D, shape, n_int).loop_blocks(bc)
    dw = det_winding(blocks)
    args = np.angle(np.roll(dw["dets"], -1) / dw["dets"])
    off = float(np.min(np.stack([np.abs(args), np.abs(np.abs(args) - np.pi)]), 0).max())
    n = dw["n_half_turns"]
    half = abs(dw["winding"] - 0.5 * round(2 * dw["winding"])) < 1e-6 and abs(dw["winding"]) <= n / 2 + 1e-6
    c.item(
        f"{name}: max step distance from {{0, pi}}, max |Im det|/|det|; half turns {n}, N_det {dw['winding']:+.2f}",
        (float(f"{off:.1e}"), float(f"{dw['max_im_ratio']:.1e}")),
        off < 1e-8 and dw["max_im_ratio"] < 1e-9 and half,
    )
    return n


rng = np.random.default_rng(180)
turns = {}
for L in (4, 6):
    lat = Lattice((L,) * 4)
    z = np.zeros((3,) + lat.shape)
    loop_test(doublet_operator(lat, 0.0, z), lat.shape, lat.bc, 2, f"L={L} free massless")
    loop_test(doublet_operator(lat, 0.0, z, mass=1.0), lat.shape, lat.bc, 2, f"L={L} free Dirac mass m = 1")
    s = np.zeros((3,) + lat.shape)
    s[2] = 0.4 * np.asarray(lat.eps)
    loop_test(doublet_operator(lat, 2.45, s), lat.shape, lat.bc, 2, f"L={L} coherent AFM sigma = 0.4 eps n, y = 2.45")
    noise = rng.normal(size=(3,) + lat.shape) * 0.8
    turns[L] = loop_test(doublet_operator(lat, 2.45, noise), lat.shape, lat.bc, 2, f"L={L} white-noise sigma, y = 2.45")
c.item(
    "white-noise backgrounds: half turns at L = 4, 6 (sign changes of a real det)",
    (turns[4], turns[6]),
    (turns[4], turns[6]) == (0, 4),
)
for L in (4, 6, 8):
    lat = Lattice((L,) * 4)
    K = kinetic_matrix(lat)
    C = chiral_mass(lat)
    for h in (0.5, 1.0, 3.0):
        loop_test((K - h * C).astype(complex).tocsc(), lat.shape, lat.bc, 1, f"L={L} K - {h} C_chi")
    loop_test((K - source_pattern(lat)).astype(complex).tocsc(), lat.shape, lat.bc, 1, f"L={L} K - C_0")
for name in ("L6_y2.028_eps_pppa", "L6_y2.792_eps_pppa"):
    lat, meta, sig = stored_config(name)
    loop_test(
        doublet_operator(lat, float(meta["y"]), sig), lat.shape, lat.bc, 2, f"stored 6^4 configuration y = {meta['y']}"
    )
loop, _ = p0_loop((4, 4, 4, 8), (1, 1, 1, -1))
dw = det_winding([np.diag([np.exp(1j * p)] + [1.0] * 31) for p in loop])
c.item(
    "control: synthetic e^(i p0) (+) 1 gives N_det = 1",
    dw["winding"],
    abs(dw["winding"] - 1) < 1e-12 and dw["n_half_turns"] == 0,
    "{:.6f}",
)
dw = det_winding([np.diag([np.cos(p)] + [1.0] * 31) for p in loop])
c.item("control: synthetic real cos p0 (+) 1 gives two half turns", dw["n_half_turns"], dw["n_half_turns"] == 2)
r = analyse_blocks([np.diag([np.exp(1j * p)] + [1.0] * 31) for p in loop], loop)["all"]
c.item("analyse_blocks reports the same winding", r["det_winding"], abs(r["det_winding"] - 1) < 1e-12, "{:.6f}")

# ---- (2) tau_2 K relation
rng = np.random.default_rng(181)
T2 = np.kron(np.eye(16), np.array([[0, -1j], [1j, 0]]))
for L in (4, 6):
    lat = Lattice((L,) * 4)
    cb = CornerBlocks(doublet_operator(lat, 2.45, rng.normal(size=(3,) + lat.shape) * 0.8), lat.shape, 2)
    loop, base = p0_loop(lat.shape, lat.bc)
    Gp, Gm = cb.block(np.concatenate([base, [loop[0]]])), cb.block(np.concatenate([base, [loop[-1]]]))
    dev = np.abs(Gm - T2 @ Gp.conj() @ T2).max() / np.abs(Gp).max()
    wp, wm = np.sort(np.linalg.eigvalsh(1j * Gp)), np.sort(np.linalg.eigvalsh(1j * Gm))
    c.item(
        f"L={L}: G_B(-p) = tau_2 G_B(p)* tau_2; spectrum of H(-p) = -spectrum of H(p)",
        dev,
        dev < 1e-10 and np.abs(wp + wm[::-1]).max() < 1e-10,
        "{:.1e}",
    )

# ---- (3) formulas, (4) quantisation
table = {}
for L in (4, 6, 8):
    lat = Lattice((L,) * 4)
    p1 = np.pi / L
    r = analyse(CornerBlocks(doublet_operator(lat, 0.0, np.zeros((3,) + lat.shape)), lat.shape, 2), lat.bc)["all"]
    a_ok = L == 4 or abs(r["alpha"] + 1) < 1e-10
    c.item(
        f"L={L} free massless: f_M, N_odd, alpha",
        (float(f"{r['f_M']:.1e}"), round(r["N_odd"], 3), round(r["alpha"], 3)),
        r["f_M"] < 1e-14 and abs(r["N_odd"] - 32) < 1e-10 and a_ok,
    )
    table[(L, 0.0)] = quantised(r, 32)
    for m in (0.37, 0.98, 2.0, 4.0, 8.0):
        sig = np.zeros((3,) + lat.shape)
        sig[2] = m / 2.45 * np.asarray(lat.eps)
        r = analyse(CornerBlocks(doublet_operator(lat, 2.45, sig), lat.shape, 2), lat.bc)["all"]
        fp = m**2 / (m**2 + 4 * np.sin(p1) ** 2)
        npred = 32 * 4 * np.sin(p1) ** 2 / (m**2 + 4 * np.sin(p1) ** 2)
        c.item(
            f"L={L} AFM m={m}: f_M = {r['f_M']:.5f} ({fp:.5f}), N_odd = {r['N_odd']:.4f} ({npred:.4f}), alpha",
            round(r["alpha"], 3),
            abs(r["f_M"] - fp) < 1e-8 and abs(r["N_odd"] - npred) < 1e-6 and r["herm_defect"] < 1e-10,
        )
        table[(L, m)] = quantised(r, 32)
c.record("quantised (L_t, m)", sorted(k for k, q in table.items() if q))
c.item(
    "no Majorana mass 0.37 <= m <= 4 quantises at any L_t <= 8",
    True,
    all(not q for (L, m), q in table.items() if 0 < m <= 4),
)
c.item("m = 8 quantises at L_t = 6 and 8, not at 4", True, table[(8, 8.0)] and table[(6, 8.0)] and not table[(4, 8.0)])
c.item(
    "the massless free operator is quantised (trivially: N_odd = 32, f_M = 0)",
    True,
    all(table[(L, 0.0)] for L in (4, 6, 8)),
)
m_star = {L: float(np.sqrt(31) * 2 * np.sin(np.pi / L)) for L in (4, 6, 8)}
c.item(
    "threshold m* = sqrt(31) 2 sin(pi/L_t) at L_t = 4, 6, 8 (> bandwidth 4)",
    {L: round(v, 2) for L, v in m_star.items()},
    all(v > 4 for v in m_star.values()),
)
c.done()
