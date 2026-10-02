"""I.5: the link-field HMC at fixed trajectory length freezes at the prior resonance omega_j tau = n pi;
trajectory-length jitter removes it.

Each pair field s has the Gaussian prior s^2/(4 g) and unit mass: a harmonic oscillator of frequency
omega = (2 g)^(-1/2), on which the fermion force is a small perturbation. A trajectory of length tau maps
x -> x cos(omega tau) + (p/omega) sin(omega tau); with fresh momenta <x^2>_n / 2g = 1 - cos^(2n)(omega tau) from x = 0.
(1) Stored 4^4 chains at g10 = 0.05, g6 = 0 (omega tau = 3.162, tau = 1): <s^2>/2g is 0.40-0.43 in the first and
    0.47-0.53 in the last quarter of 400 trajectories (rising), while the exact conditional variance of single link
    fields on the final configurations (dense determinant scan) is 2g within 3 %; the controls at omega tau = 5.0 and
    2.24 are equilibrated (0.9-1.2).
(2) The free-oscillator ramp and the per-trajectory mixing sin^2(omega tau): 0.0004 at tau = 1, 0.64 at tau = 0.7.
(3) Live (scripts/run_chain.py, 4^4, y = 1, g10 = 0.05, g6 = 0, 150 trajectories from s = 0, 10 Omelyan steps):
    tau = 1.0 stays below 0.6 over its second half, tau = 0.7 is above 0.9; both series are bit-identical to the
    stored runs.
"""

import importlib.util
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from masspairing.analysis import k3, k3_exact
from masspairing.claimcheck import Check

STORED = [  # (archive chain, configuration name, resonant)
    ("results/laneW2/c072/L4_y1_k-0.01_g0.05_g60.npz", "c072_L4_y1_g0.05_g60", True),
    ("results/laneW2/c072/L4_y1.7_k-0.01_g0.05_g60.npz", "c072_L4_y1.7_g0.05_g60", True),
    ("results/laneW2/c072/n20/L4_y2.41_k-0.01_g0.05_g60.npz", "c072n20_L4_y2.41_g0.05_g60", True),
    ("results/laneW2/c072/n20/L4_y3_k-0.01_g0.05_g60.npz", "c072n20_L4_y3_g0.05_g60", True),
    ("results/laneW2/c072/L4_y2.41_k-0.01_g0.02_g60.npz", "c072_L4_y2.41_g0.02_g60", False),
    ("results/laneW2/c072/L4_y2.41_k-0.01_g0.05_g60.05.npz", "c072_L4_y2.41_g0.05_g60.05", False),
]
c = Check("I.5")
for rel, cfg, resonant in STORED:
    s, meta = k3.load(rel)
    F, model = k3_exact.load_config(cfg)
    g, lat = model.geom, model.lat
    s2 = s["ts_s2"]
    n = len(s2)
    gpair = g.g[1]  # g6 = 0: both pairs carry g10; the edge point has only the mu pair
    first, last = s2[: n // 4].mean() / (2 * gpair), s2[-(n // 4) :].mean() / (2 * gpair)
    rng = np.random.default_rng(1)
    vs = []
    for fi in rng.choice(g.n_fields, 4, replace=False):
        gj = g.g[g.f_pair[fi]]
        grid = np.linspace(-4 * np.sqrt(2 * gj), 4 * np.sqrt(2 * gj), 33)
        lw = []
        for sv in grid:
            fl = F.copy()
            fl[3 * lat.V + fi] = sv
            lw.append(-(sv**2) / (4 * gj) + np.linalg.slogdet(model.D.dense(fl))[1])
        lw = np.array(lw)
        wgt = np.exp(lw - lw.max())
        wgt /= wgt.sum()
        mu = (wgt * grid).sum()
        vs.append((wgt * (grid - mu) ** 2).sum() / (2 * gj))
    cond = float(np.mean(vs))
    om = meta["tau"] / np.sqrt(2 * gpair)
    good = (
        (first < 0.6 and last < 0.6 and last > first + 0.03) if resonant else (0.9 < first < 1.2 and 0.9 < last < 1.2)
    )
    c.item(
        f"(1) y={meta['y']:g} g10={meta['g10']:g} g6={meta['g6']:g}, omega tau = {om:.3f}: <s^2>/2g first -> last "
        f"quarter; exact conditional variance / 2g ({'resonant: frozen' if resonant else 'control: equilibrated'})",
        (first, last, cond),
        good and abs(cond - 1) < 0.03,
        fmt="{0[0]:.3f} -> {0[1]:.3f}; {0[2]:.3f}",
    )
for tau, gj in ((1.0, 0.05), (1.0, 0.05 * (3.175 / 3.162) ** -2), (0.7, 0.05), (1.0, 0.02), (1.0, 0.1)):
    cc = np.cos(tau / np.sqrt(2 * gj))
    c.record(
        f"    free-oscillator ramp tau = {tau}, g = {gj:.4f} (omega tau = {tau / np.sqrt(2 * gj):.3f}), n = 1, 60, "
        "150, 460",
        ", ".join(f"{1 - cc ** (2 * nn):.3f}" for nn in (1, 60, 150, 460)),
    )
m1, m7 = 1 - np.cos(1.0 / np.sqrt(0.1)) ** 2, 1 - np.cos(0.7 / np.sqrt(0.1)) ** 2
c.item(
    "(2) per-trajectory mixing sin^2(omega tau) at g = 0.05: tau = 1.0, tau = 0.7",
    (m1, m7),
    m1 < 1e-3 and m7 > 0.5,
    fmt="{0[0]:.4f}, {0[1]:.3f}",
)
spec = importlib.util.spec_from_file_location("run_chain", ROOT / "scripts" / "run_chain.py")
run_chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_chain)
res = {}
with tempfile.TemporaryDirectory() as tmp:
    for tau in (1.0, 0.7):
        args = run_chain.parse(
            [
                "--L",
                "4",
                "--y",
                "1",
                "--kappa",
                "-0.01",
                "--g10",
                "0.05",
                "--g6",
                "0",
                "--ntraj",
                "150",
                "--ntherm",
                "0",
                "--tau",
                str(tau),
                "--nsteps",
                "10",
                "--seed",
                "71",
                "--no-corr",
                "--ferm-every",
                "100000",
                "--exact",
                "--measure",
                "channels",
                "--checkpoint-every",
                "150",
                "--out",
                f"{tmp}/tau{tau}",
            ]
        )
        d = np.load(run_chain.run(args), allow_pickle=True)
        s2 = d["ts_s2"]
        stored, _ = k3.load(f"results/laneR/c102/tau{tau}/L4_y1_k-0.01_g0.05_g60.npz")
        res[tau] = (
            s2[:75].mean() / 0.1,
            s2[75:].mean() / 0.1,
            d["ts_accepted"].mean(),
            np.array_equal(s2, stored["ts_s2"]) and np.array_equal(d["ts_accepted"], stored["ts_accepted"]),
        )
        c.record(
            f"    live tau = {tau}: <s^2>/2g first half, second half, acceptance",
            f"{res[tau][0]:.3f}, {res[tau][1]:.3f}, {res[tau][2]:.2f}",
        )
c.item(
    "(3) live 4^4 chains: tau = 1.0 second half < 0.6, tau = 0.7 both halves > 0.9",
    (res[1.0][1], res[0.7][0], res[0.7][1]),
    res[1.0][1] < 0.6 and res[0.7][0] > 0.9 and res[0.7][1] > 0.9,
    fmt="{0[0]:.3f}; {0[1]:.3f}, {0[2]:.3f}",
)
c.item(
    "(3) live series (<s^2>, accept/reject) bit-identical to the stored runs",
    (res[1.0][3], res[0.7][3]),
    res[1.0][3] and res[0.7][3],
)
c.done()
