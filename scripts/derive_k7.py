#!/usr/bin/env python3
"""Derived data of claims I.1, I.2, I.3 and K7.1-K7.4 (group k7).

    python scripts/derive_k7.py archive     # time series of the stored chains (needs MASSPAIRING_ARCHIVE)
    python scripts/derive_k7.py mc          # the Monte Carlo comparisons of I.1 / I.2 (clean package, ~25 min CPU)
    python scripts/derive_k7.py             # both

archive -> data/derived/k7/
  t3a/<dir>/<file>.npz     the 44 epsilon-model chains of K7.1-K7.3 (series of masspairing.analysis.t3a.SERIES;
                           ts_Sigma_stag in addition on the kappa = -0.01 grid chains used by K7.1)
  calib/<set>/<file>.npz   the chains of the calibration I.3 (every scalar and fermion series of the chain summary)
  c060/{plain,hasenbusch}.npz   stored 4^4 chains at P_c, plain and Hasenbusch/multiple-time-scale RHMC (I.2 part D)
mc -> data/derived/k7/mc/<name>.npz: series of the exact-determinant Metropolis reference and of the RHMC runs it is
  compared with (2^4, all directions antiperiodic, kappa = 0, lambda = 1). The functions below are imported by the
  checks of I.1 / I.2, which regenerate a prefix of every run and require bit identity with the stored file.
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.action import build_model  # noqa: E402
from masspairing.data import DERIVED, archive_root, strip_chain  # noqa: E402
from masspairing.hmc import HMC, MTSHMC  # noqa: E402
from masspairing.lattice import Lattice  # noqa: E402
from masspairing.measure import FermionMeasure, scalar_observables  # noqa: E402

OUT = DERIVED / "k7"

# ---------------------------------------------------------------------------------------------- archive sets
T3A_PILOT = [
    "F_L6_k-0.01",
    "F5_L6_k-0.01_seed2",
    "L6_k-0.04",
    "L6_k0.02",
    "F_L8_k-0.01",
    "F2_L8_k-0.04",
    "F2_L8_k0.02",
    "L8_k-0.04",
    "F_hyst_L6_afm",
    "F_hyst_L6_cold",
    "F_hyst_L6_hot",
]
T3A_NEW = ["F12_L8_k-0.01_inner", "F12_hyst_L8_afm", "F12_hyst_L8_cold", "F12_hyst_L8_hot"]
T3A_KEYS = [
    "ts_Sigma_stag_abs",
    "ts_S_pi",
    "ts_S_pi_pmin",
    "ts_sigma2",
    "ts_O4",
    "ts_ferm_flag",
    "ts_accepted",
]
K71_DIRS = ("F_L6_k-0.01", "F_L8_k-0.01")  # also ts_Sigma_stag (vector) for K7.1
# calibration chains, in the order in which a later set replaces a point (L, kappa, y) of an earlier one
CALIB_SETS = [
    "results/calib/L8_k0.2",
    "results/calib/L6_k0.2",
    "results/calib/L12_k0.2",
    "results/calib/kscan2/*",
    "results/hmc_validation/L4_k0.2",
    "results/hmc_validation/L6_k0.2",
]
CALIB_KEYS = ["ts_dH", "ts_accepted"] + [
    "ts_" + k
    for k in (
        "sigma2",
        "Sigma_abs",
        "Sigma_stag_abs",
        "S_0",
        "S_pi",
        "S_pmin",
        "S_pi_pmin",
        "phi_abs",
        "phi_stag_abs",
        "phi_sq",
        "phi_stag_sq",
        "O4",
    )
]
C060_STORED = {
    "plain": "results/laneE/c060/plain/L4_y2.41_k-0.01.npz",
    "hasenbusch": "results/laneE/c060/hasenbusch/L4_y2.41_k-0.01.npz",
}
C060_KEYS = ["ts_sigma2", "ts_Sigma_abs", "ts_Sigma_stag_abs", "ts_O4", "ts_phi_stag_sq", "ts_dH", "ts_accepted"]


def derive_archive():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the root of the raw archive"
    raw, out = [], []
    for d in T3A_PILOT + T3A_NEW:
        for f in sorted(glob.glob(str(root / "results/xi_scan" / d / "*.npz"))):
            f = pathlib.Path(f)
            rel = f.relative_to(root).as_posix()
            z = np.load(f, allow_pickle=True)
            keys = T3A_KEYS + (["ts_Sigma_stag"] if d in K71_DIRS else [])
            extra = dict(
                tag=f"{d}/{f.name}",
                dir=d,
                set="pilot" if d in T3A_PILOT else "sharpened",
                config_shape=list(z["sigma_final"].shape),
            )
            dst = OUT / "t3a" / d / f.name
            strip_chain(f, dst, keys=keys, extra_meta=extra, rel=rel)
            raw.append(rel)
            out.append(dst)
    points = {}
    for pattern in CALIB_SETS:
        for f in sorted(glob.glob(str(root / pattern / "*.npz"))):
            f = pathlib.Path(f)
            meta = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
            points[(meta["shape"][0], round(meta["kappa"], 4), round(meta["y"], 4))] = f
    for f in sorted(points.values()):
        rel = f.relative_to(root).as_posix()
        dst = OUT / "calib" / rel.replace("results/", "").replace("/", "__")
        strip_chain(f, dst, keys=CALIB_KEYS, rel=rel)
        raw.append(rel)
        out.append(dst)
    for name, rel in C060_STORED.items():
        dst = OUT / "c060" / f"{name}.npz"
        strip_chain(root / rel, dst, keys=C060_KEYS, rel=rel)
        raw.append(rel)
        out.append(dst)
    return raw, out


# ---------------------------------------------------------------------------------------------- Monte Carlo (I.1, I.2)
KAPPA, LAM = 0.0, 1.0
KEYS = ("sigma2", "O4", "Sigma_abs", "Sigma_stag_abs", "phi_stag_sq")


def lattice_2x4():
    """2^4 with every direction antiperiodic (with L = 2 a periodic direction has no hopping)."""
    return Lattice((2, 2, 2, 2), bc=(-1, -1, -1, -1))


def _measure(series, lat, fm, fields):
    so = scalar_observables(lat, fm.model.split(fields)[0])
    fo = fm.bilinears(fields)
    for k in KEYS:
        series[k].append(so[k] if k in so else fo[k])


def metropolis(y, nsweeps, seed, delta=0.6, meas_every=5):
    """Single-site Metropolis on sigma with the weight det D(sigma) exp(-S_B): exact determinant (dense slogdet of the
    32 x 32 operator, one 2 x 2 diagonal block changes per proposal), no pseudofermions, no rational approximation, no
    molecular dynamics. Measures the five observables every ``meas_every`` sweeps with exact propagator blocks."""
    lat = lattice_2x4()
    V = lat.V
    rng = np.random.default_rng(seed)
    m = build_model(lat, y, KAPPA, LAM)
    fm = FermionMeasure(m, exact=True)
    sig = rng.normal(size=(3,) + lat.shape) * 0.5
    pack = lambda s: m.pack(s, np.zeros(0))
    M = m.D.dense(pack(sig))
    ld = np.linalg.slogdet(M)[1]
    sb = m.s_boson(pack(sig))
    series = {k: [] for k in KEYS}
    sites = [np.unravel_index(i, lat.shape) for i in range(V)]

    def block(s3):  # the on-site Yukawa block i y sigma.tau (the kinetic term has no diagonal)
        return 1j * y * np.array([[s3[2], s3[0] - 1j * s3[1]], [s3[0] + 1j * s3[1], -s3[2]]])

    for sweep in range(nsweeps):
        for i in rng.permutation(V):
            x = (slice(None),) + tuple(sites[i])
            prop = sig.copy()
            prop[x] += delta * rng.normal(size=3)
            Mp = M.copy()
            Mp[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block(prop[x])
            ldp = np.linalg.slogdet(Mp)[1]
            sbp = m.s_boson(pack(prop))
            if np.log(rng.random()) < (ldp - ld) - (sbp - sb):  # det D = exp(ld) > 0 on every configuration
                sig, ld, sb, M = prop, ldp, sbp, Mp
        if sweep % meas_every == 0:
            _measure(series, lat, fm, pack(sig))
    return {k: np.asarray(v) for k, v in series.items()}


def rhmc(y, ntraj, seed, ntherm=200, nsteps=6):
    """Plain RHMC (Omelyan, tau = 1) on the same 2^4 system; the five observables after every trajectory."""
    lat = lattice_2x4()
    m = build_model(lat, y, KAPPA, LAM, cg_tol=1e-10)
    h = HMC(m, tau=1.0, nsteps=nsteps, seed=seed)
    return _run(h, m, lat, ntraj, ntherm)


def rhmc_hasenbusch(y, ntraj, seed, shifts=(0.3, 3.0), levels=(3, 1, 1), ntherm=200, drop_ratios=False):
    """Hasenbusch-split RHMC with nested time scales on the same system. ``drop_ratios`` (sabotage) keeps only the
    heavy term, i.e. simulates det(D^dagger D + s_n)^(1/2) instead of det(D^dagger D)^(1/2)."""
    lat = lattice_2x4()
    m = build_model(lat, y, KAPPA, LAM, hasenbusch=list(shifts), cg_tol=1e-10)
    if drop_ratios:
        m.terms = [m.terms[-1]]
        m.stats["cg_iters_term"] = [0]
        m.stats["lanczos_iters_term"] = [0]
        levels = (levels[0],)
    h = MTSHMC(m, tau=1.0, levels=levels, seed=seed)
    return _run(h, m, lat, ntraj, ntherm)


def _run(h, m, lat, ntraj, ntherm):
    fm = FermionMeasure(m, exact=True)
    fields = m.pack(lat.random_sigma(h.rng, 0.5), np.zeros(0))
    series = {k: [] for k in KEYS}
    acc = []
    for it in range(ntherm + ntraj):
        fields, info = h.trajectory(fields)
        if it < ntherm:
            continue
        acc.append(info["accepted"])
        _measure(series, lat, fm, fields)
    out = {k: np.asarray(v) for k, v in series.items()}
    out["accepted"] = np.asarray(acc)
    return out


# name: (function, positional arguments, keyword arguments). The y = 1.2 Metropolis reference serves I.1 and I.2.
MC_RUNS = {
    "metropolis_y0": (metropolis, (0.0, 4000, 1), {}),
    "rhmc_y0": (rhmc, (0.0, 600, 2), dict(ntherm=100)),
    "metropolis_y1.2": (metropolis, (1.2, 40000, 3), {}),
    "rhmc_y1.2": (rhmc, (1.2, 2500, 4), {}),
    "hasenbusch_y1.2": (rhmc_hasenbusch, (1.2, 2500, 4), {}),
    "metropolis_y2.5": (metropolis, (2.5, 20000, 6), {}),
    "hasenbusch_y2.5": (rhmc_hasenbusch, (2.5, 1500, 7), {}),
    "sabotage_heavy_only_y2.5": (rhmc_hasenbusch, (2.5, 1000, 8), dict(drop_ratios=True)),
}
# the length argument (sweeps or trajectories) is the second positional argument of every run
PREFIX = {"metropolis": 200, "rhmc": 40, "hasenbusch": 40, "sabotage": 40}


def mc_run(name, length=None):
    """Run ``name`` of MC_RUNS, with its length replaced by ``length`` (the prefix used by the checks)."""
    f, args, kw = MC_RUNS[name]
    if length is not None:
        args = (args[0], length) + tuple(args[2:])
    return f(*args, **kw)


def prefix_length(name):
    return PREFIX[name.split("_")[0]]


def derive_mc(names=None):
    out = []
    for name in names or MC_RUNS:
        t0 = time.time()
        res = mc_run(name)
        f, args, kw = MC_RUNS[name]
        meta = dict(function=f.__name__, args=list(args), kwargs=kw)
        dst = OUT / "mc" / f"{name}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, meta=json.dumps(meta), **{"ts_" + k: v for k, v in res.items()})
        print(f"{name}: {time.time() - t0:.1f} s -> {dst}", flush=True)
        out.append(dst)
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("archive", "all"):
        raw, files = derive_archive()
        print(f"archive: {len(raw)} raw files -> {len(files)} derived files")
    if what in ("mc", "all"):
        derive_mc(sys.argv[2:] or None)
