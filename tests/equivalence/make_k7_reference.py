#!/usr/bin/env python3
"""Freeze the notebook side of tests/equivalence/test_k7.py into tests/reference/k7/.

Needs MASSPAIRING_NOTEBOOK (research notebook at 3c4a02c) and MASSPAIRING_ARCHIVE (raw chains; for the notebook it is
the notebook root itself).

  t3a_pilot_{verdict,fss,labels}.json      notebook frozen outputs of the pilot-set analysis (39 chains)
  t3a_sharpened_{verdict,fss,labels}.json  notebook frozen outputs of the 44-chain analysis (2000-trajectory starts)
  t3a_sabotage.json                        notebook frozen sabotage results of the 44-chain analysis
  k71_table.json                           notebook frozen ratio table of the kappa = -0.01 prefixes
  k74_discrimination.json                  notebook frozen walking / power-law mock table
  calib_points.json                        notebook chain summary (scripts/analyse.py) of every calibration chain
  mc_short.npz                             short runs of the notebook brute-force Metropolis and RHMC codes
"""

import glob
import importlib.util
import json
import os
import pathlib
import shutil
import sys

import numpy as np

NB = pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"])
ARCHIVE = pathlib.Path(os.environ.get("MASSPAIRING_ARCHIVE", NB))
OUT = pathlib.Path(__file__).resolve().parents[1] / "reference" / "k7"
sys.path.insert(0, str(NB / "src"))


def copy_frozen():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = {
        "results/laneK2/c051/verdict.json": "t3a_pilot_verdict.json",
        "results/laneK2/c051/fss_table.json": "t3a_pilot_fss.json",
        "results/laneK2/c051/phase_labels.json": "t3a_pilot_labels.json",
        "results/laneK2/c192/verdict.json": "t3a_sharpened_verdict.json",
        "results/laneK2/c192/fss_table.json": "t3a_sharpened_fss.json",
        "results/laneK2/c192/phase_labels_new.json": "t3a_sharpened_labels.json",
        "results/laneK2/c192/sabotage.json": "t3a_sabotage.json",
        "results/laneX/c083/fss_table.json": "k71_table.json",
        "results/laneX/c082/discriminate_mock.json": "k74_discrimination.json",
    }
    for src, dst in pairs.items():
        shutil.copy(NB / src, OUT / dst)


def calib_points():
    spec = importlib.util.spec_from_file_location("analyse", NB / "scripts" / "analyse.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    out = {}
    for pattern in (
        "results/calib/L8_k0.2/*.npz",
        "results/calib/L6_k0.2/*.npz",
        "results/calib/L12_k0.2/*.npz",
        "results/calib/kscan2/*/*.npz",
        "results/hmc_validation/L4_k0.2/*.npz",
        "results/hmc_validation/L6_k0.2/*.npz",
    ):
        for f in sorted(glob.glob(str(ARCHIVE / pattern))):
            r = an.analyse_file(pathlib.Path(f), quiet=True)
            key = f"{r['shape'][0]}|{round(r['kappa'], 4)}|{round(r['y'], 4)}"
            out[key] = dict(
                ntraj=r["ntraj"],
                acceptance=r["acceptance"],
                blen=r["blen"],
                tau_max=max(v["tau_int"] for v in r.values() if isinstance(v, dict) and "tau_int" in v),
                Sigma_stag_abs=[r["Sigma_stag_abs"]["mean"], r["Sigma_stag_abs"]["err"]],
                chi_Sigma_stag=[r["chi_Sigma_stag"]["mean"], r["chi_Sigma_stag"]["err"]],
                xi2_stag=[r["xi2_stag"]["mean"], r["xi2_stag"]["err"]],
            )
    (OUT / "calib_points.json").write_text(json.dumps(out, indent=1) + "\n")


# ---- the notebook's brute-force comparison codes (2^4 all antiperiodic, kappa = 0, lambda = 1), short runs
KEYS = ("sigma2", "O4", "Sigma_abs", "Sigma_stag_abs", "phi_stag_sq")


def mc_short():
    from unhiggsed.hmc.action import HasenbuschModel, Model
    from unhiggsed.hmc.dirac import Dirac
    from unhiggsed.hmc.hmc import HMC, MTSHMC
    from unhiggsed.hmc.lattice import Lattice
    from unhiggsed.hmc.observables import FermionMeasure, scalar_observables

    lat = Lattice((2, 2, 2, 2), bc=(-1, -1, -1, -1))
    V = lat.V

    def measure(series, fm, s):
        so, fo = scalar_observables(lat, s), fm.bilinears(s)
        for k in KEYS:
            series[k].append(so[k] if k in so else fo[k])

    def metropolis(y, nsweeps, seed, delta=0.6, meas_every=5):
        rng = np.random.default_rng(seed)
        D, m = Dirac(lat, y), Model(lat, y, 0.0, 1.0)
        fm = FermionMeasure(m, exact=True)
        s = rng.normal(size=(3,) + lat.shape) * 0.5
        M = D.dense(s)
        ld, sb = np.linalg.slogdet(M)[1], m.s_boson(s)
        series = {k: [] for k in KEYS}
        sites = [np.unravel_index(i, lat.shape) for i in range(V)]

        def block(s3):
            return 1j * y * np.array([[s3[2], s3[0] - 1j * s3[1]], [s3[0] + 1j * s3[1], -s3[2]]])

        for sweep in range(nsweeps):
            for i in rng.permutation(V):
                x = sites[i]
                prop = s.copy()
                prop[(slice(None),) + tuple(x)] += delta * rng.normal(size=3)
                Mp = M.copy()
                Mp[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block(prop[(slice(None),) + tuple(x)])
                ldp, sbp = np.linalg.slogdet(Mp)[1], m.s_boson(prop)
                if np.log(rng.random()) < (ldp - ld) - (sbp - sb):
                    s, ld, sb, M = prop, ldp, sbp, Mp
            if sweep % meas_every == 0:
                measure(series, fm, s)
        return series

    def run(h, m, ntraj, ntherm):
        fm = FermionMeasure(m, exact=True)
        s = lat.random_sigma(h.rng, 0.5)
        series = {k: [] for k in KEYS}
        acc = []
        for it in range(ntherm + ntraj):
            s, info = h.trajectory(s)
            if it >= ntherm:
                acc.append(info["accepted"])
                measure(series, fm, s)
        series["accepted"] = acc
        return series

    def rhmc(y, ntraj, seed, ntherm):
        m = Model(lat, y, 0.0, 1.0, cg_tol=1e-10)
        return run(HMC(m, tau=1.0, nsteps=6, seed=seed), m, ntraj, ntherm)

    def rhmc_h(y, ntraj, seed, ntherm, drop_ratios=False):
        m = HasenbuschModel(lat, y, 0.0, 1.0, [0.3, 3.0], cg_tol=1e-10)
        levels = (3, 1, 1)
        if drop_ratios:
            m.terms = [m.terms[-1]]
            m.stats["cg_iters_term"] = [0]
            m.stats["lanczos_iters_term"] = [0]
            levels = (3,)
        return run(MTSHMC(m, tau=1.0, levels=levels, seed=seed), m, ntraj, ntherm)

    out = {}
    for name, res in (
        ("metropolis_y1.2", metropolis(1.2, 30, 3)),
        ("metropolis_y2.5", metropolis(2.5, 30, 6)),
        ("rhmc_y1.2", rhmc(1.2, 10, 4, 5)),
        ("hasenbusch_y2.5", rhmc_h(2.5, 10, 7, 5)),
        ("sabotage_y2.5", rhmc_h(2.5, 10, 8, 5, drop_ratios=True)),
    ):
        for k, v in res.items():
            out[f"{name}/{k}"] = np.asarray(v)
    np.savez_compressed(OUT / "mc_short.npz", **out)


if __name__ == "__main__":
    copy_frozen()
    calib_points()
    mc_short()
    print(f"wrote {sorted(p.name for p in OUT.iterdir())}")
