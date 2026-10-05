#!/usr/bin/env python3
"""The restricted-mode control of K5.15: chi_L on stored 8^4 (3.0, h = 2) configurations with P_L restricted.

    MASSPAIRING_ARCHIVE=/path python scripts/run_k5_control.py free            # free values (sigma = 0), ~6 min
    MASSPAIRING_ARCHIVE=/path python scripts/run_k5_control.py L8_h2_rA K N    # shard K of N of one replica
    python scripts/run_k5_control.py collect                                   # write data/derived/k5control/

Configurations: the stage-1c replicas rA, rB at (y, kappa, h) = (3.0, -0.01, 2) on 8^4 aaaa (raw archive files
results/xi_scan/K_N1_S1c/L8_h2_r{A,B}/..., ``ts_cfg``), every second stored configuration with trajectory >= 100.
Two restrictions of the light pattern (``restricted_modes.light_basis``): the IR class (256 momenta with every
|cos p_mu| = cos(pi/8), basis rank 128) and the source-free class (1536 momenta with P_L C_chi P_L v_p = 0, rank 768).
Per configuration one sparse LU of the 8^4 doublet operator (about 65 s, 40 M non-zeros in L + U) and 2r solves
(IR class about 22 s, source-free class about 121 s). The two bases are dense 4096 x 768 at most, but building them
goes through dense 4096 x 4096 projectors: run under a memory cap (a few GB). The 90 configurations of K5.15 took
12.7 CPU-hours (8 single-thread shards).
Not a ``derive_*`` script: ``make derived`` does not run it.
"""

import json
import os
import pathlib
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from masspairing.analysis import restricted_modes as R  # noqa: E402
from masspairing.corner import doublet_operator  # noqa: E402
from masspairing.data import archive_root  # noqa: E402
from masspairing.lattice import Lattice  # noqa: E402

S1C = "results/xi_scan/K_N1_S1c"
CHAIN = "L8_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz"
WORK = pathlib.Path(os.environ.get("K5_CONTROL_WORK", ROOT / "k5control_work"))


def bases(lat):
    cache = WORK / "bases_L8.npz"
    if cache.exists():
        z = np.load(cache)
        return z["W_ir"], z["c_ir"], z["W_sf"], z["c_sf"]
    sf = R.light_block(lat.shape, lat.bc)["source_free_mask"]
    ir = R.ir_class_mask(lat.shape, lat.bc)
    W_ir, c_ir = R.light_basis(lat, ir)
    W_sf, c_sf = R.light_basis(lat, sf)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(cache, W_ir=W_ir, c_ir=c_ir, W_sf=W_sf, c_sf=c_sf)
    return W_ir, c_ir, W_sf, c_sf


def main():
    lat = Lattice((8, 8, 8, 8), bc=(-1,) * 4)
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "collect":
        out = {}
        for rep in ("rA", "rB"):
            a = np.concatenate([np.load(f) for f in sorted(WORK.glob(f"L8_h2_{rep}_shard*.npy"))])
            out[rep] = a[np.argsort(a[:, 0])]
        free = json.loads((WORK / "free.json").read_text())
        meta = dict(
            observable="chi_L' = V <phi_L'^2>, flavour-summed p = 0 pair channel with P_L restricted",
            ensemble="8^4 aaaa, y = 3.0, kappa = -0.01, h = 2, stage-1c replicas rA, rB; every second stored "
            "configuration with trajectory >= 100",
            columns=list(R.CONTROL_COLUMNS),
            ir_class="256 momenta with every |cos p_mu| = cos(pi/8) (basis rank 128)",
            source_free="1536 momenta with P_L C_chi P_L v_p = 0 (basis rank 768)",
            producer="scripts/run_k5_control.py",
        )
        R.CONTROL.mkdir(parents=True, exist_ok=True)
        np.savez(
            R.CONTROL / "control_L8_y3_h2.npz",
            rA=out["rA"],
            rB=out["rB"],
            columns=np.array(meta["columns"]),
            meta=json.dumps(meta),
        )
        (R.CONTROL / "free_L8_h2_restricted.json").write_text(json.dumps(free, indent=1))
        return
    W_ir, c_ir, W_sf, c_sf = bases(lat)
    if mode == "free":
        free = dict(L=8, Lt=8, bc="aaaa", h=2.0)
        for h, sfx in ((2.0, ""), (0.0, "_h0")):
            lu = spla.splu(doublet_operator(lat, 0.0, np.zeros(3 * lat.V), h=h, pattern="chiral"))
            ci, pi_ = R.chi_reduced(lat, 0.0, np.zeros(3 * lat.V), h, W_ir, c_ir, lu)
            cs, ps = R.chi_reduced(lat, 0.0, np.zeros(3 * lat.V), h, W_sf, c_sf, lu)
            free.update({f"chi_ir{sfx}": ci, f"chi_sf{sfx}": cs})
            if not sfx:
                free.update(phi_ir=pi_, phi_sf=ps)
        (WORK / "free.json").write_text(json.dumps(free, indent=1))
        return
    if not mode.startswith("L8_h2_r") or len(sys.argv) != 4:
        sys.exit(__doc__)
    root = archive_root()
    if root is None:
        sys.exit("set MASSPAIRING_ARCHIVE")
    k, n = int(sys.argv[2]), int(sys.argv[3])
    z = np.load(root / S1C / mode / CHAIN, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    traj = z["ts_cfg_traj"]
    mine = np.where(traj >= 100)[0][::2][k::n]
    res = []
    for i in mine:
        t0 = time.time()
        sig = z["ts_cfg"][i]
        lu = spla.splu(doublet_operator(lat, meta["y"], sig, h=meta["h10"], pattern="chiral"))
        ci, pi_ = R.chi_reduced(lat, meta["y"], sig, meta["h10"], W_ir, c_ir, lu)
        cs, ps = R.chi_reduced(lat, meta["y"], sig, meta["h10"], W_sf, c_sf, lu)
        res.append((int(traj[i]), ci, cs, pi_, ps, float(z["ts_chi_L_sum"][i])))
        print(f"{mode} traj {traj[i]}: chi_ir {ci:.6e} chi_sf {cs:.6e} [{time.time() - t0:.0f} s]", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / f"{mode}_shard{k}.npy", np.array(res))


if __name__ == "__main__":
    main()
