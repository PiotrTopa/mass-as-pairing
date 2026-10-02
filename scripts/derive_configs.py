#!/usr/bin/env python3
"""Extract the stored configurations used by the re-measurement tests (needs MASSPAIRING_ARCHIVE).

Each output data/configs/<name>.npz holds `fields` (flat [sigma, s]), `row` (the chain's stored measurement of that
configuration: every ts_ series at that index, as ts_<key>), and a JSON `meta` (chain parameters, source path, md5).
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import CONFIGS, archive_root, md5  # noqa: E402

# name: (archive path, row selector) -- "cfg:i" takes ts_cfg[i] and the fermion-measurement row i;
# "final" takes the final configuration and the last row of the per-trajectory (bosonic) series
SOURCES = {
    "n1_L4_y3_h2_aaaa": ("results/xi_scan/K_N1_S0/L4/L4_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", ["cfg:-1", "cfg:-2"]),
    "n1_L4_y2.41_h0.5_aaaa": ("results/xi_scan/K_N1_S0/L4/L4_y2.41_k-0.01_g0_g60_h0.5_chiral_bcaaaa.npz", ["cfg:-1"]),
    "n1_L6_y3_h2_aaaa": ("results/xi_scan/K_N1_S1/L6/L6_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", ["cfg:-1"]),
    "n1_L6_y2.41_h2_aaaa": ("results/xi_scan/K_N1_S1/L6/L6_y2.41_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", ["cfg:-1"]),
    "wedge_L6_y2.41_g0.1_edge-": ("results/xi_scan/F9_X_y0exit/L6_y2.41_k-0.01_g0.1_g6-0.1.npz", ["final"]),
    "eps_L8_y2.41_afm": ("results/xi_scan/F12_hyst_L8_afm/L8_y2.41_k-0.01.npz", ["final"]),
}

PER_TRAJ = (
    "Sigma",
    "Sigma_stag",
    "Sigma_abs",
    "Sigma_stag_abs",
    "sigma2",
    "sigma4",
    "S_0",
    "S_pi",
    "S_pmin",
    "S_pi_pmin",
    "Ct",
    "Ct_stag",
    "s2",
    "s_mean_0",
    "s_mean_1",
    "s2_0",
    "s2_1",
    "Ss_0_0",
    "Ss_pi_0",
    "Ss_pmin_0",
    "Ss_pi_pmin_0",
    "Ss_0_1",
    "Ss_pi_1",
    "Ss_pmin_1",
    "Ss_pi_pmin_1",
)


def main():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the archive root"
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for name, (rel, rows) in SOURCES.items():
        src = root / rel
        z = np.load(src, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        fermion_keys = [
            k
            for k in z.files
            if k.startswith("ts_") and len(z[k]) == len(z.get("ts_cfg", [])) and k not in ("ts_cfg", "ts_cfg_traj")
        ]
        out = {}
        for j, sel in enumerate(rows):
            if sel == "final":
                f = z["fields_final"] if "fields_final" in z.files else z["sigma_final"].reshape(-1)
                row = {k: z["ts_" + k][-1] for k in PER_TRAJ if "ts_" + k in z.files}
            else:
                i = int(sel.split(":")[1])
                f = z["ts_cfg"][i]
                row = {k[3:]: z[k][i] for k in fermion_keys}
                row["cfg_traj"] = z["ts_cfg_traj"][i]
            out[f"fields_{j}"] = np.asarray(f, float)
            for k, v in row.items():
                out[f"row{j}_{k}"] = np.asarray(v)
        info = dict(chain=meta, source=rel, source_md5=md5(src), source_bytes=src.stat().st_size, rows=rows)
        np.savez_compressed(CONFIGS / f"{name}.npz", meta=json.dumps(info, default=str), **out)
        print(name, len(out), "arrays", flush=True)


if __name__ == "__main__":
    main()
