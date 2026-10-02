#!/usr/bin/env python3
"""Derived data of the algebra group (needs MASSPAIRING_ARCHIVE): the final configurations of the eps-model chains at
P_c (y = 2.41, kappa = -0.01, no link fields) on 4^4 and 6^4, read by the checks of K2.4 and K2.5.

Each output data/derived/algebra/<name>.npz holds ``fields`` (the flat sigma triplet, 3V) and a JSON ``meta`` (the
chain's parameters plus source path in the archive, md5 and size).
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import archive_root, derived, md5  # noqa: E402

SOURCES = {
    "eps_L4_y2.41_k-0.01_final": "results/xi_scan/F8_P3_eps/L4_y2.41_k-0.01_g0_g60.npz",
    "eps_L6_y2.41_k-0.01_final": "results/xi_scan/F8_P3_eps/L6_y2.41_k-0.01_g0_g60.npz",
}
KEEP = ("L", "y", "kappa", "lam", "g10", "g6", "quads", "h10", "ntraj", "ntherm", "seed", "start", "shape", "bc")


def main():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the archive root"
    for name, rel in SOURCES.items():
        src = root / rel
        z = np.load(src, allow_pickle=True)
        meta = {k: v for k, v in json.loads(str(z["meta"])).items() if k in KEEP}
        fields = np.asarray(z["fields_final"], float)
        assert fields.size == 3 * int(np.prod(meta["shape"])), "expected sigma only (no link fields)"
        meta.update(source=rel, source_md5=md5(src), source_bytes=src.stat().st_size, content="final configuration")
        dst = derived("algebra", name + ".npz")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, fields=fields, meta=json.dumps(meta))
        print(dst.relative_to(dst.parents[3]), fields.size, meta["source_md5"])


if __name__ == "__main__":
    main()
