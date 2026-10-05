#!/usr/bin/env python3
"""Derived data of the N1 stage-1c claims (K5.8, K5.9, K5.13, K8.3); needs MASSPAIRING_ARCHIVE.

Writes
  data/derived/n1stage/S1c/<run>/<chain>.npz        the nine stage-1c chains: four independent 8^4 replicas at y = 3.0
                                                   (h = 1, 2; runs L8_h{1,2}_r{A,B}) and five 6^4 chains at P_c
                                                   (h = 0.25, 0.5, 0.75 in run L6; replicas h = 0.25, 0.5 in run L6_r2)
  data/derived/free/free_L6_aaaa_h0.25_0.5_0.75.json   the free 6^4 baseline at the new h (copied; identical to the
                                                   archive file)
  data/manifest/n1stage1c.tsv                      the manifest fragment
Chain files keep only the time series the analyses read (no configurations) plus `cfg_digest`, as in
scripts/derive_n1stage.py.
"""

import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from derive_n1stage import KEYS, META_DROP, digest  # noqa: E402

from masspairing.data import DERIVED, archive_root, md5, strip_chain  # noqa: E402

OUT = DERIVED / "n1stage" / "S1c"
FREE = DERIVED / "free"
S1C = "results/xi_scan/K_N1_S1c"
RUNS = ("L8_h2_rA", "L8_h2_rB", "L8_h1_rA", "L8_h1_rB", "L6", "L6_r2")
FREE_FILE = "free_L6_aaaa_h0.25_0.5_0.75.json"
USERS_8 = "K5.8,K5.9,K5.13"
USERS_6 = "K5.9,K8.3"
RAW = []  # (archive path, used_by)


def chain(root, rel, dst, used_by):
    src = root / rel
    RAW.append((rel, used_by))
    strip_chain(src, dst, keys=KEYS, rel=rel)
    z = dict(np.load(dst, allow_pickle=True))
    meta = json.loads(str(z.pop("meta")))
    for k in META_DROP:
        meta.pop(k, None)
    d = np.load(src, allow_pickle=True)
    z["cfg_digest"] = np.array([digest(c) for c in d["ts_cfg"]])
    np.savez_compressed(dst, meta=json.dumps(meta, default=str), **z)


def chains(root):
    for run in RUNS:
        for f in sorted((root / S1C / run).glob("*.npz")):
            chain(root, f"{S1C}/{run}/{f.name}", OUT / run / f.name, USERS_8 if run.startswith("L8") else USERS_6)


def free(root):
    rel = f"results/laneK1/{FREE_FILE}"
    RAW.append((rel, USERS_6))
    dst = FREE / FREE_FILE
    if dst.exists():
        assert md5(dst) == md5(root / rel), f"{dst} differs from the archive file"
    else:
        shutil.copyfile(root / rel, dst)


def manifest(root):
    lines = ["kind\tpath\tmd5\tbytes\tproduced_by\tused_by"]
    for rel, u in sorted(RAW):
        p = root / rel
        prod = "chain run (archive)" if rel.endswith(".npz") else "notebook output (archive)"
        lines.append(f"raw\t{rel}\t{md5(p)}\t{p.stat().st_size}\t{prod}\t{u}")
    prod = "scripts/derive_n1stage1c.py"
    for p in sorted(OUT.rglob("*.npz")):
        u = USERS_8 if p.parent.name.startswith("L8") else USERS_6
        lines.append(f"derived\t{p.relative_to(OUT.parents[3]).as_posix()}\t-\t-\t{prod}\t{u}")
    lines.append(f"derived\tdata/derived/free/{FREE_FILE}\t-\t-\t{prod}\t{USERS_6}")
    (OUT.parents[2] / "manifest" / "n1stage1c.tsv").write_text("\n".join(lines) + "\n")


def main():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the archive root"
    OUT.mkdir(parents=True, exist_ok=True)
    chains(root)
    free(root)
    manifest(root)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"data/derived/n1stage/S1c: {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
