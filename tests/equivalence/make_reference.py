#!/usr/bin/env python3
"""Freeze the notebook runners' output for the chain cases (needs a notebook checkout).

    MASSPAIRING_NOTEBOOK=/path/to/unhiggsed_notebook python tests/equivalence/make_reference.py

The notebook is run with its own interpreter (``$MASSPAIRING_NOTEBOOK/.venv/bin/python`` if present) and
OPENBLAS_NUM_THREADS=1; each case's ts_ series and final configuration go to tests/reference/chains/<case>.npz.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES, COMMON  # noqa: E402

OUT = HERE.parent / "reference" / "chains"


def main():
    nb = pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"])
    py = nb / ".venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    for name, (runner, old, _new) in CASES.items():
        if only and name not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [py, str(nb / "scripts" / runner)] + COMMON + old + ["--out", tmp]
            subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
            (f,) = list(pathlib.Path(tmp).glob("*.npz"))
            d = np.load(f, allow_pickle=True)
            keep = {k: d[k] for k in d.files if k.startswith("ts_")}
            keep["fields_final"] = (d["fields_final"] if "fields_final" in d.files else d["sigma_final"]).reshape(-1)
            np.savez_compressed(OUT / f"{name}.npz", **keep)
            print(f"{name}: {len(keep)} arrays from {f.name}", flush=True)


if __name__ == "__main__":
    main()
