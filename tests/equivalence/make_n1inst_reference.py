#!/usr/bin/env python3
"""Freeze notebook-side outputs for tests/equivalence/test_n1inst.py (needs a notebook checkout; run it with
the notebook's interpreter, NOTEBOOK_PYTHON).

    MASSPAIRING_NOTEBOOK=/path/to/notebook $NOTEBOOK_PYTHON tests/equivalence/make_n1inst_reference.py

Writes tests/reference/n1inst/:
  met_woodbury_prefix.npz  the notebook's exact-determinant Metropolis of the N1 comparison, 305 sweeps (seed 1610)
  i8_prefix.npz            the |Pf| Metropolis (206 sweeps, seed 1400) and the two RHMC chains (4 measured
                           trajectories, seeds 1402 and 1403 with the odd-sublattice sabotage) of the selective source
  takeover_rows.json       the composite-takeover rows of the notebook analysis (full precision)
  winding_stored.json      the notebook's loop readouts of the 39 stored configurations
  e9_box_shape.json        the notebook's free box-shape series and stabiliser-element counts
"""

import importlib.util
import json
import os
import pathlib
import runpy
import sys

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
NB = pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"])
sys.path.insert(0, str(NB / "src"))
OUT = pathlib.Path(__file__).resolve().parents[1] / "reference" / "n1inst"


def claim_module(prefix):
    (path,) = list((NB / "claims").glob(f"{prefix}_*/check.py"))
    spec = importlib.util.spec_from_file_location(prefix, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # the checks guard their main part
    return mod, path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m161, _ = claim_module("C161")
    r = m161.metropolis((305, 1610))
    np.savez_compressed(OUT / "met_woodbury_prefix.npz", **{k: np.asarray(v) for k, v in r.items()})
    m140, _ = claim_module("C140")
    out = {}
    for name, res in (
        ("met", m140.metropolis((206, 1400))),
        ("rhmc", m140.rhmc((4, 1402, False))),
        ("sabotage", m140.rhmc((4, 1403, True))),
    ):
        out.update({f"{name}__{k}": np.asarray(v) for k, v in res.items() if k != "s2"})
    np.savez_compressed(OUT / "i8_prefix.npz", **out)
    (path,) = list((NB / "claims").glob("C150_*/check.py"))
    exit_ = sys.exit
    sys.exit = lambda *a: None
    try:
        g = runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.exit = exit_
    rows = [{k: v for k, v in r.items()} for r in g["rows"]]
    (OUT / "takeover_rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "winding_stored.json").write_text((NB / "results/laneK6/winding_stored.json").read_text())
    e9 = json.loads((NB / "results/laneK5S1bA/e9_box_shape.json").read_text())
    keep = {k: e9[k] for k in ("L_series_aspect2", "hypercubic", "validation")}
    keep["elements"] = {s: {k: v for k, v in e.items() if k != "elements"} for s, e in e9["elements"].items()}
    (OUT / "e9_box_shape.json").write_text(json.dumps(keep, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
