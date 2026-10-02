#!/usr/bin/env python3
"""Freeze the notebook side of tests/equivalence/algebra_cases.py (needs MASSPAIRING_NOTEBOOK=/path)."""

import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"]) / "src"))
from algebra_cases import notebook_side  # noqa: E402

out = notebook_side()
dst = HERE.parent / "reference" / "algebra" / "algebra.npz"
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst, **out)
print(f"{len(out)} arrays -> {dst}")
