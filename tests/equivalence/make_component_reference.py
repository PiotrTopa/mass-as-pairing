#!/usr/bin/env python3
"""Freeze the notebook side of tests/equivalence/components.py (needs MASSPAIRING_NOTEBOOK=/path)."""

import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"]) / "src"))
from components import notebook_side  # noqa: E402

out = notebook_side()
np.savez_compressed(HERE.parent / "reference" / "components.npz", **out)
print(f"{len(out)} arrays")
