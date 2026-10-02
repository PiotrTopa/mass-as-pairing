"""Equivalence of masspairing.algebra with the notebook modules (spin10, liealg, grassmann, eta, staggered and the
claim-check code of C040, C148, C152, C165): every array of algebra_cases.py against the frozen notebook side.

Tolerance: bit identity except the items in TOL (relative to the array's largest entry), where the arithmetic order
differs (onsite blocks via a different diagonal-extraction path; traces of matrix powers vs repeated products).
When MASSPAIRING_NOTEBOOK points at a notebook checkout, the notebook side is also recomputed live.
"""

import os
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from algebra_cases import clean_side  # noqa: E402

REF = HERE.parent / "reference" / "algebra" / "algebra.npz"
TOL = {"chars_W23": 1e-12}


@pytest.fixture(scope="module")
def new():
    return clean_side()


def _compare(ref, new):
    assert set(ref) == set(new), set(ref) ^ set(new)
    worst = {}
    for k in ref:
        a, b = np.asarray(ref[k]), np.asarray(new[k])
        if a.dtype == bool:
            a, b = a.astype(float), b.astype(float)
        assert a.shape == b.shape, (k, a.shape, b.shape)
        scale = max(float(np.max(np.abs(a))) if a.size else 0.0, 1e-300)
        worst[k] = float(np.max(np.abs(a - b))) / scale if a.size else 0.0
    bad = {k: v for k, v in worst.items() if v > TOL.get(k, 0.0)}
    assert not bad, bad
    return worst


def test_algebra_frozen(new):
    if not REF.exists():
        pytest.skip("no frozen algebra reference")
    ref = np.load(REF)
    worst = _compare({k: ref[k] for k in ref.files}, new)
    nonzero = {k: v for k, v in worst.items() if v}
    print(f"{len(worst)} arrays, bit-identical except {nonzero}")


@pytest.mark.skipif(not os.environ.get("MASSPAIRING_NOTEBOOK"), reason="needs a notebook checkout")
def test_algebra_live(new):
    sys.path.insert(0, str(pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"]) / "src"))
    from algebra_cases import notebook_side

    _compare(notebook_side(), new)
