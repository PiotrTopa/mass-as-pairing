"""Component-level equivalence with the notebook package: operators, patterns, projectors, actions, forces, Lanczos,
partial fractions, every measure (dense and stochastic), Pfaffians, symmetry classification and corner blocks.

Tolerance: bit identity for everything except the items in TOL (stated bound, relative to the array's largest entry).
"""

import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from components import clean_side  # noqa: E402

REF = HERE.parent / "reference" / "components.npz"
TOL = {}


@pytest.fixture(scope="module")
def both():
    if not REF.exists():
        pytest.skip("no frozen component reference")
    return np.load(REF), clean_side()


def test_components(both):
    ref, new = both
    assert set(ref.files) == set(new), set(ref.files) ^ set(new)
    worst = {}
    for k in ref.files:
        a, b = np.asarray(ref[k]), np.asarray(new[k])
        assert a.shape == b.shape, (k, a.shape, b.shape)
        scale = max(float(np.max(np.abs(a))) if a.size else 0.0, 1e-300)
        worst[k] = float(np.max(np.abs(a - b))) / scale if a.size else 0.0
    bad = {k: v for k, v in worst.items() if v > TOL.get(k, 0.0)}
    assert not bad, bad
