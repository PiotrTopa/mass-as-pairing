"""The clean runner reproduces the notebook runners chain by chain (tests/reference/chains, frozen notebook output).

Every ts_ series common to both files and the final configuration are compared; the tolerance is bit identity
(0.0) unless the case is listed in TOL with its stated bound.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from cases import CASES, COMMON  # noqa: E402

REF = ROOT / "tests" / "reference" / "chains"
spec = importlib.util.spec_from_file_location("run_chain", ROOT / "scripts" / "run_chain.py")
run_chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_chain)

TOL = {}  # case -> max relative difference allowed (0.0 = bit identical)

# series of the notebook's bond-product estimator, which the clean package does not carry (superseded by the
# complete 10-channel estimator, see docs/PLAN.md section 3)
DROPPED = {
    "ts_phi10_sq",
    "ts_phi10bar_sq",
    "ts_chi_bond",
    "ts_chi_bond_disc",
    "ts_chi_bond_ex1",
    "ts_chi_bond_ex2",
    "ts_E_bondprod",
}


def compare(ref, new):
    diffs = {}
    for k in ref.files:
        if k == "fields_final":
            b = new["fields_final"].reshape(-1)
        elif k in new.files:
            b = new[k]
        else:
            continue
        a = np.asarray(ref[k])
        b = np.asarray(b)
        assert a.shape == b.shape, (k, a.shape, b.shape)
        if a.dtype.kind in "biu":
            diffs[k] = float(np.sum(a != b))
        elif a.size:
            scale = max(np.max(np.abs(a)), 1e-300)
            diffs[k] = float(np.max(np.abs(a - b)) / scale)
        else:
            diffs[k] = 0.0
    return diffs


@pytest.mark.parametrize("name", sorted(CASES))
def test_chain_case(name, tmp_path):
    ref_file = REF / f"{name}.npz"
    if not ref_file.exists():
        pytest.skip(f"no frozen reference for {name}")
    _runner, _old, new = CASES[name]
    path = run_chain.run(run_chain.parse(COMMON + new + ["--out", str(tmp_path)]))
    ref = np.load(ref_file, allow_pickle=True)
    got = np.load(path, allow_pickle=True)
    diffs = compare(ref, got)
    missing = [k for k in ref.files if k.startswith("ts_") and k not in got.files and k not in DROPPED]
    assert not missing, f"series missing in the clean output: {missing}"
    worst = max(diffs.items(), key=lambda kv: kv[1])
    assert worst[1] <= TOL.get(name, 0.0), f"{name}: largest difference {worst}"
