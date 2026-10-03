"""The k7 analyses (masspairing.analysis.t3a, fss, calibration) and the brute-force codes of scripts/derive_k7.py
against the notebook (tests/reference/k7, frozen by tests/equivalence/make_k7_reference.py at notebook commit 3c4a02c).

Tolerance: bit identity for everything (the derived series are the archive's series; the estimators keep the notebook's
arithmetic order), except the chain summaries of the calibration, compared to 1e-12 relative. Renamed keys: the
notebook's "c083_alive" / "c083" are "crossing_persists" / "crossing"; its "m_gate_open" is not carried.
"""

import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

from masspairing.analysis import t3a
from masspairing.analysis.calibration import point_summary
from masspairing.analysis.fss import discrimination_table
from masspairing.data import derived, load_chain

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "tests" / "reference" / "k7"
RENAME = {"c083_alive": "crossing_persists", "c083": "crossing"}
SKIP = {"m_gate_open"}


def ref(name):
    return json.loads((REF / name).read_text())


def compare(a, b, path="", tol=0.0, out=None):
    """Recursive comparison of new output a against reference b; returns the largest relative deviation."""
    out = [0.0] if out is None else out
    if isinstance(b, dict):
        for k, v in b.items():
            if k in SKIP:
                continue
            ka = RENAME.get(k, k)
            assert ka in a, f"{path}/{ka} missing"
            compare(a[ka], v, f"{path}/{k}", tol, out)
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(b, float) and not isinstance(b, bool):
        if math.isnan(b):
            assert a is None or math.isnan(a), path
            return out[0]
        assert isinstance(a, (int, float)), path
        dev = abs(a - b) / max(abs(b), 1e-300) if a != b else 0.0
        out[0] = max(out[0], dev)
        assert dev <= tol, f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"
    return out[0]


@pytest.fixture(scope="module")
def analyses():
    pilot, new = t3a.load_records("pilot"), t3a.load_records("sharpened")
    P = t3a.analyse_pilot(pilot)
    S = t3a.analyse_sharpened(pilot, new, P)
    return pilot, new, P, S


def test_t3a_pilot(analyses):
    _, _, P, _ = analyses
    assert compare(t3a.jsonable(P["verdict"]), ref("t3a_pilot_verdict.json")) == 0.0
    assert compare(t3a.jsonable(P["fss"]), ref("t3a_pilot_fss.json")) == 0.0
    assert compare(t3a.jsonable(P["labels"]), ref("t3a_pilot_labels.json")["chains"]) == 0.0


def test_t3a_sharpened(analyses):
    _, _, _, S = analyses
    assert compare(t3a.jsonable(S["verdict"]), ref("t3a_sharpened_verdict.json")) == 0.0
    assert compare(t3a.jsonable(S["fss"]), ref("t3a_sharpened_fss.json")) == 0.0
    assert compare(t3a.jsonable(S["labels"]), ref("t3a_sharpened_labels.json")["chains"]) == 0.0


def test_k73_discrimination():
    new, old = discrimination_table(), ref("k73_discrimination.json")
    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        compare(a, b)


def test_calibration_points():
    old = ref("calib_points.json")
    seen = set()
    for p in sorted(derived("k7", "calib").glob("*.npz")):
        r = point_summary(*load_chain(p))
        key = f"{r['L']}|{round(r['kappa'], 4)}|{round(r['y'], 4)}"
        seen.add(key)
        o = old[key]
        assert (r["ntraj"], r["blen"]) == (o["ntraj"], o["blen"])
        new = dict(
            acceptance=r["acceptance"],
            tau_max=r["tau_max"],
            Sigma_stag_abs=[r["Sigma_stag_abs"]["mean"], r["Sigma_stag_abs"]["err"]],
            chi_Sigma_stag=[r["chi_Sigma_stag"]["mean"], r["chi_Sigma_stag"]["err"]],
            xi2_stag=[r["xi2_stag"]["mean"], r["xi2_stag"]["err"]],
        )
        compare(new, {k: o[k] for k in new}, key, tol=1e-12)
    assert seen == set(old)


def test_brute_force_codes():
    spec = importlib.util.spec_from_file_location("derive_k7", ROOT / "scripts" / "derive_k7.py")
    DK = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DK)
    old = np.load(REF / "mc_short.npz")
    runs = {
        "metropolis_y1.2": DK.metropolis(1.2, 30, 3),
        "metropolis_y2.5": DK.metropolis(2.5, 30, 6),
        "rhmc_y1.2": DK.rhmc(1.2, 10, 4, ntherm=5),
        "hasenbusch_y2.5": DK.rhmc_hasenbusch(2.5, 10, 7, ntherm=5),
        "sabotage_y2.5": DK.rhmc_hasenbusch(2.5, 10, 8, ntherm=5, drop_ratios=True),
    }
    for name, res in runs.items():
        for k, v in res.items():
            assert np.array_equal(v, old[f"{name}/{k}"]), (name, k)
