"""Equivalence of the n1inst analysis code with the notebook.

Compared against notebook outputs frozen in tests/reference/n1inst/ (make_n1inst_reference.py) and the verbatim
notebook baselines in data/derived/free/:
  * analysis.free.free_point             vs the frozen baseline rows (<= 1e-12, every key)
  * analysis.n1inst.metropolis_woodbury  vs the notebook's exact-determinant Metropolis (bit-identical, 305 sweeps)
  * metropolis_pfaffian / rhmc_series    vs the notebook's |Pf| Metropolis and RHMC chains (bit-identical prefixes)
  * takeover_row                         vs the notebook's composite-takeover rows (<= 1e-12)
  * winding_record                       vs the notebook's 39 stored loop records (<= 1e-9; sparse LU rounding)
  * free_phi_L, stabiliser_elements      vs the notebook's box-shape series and element counts (<= 1e-12)
"""

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from masspairing.analysis import n1inst as X  # noqa: E402
from masspairing.analysis.free import FREE_DIR, free_lookup, free_point, max_deviation  # noqa: E402
from masspairing.data import derived, load_chain  # noqa: E402

REF = ROOT / "tests" / "reference" / "n1inst"


def ref(name):
    p = REF / name
    if not p.exists():
        pytest.skip(f"no frozen reference {name}")
    return p


@pytest.mark.parametrize("box", [(4, 4, "aaaa", 0.5), (4, 4, "pppa", 2.0), (4, 8, "aaaa", 0.5), (4, 4, "pppa", 0.0)])
def test_free_point(box):
    rows = json.loads((FREE_DIR / "free_baselines.json").read_text())
    assert max_deviation(free_point(*box), free_lookup(rows, *box)) <= 1e-12


def test_metropolis_woodbury_prefix():
    nb = np.load(ref("met_woodbury_prefix.npz"))
    r = X.i10_metropolis(305, 1610)
    for k in nb.files:
        assert np.array_equal(nb[k], np.asarray(r[k])), k


def test_i8_prefix():
    nb = np.load(ref("i8_prefix.npz"))
    new = {"met": X.i8_metropolis(206, 1400), "rhmc": X.i8_rhmc(4, 1402, False), "sabotage": X.i8_rhmc(4, 1403, True)}
    for k in nb.files:
        name, key = k.split("__", 1)
        assert np.array_equal(nb[k], np.asarray(new[name][key])), k


def test_takeover_rows():
    nb = json.loads(ref("takeover_rows.json").read_text())
    new = {}
    for f in sorted(derived("n1inst", "takeover").glob("*.npz")):
        r = X.takeover_row(*load_chain(f))
        new.setdefault((r["L"], r["y"], r["h"]), []).append(r)
    pairs = {"phiT": "phiT_h", "heavy": "heavy_h", "light": "light_h"}
    assert len(nb) == sum(len(v) for v in new.values())
    for r in nb:
        cands = new[(r["L"], r["y"], r["h"])]
        best = min(cands, key=lambda c: abs(c["phiT_h"] * c["h"] - r["phiT"]))
        assert best["n"] == r["n"]
        for a, b in pairs.items():
            assert abs(best[b] * r["h"] - r[a]) <= 1e-12
        assert abs(best["ephiT_h"] * r["h"] - r["ephiT"]) <= 1e-12
        assert abs(best["ratio"] - r["ratio"]) <= 1e-9


def test_winding_records():
    nb = {r["file"]: r for r in json.loads(ref("winding_stored.json").read_text())["records"]}
    new = json.loads(derived("n1inst", "K6_winding_stored.json").read_text())["records"]
    assert len(new) == len(nb) == 39
    for r in new:
        o = nb[r["file"]]
        for k in ("f_M", "f_M3", "N_odd", "alpha", "normO1", "normO3", "normE1", "normE3"):
            assert abs(r[k] - o[k]) <= 1e-9 * max(1.0, abs(o[k])), (r["file"], k)
        for k in ("quantised", "reading", "det_half_turns", "phase", "scored", "near_critical"):
            assert r[k] == o[k], (r["file"], k)


def test_box_shape():
    nb = json.loads(ref("e9_box_shape.json").read_text())
    new = json.loads(derived("n1inst", "K52_box_shape.json").read_text())
    old = {(r["L"], r["Lt"], r["h"]): r["phi_L"] for r in nb["L_series_aspect2"] + nb["hypercubic"]}
    for r in new["aspect2"] + new["hypercubic"]:
        assert abs(r["phi_L"] - old[(r["L"], r["Lt"], r["h"])]) <= 1e-12
    assert abs(X.free_phi_L(4, 8, 2.0) - old[(4, 8, 2.0)]) <= 1e-12
    for shape in ((4, 4, 4, 4), (4, 4, 4, 8)):
        e = nb["elements"]["x".join(map(str, shape))]
        nG, rows = X.stabiliser_elements(shape)
        flips = [r for r in rows if r["act"] < -0.5]
        assert (nG, len(rows), len(flips)) == (e["G"], e["stab"], e["n_flip"])
        assert all(r["moves_time"] for r in flips) == e["all_flips_move_time"]
