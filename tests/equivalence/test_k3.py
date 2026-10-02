"""The K3 / I.4-I.7 analysis code against frozen notebook numbers (tests/reference/k3, make_k3_reference.py).

* analysis: every ensemble of the K3 claims analysed from the derived series (data/derived/k3) equals the notebook's
  analysis of the raw archive chains -- means, errors, block lengths: bit-identical up to TOL_ANALYSIS (relative);
* exact: the exact 2^3 Hubbard-Stratonovich numbers and the all-orders bag references (TOL_EXACT, relative);
* chains: prefixes of the exact-determinant Metropolis and RHMC runs bit-identical to the notebook's;
* channels: the independent real-flavour Wick evaluation equals the notebook's reference (TOL_EXACT).
"""

import json
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from k3_cases import KEYS, ensembles, stochastic8  # noqa: E402

from masspairing.analysis import k3, k3_exact  # noqa: E402
from masspairing.data import DERIVED  # noqa: E402
from masspairing.measure import ChannelMeasure  # noqa: E402

REF = HERE.parent / "reference" / "k3"
TOL_ANALYSIS = 1e-12
TOL_EXACT = 1e-12


def need(name):
    p = REF / name
    if not p.exists():
        pytest.skip(f"no frozen reference {name}")
    return p


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare_row(ref, r):
    assert ref["ntraj"] == r["ntraj"] and ref["blen"] == r["blen"]
    worst = rel(ref["acceptance"], r["acceptance"])
    for k in KEYS:
        if k not in r or k not in ref:
            continue
        a, b = ref[k], r[k]
        worst = max(worst, rel(a[0], b["mean"]), rel(a[1], b["err"]))
        if len(a) > 2:
            assert a[2] == b["corner"], k
    return worst


def test_analysis():
    ref = json.loads(need("analysis.json").read_text())
    worst = {}
    for name, spec in ensembles().items():
        worst[name] = compare_row(ref[name], k3.ensemble_of(spec))
    for name, case in stochastic8().items():
        parts, meta = [], None
        for rel_, k0, k1 in case["clean"]:
            ((s, kk),), metas = k3.streams([(rel_, k0)])
            meta = meta or metas[0]
            parts.append(k3.window(s, kk, None if k1 is None else k1 - metas[0]["k0_stored"]))
        r = k3.analyse(k3.concat(parts), meta)
        worst[name] = compare_row(ref[name], r)
        o_mean = float(np.mean(parts[0]["ts_chi10"]))
        worst[name] = max(worst[name], rel(o_mean, ref[name]["o_stream_mean_chi10"]))
    bad = {k: v for k, v in worst.items() if v > TOL_ANALYSIS}
    assert not bad, bad


def test_exact():
    ref = json.loads(need("exact.json").read_text())
    for name, (g10, g6) in (("interior", (0.4, 0.1)), ("edge+", (0.5, 0.5))):
        geom = k3_exact.model_2x3(g10, g6).geom
        r = k3_exact.hs_exact(geom, geom.K0.toarray())
        for k in ("Z_over_Z0", "sum_s2", "sum_term"):
            assert rel(r[k], ref[name][k]) < TOL_EXACT, (name, k)
        assert rel(float(r["s1"].sum()), ref[name]["s1"]) < TOL_EXACT
    gl = k3_exact.model_2x3(0.3, 0.0, quads="links").geom
    z = k3_exact.hs_exact(gl, gl.K0.toarray(), sigma=np.sqrt(1.0 / (2.0 * gl.inv4g)))["Z_over_Z0"]
    assert rel(z, ref["links"]["Z_over_Z0"]) < TOL_EXACT
    bags = DERIVED / "k3" / "bags_2x2x2.json"
    if bags.exists():
        mine = json.loads(bags.read_text())
        for name in ("interior", "edge+", "pure"):
            nb = ref["bags"][name]
            assert mine[name]["n_configs"] == nb["n_configs"]
            assert rel(mine[name]["Z_over_Z0"], nb["Z_over_Z0"]) < TOL_EXACT, name
            assert rel(mine[name]["sum_term"], nb["sum_term"]) < 1e-11, name


PREFIX = {
    "wedge_2x4_metropolis": 100,
    "wedge_2x4_rhmc": 10,
    "links_2x4_metropolis": 100,
    "links_2x4_rhmc": 10,
    "source_metropolis_h0.1": 260,
    "source_rhmc_h0.1": 10,
    "wedge_2x3_hmc": 20,
}


@pytest.mark.parametrize("name", list(PREFIX))
def test_chains(name):
    ref = np.load(need("chains.npz"))
    new = k3_exact.run_chain(name, PREFIX[name])
    keys = [k.split("/", 1)[1] for k in ref.files if k.startswith(name + "/")]
    assert keys
    for k in keys:
        a = ref[f"{name}/{k}"]
        b = np.asarray(new[k], float)
        assert a.shape == b.shape, (k, a.shape, b.shape)
        assert np.array_equal(a, b), (k, float(np.max(np.abs(a - b))))


def test_channels():
    ref = json.loads(need("channels.json").read_text())
    F, model = k3_exact.load_config("c072_L4_y2.41_g0.1_g60", cg_tol=1e-9)
    ms = ChannelMeasure(model, exact=True, bc_signs=False)
    r = k3_exact.channel_reference(ms, model.D.dense_S(F), bc_signs=False)
    for k in ("chi10_corners", "chi6_corners"):
        assert np.max(np.abs(np.asarray(ref[k]) - r[k])) < TOL_EXACT * np.max(np.abs(ref[k])), k
    for k in ("E10", "E6"):
        assert rel(ref[k], r[k]) < TOL_EXACT, k
