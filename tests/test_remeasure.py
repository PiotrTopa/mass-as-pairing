"""Re-measurement of stored configurations with the clean package.

The dense N1 measure of stored stage-0 (4^4) and stage-1 (6^4) configurations reproduces the values the production
chains stored for them; the sigma and link-field observables of stored final configurations reproduce the chains'
last rows. Tolerance: 1e-12 relative to each array's largest entry, with an absolute floor of 1e-18 (entries below 1e-6
are cancellation-dominated connected parts); the production measurements ran on other hosts and BLAS builds.
"""

import json

import numpy as np
import pytest

from masspairing.action import build_model
from masspairing.data import CONFIGS
from masspairing.lattice import Lattice, parse_bc
from masspairing.measure import N1Measure, scalar_observables
from masspairing.wedge import link_field_observables

TOL = 1e-12
N1 = ["n1_L4_y3_h2_aaaa", "n1_L4_y2.41_h0.5_aaaa", "n1_L6_y3_h2_aaaa", "n1_L6_y2.41_h2_aaaa"]


def load(name):
    z = np.load(CONFIGS / f"{name}.npz", allow_pickle=True)
    return z, json.loads(str(z["meta"]))


def lattice(chain):
    L, Lt = chain["L"], chain.get("Lt") or chain["L"]
    return Lattice((L, L, L, Lt), bc=parse_bc(chain.get("bc_str", "pppa")))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-6))


@pytest.mark.parametrize("name", N1)
def test_n1_dense_remeasure(name):
    z, info = load(name)
    ch = info["chain"]
    lat = lattice(ch)
    model = build_model(lat, ch["y"], ch["kappa"], ch["lam"], h=ch["h10"], pattern=ch["h10_pattern"])
    meas = N1Measure(model, exact=True, seed=0)
    for j in range(len(info["rows"])):
        if name.startswith("n1_L6") and j > 0:
            break
        got = meas.bilinears(z[f"fields_{j}"])
        keys = [k[len(f"row{j}_") :] for k in z.files if k.startswith(f"row{j}_")]
        keys = [k for k in keys if k in got]
        assert len(keys) > 40
        worst = max((rel(z[f"row{j}_{k}"], got[k]), k) for k in keys)
        assert worst[0] <= TOL, worst


@pytest.mark.parametrize("name", ["wedge_L6_y2.41_g0.1_edge-", "eps_L8_y2.41_afm"])
def test_final_configuration_bosonic(name):
    z, info = load(name)
    ch = info["chain"]
    lat = lattice(ch)
    g10, g6 = ch.get("g10", 0.0), ch.get("g6", 0.0)
    model = build_model(lat, ch["y"], ch["kappa"], ch["lam"], g10=g10, g6=g6, quads=ch.get("quads", "all"))
    fields = z["fields_0"]
    sigma, s = model.split(fields)
    got = scalar_observables(lat, sigma)
    if model.geom is not None:
        got.update(link_field_observables(model.geom, s))
    keys = [k[5:] for k in z.files if k.startswith("row0_")]
    assert all(k in got for k in keys), set(keys) - set(got)
    worst = max((rel(z[f"row0_{k}"], got[k]), k) for k in keys)
    assert worst[0] <= TOL, worst
