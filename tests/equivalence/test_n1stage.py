"""The N1 stage analyses (masspairing.analysis.stage0/stage1/stage1b/stage1c) against the notebook's frozen outputs.

References (tests/reference/n1stage/, copied from the research notebook at 3c4a02c, absolute paths stripped):
  rows.json, verdict.json, controls.json           stage-1 reader, verdict, M trigger, K4 criteria and controls
  rows_1b.json, verdict_1b.json, controls_1b.json  stage-1b reader, y = 3.0 clauses, T1-T5, S7', integrity, E3, controls
  recalc_E1_E2_E8.json                             the recorded stage-1b tables
  alpha_summary.json, alpha_summary_1b.json        the alpha_L readouts
  rows_1c.json, verdict_1c.json, sensitivity_1c.json   stage-1c rows, integrity, replica consistency, y = 3.0 clauses
                                                   per replica / pooled, onset fits, controls, recorded sensitivity
                                                   (notebook commit fd0c934; make_n1stage1c_reference.py)
The derived chain files carry exactly the archive's time series, so the readers are bit-identical; the alpha readout
works on the corner blocks compressed to the light / heavy subspaces (U^dag G U instead of P G P, equal Frobenius
norms), so it agrees to rounding (max 3.4e-13 relative). Tolerance: 1e-12 relative, absolute for entries below
1e-12 (the rounding residues of exact zeros, e.g. the p-even part of the free massless block on the 6^4 pppa box);
fields that name files and the unprojected readout (rd_all, massed_all), which no claim uses, are not compared.
"""

import json
import math
import pathlib

import pytest

from masspairing.analysis import stage0, stage1, stage1b, stage1c
from masspairing.analysis.n1stage import to_json

REF = pathlib.Path(__file__).resolve().parents[1] / "reference" / "n1stage"
TOL = 1e-12
FLOOR = 1e-12  # entries below are rounding residues of exact zeros (e.g. the p-even part of a free massless block)
SKIP = {"file", "name", "rd_all", "massed_all", "selftest", "md5_matches_K4Z", "md5_matches_delivered"}
SKIP_1C = {"file", "name", "origin", "reading"}  # file names and descriptive text; every number and flag is compared


def ref(name):
    return json.loads((REF / name).read_text())


def compare(a, b, path="", skip=SKIP, out=None):
    """Recursive comparison; returns [(relative deviation, path)] for numeric leaves, raises on structural mismatch."""
    out = [] if out is None else out
    if isinstance(b, dict):
        assert isinstance(a, dict), path
        for k, v in b.items():
            if k in skip:
                continue
            assert k in a, f"{path}/{k} missing"
            compare(a[k], v, f"{path}/{k}", skip, out)
    elif isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", skip, out)
    elif isinstance(b, bool) or b is None or isinstance(b, str):
        assert a == b or (b is None and isinstance(a, float) and not math.isfinite(a)), f"{path}: {a!r} != {b!r}"
    else:
        a, b = float(a), float(b)
        if math.isnan(b) or math.isinf(b):
            assert a == b or (math.isnan(a) and math.isnan(b)), f"{path}: {a} != {b}"
        else:
            out.append((abs(a - b) / abs(b) if abs(b) >= FLOOR else abs(a - b), path))
    return out


def worst(devs):
    return max(devs, default=(0.0, ""))


@pytest.fixture(scope="module")
def s1():
    return stage1.stage1_rows()


@pytest.fixture(scope="module")
def s1b():
    return stage1b.analyse()


def test_stage1_rows(s1):
    got = {r["name"]: r for r in to_json(s1)}
    devs = []
    for r in ref("rows.json"):
        compare(got[r["name"]], r, r["name"], out=devs)
    w = worst(devs)
    print(f"stage-1 rows: {len(devs)} numbers, max relative deviation {w[0]:.1e} ({w[1]})")
    assert w[0] <= TOL, w


def test_stage1_verdict_k4(s1):
    got = to_json(
        {
            "verdict_y2.41": stage1.verdict(s1, 2.41),
            "verdict_y3.0": stage1.verdict(s1, 3.0),
            "M": stage1.m_trigger(s1),
            "K4": stage1.k4(s1),
        }
    )
    w = worst(compare(got, ref("verdict.json")))
    print(f"stage-1 verdict, M, K4: max relative deviation {w[0]:.1e}")
    assert w[0] <= TOL, w


def test_stage1_controls(s1):
    got = to_json(stage1.controls(s1))
    w = worst(compare(got, ref("controls.json")))
    assert w[0] <= TOL, w
    assert got["selftest"]["ok"] and ref("controls.json")["selftest"]["rc"] == 0
    assert stage0.selftest()


def test_stage1b(s1b):
    rows, res = s1b
    rj = stage1b.to_json(
        {
            "|".join(str(x) for x in k): {kk: vv for kk, vv in r.items() if not kk.endswith("_blocks")}
            for k, r in rows.items()
        }
    )
    devs = compare(rj, ref("rows_1b.json"), skip=SKIP)
    w = worst(devs)
    print(f"stage-1b rows: {len(devs)} numbers, max relative deviation {w[0]:.1e} ({w[1]})")
    assert w[0] <= TOL, w
    got = stage1b.to_json(res)
    fz = ref("verdict_1b.json")
    for tag, rec in fz["integrity"].items():
        assert got["integrity"][tag]["md5_matches_delivered"] == rec["md5_matches_K4Z"] is True
        assert got["integrity"][tag].get("seam_ok") == rec.get("seam_ok")
    w = worst(compare(got, fz, skip=SKIP))
    print(f"stage-1b verdict, T1-T5, S7', integrity, E3: max relative deviation {w[0]:.1e} ({w[1]})")
    assert w[0] <= TOL, w


def test_stage1b_tables_controls(s1b):
    rows, _ = s1b
    got = stage1b.to_json(dict(E1=stage1b.e1_table(rows), E2=stage1b.e2_table(rows), E8=stage1b.e8_table(rows)))
    w = worst(compare(got, ref("recalc_E1_E2_E8.json")))
    print(f"E1/E2/E8: max relative deviation {w[0]:.1e} ({w[1]})")
    assert w[0] <= TOL, w
    st = stage1b.to_json(stage1b.selftest(rows))
    w = worst(compare(st, ref("controls_1b.json")))
    assert w[0] <= TOL, w


@pytest.mark.parametrize("tag,name", [("S1", "alpha_summary.json"), ("S1b", "alpha_summary_1b.json")])
def test_alpha(tag, name):
    recs = stage1.alpha_records(tag)
    got = {r["file"]: r for r in to_json(recs)}
    devs = []
    keys = (
        "rd_L",
        "rd_R",
        "free",
        "massed_L",
        "floor_m",
        "swap_sigma",
        "tau_B",
        "tau_eff",
        "infl",
        "proj_err",
        "n",
        "p1",
    )
    for r in ref(name):
        fname = pathlib.Path(r["file"]).name
        g = got[fname]
        compare({k: g[k] for k in keys}, {k: r[k] for k in keys}, fname, out=devs)
    w = worst(devs)
    print(f"alpha {tag}: {len(devs)} numbers, max relative deviation {w[0]:.1e} ({w[1]})")
    assert w[0] <= TOL, w


@pytest.fixture(scope="module")
def s1c():
    return stage1c.analyse(with_sensitivity=True)


def test_stage1c(s1c):
    """Stage 1c: every number of the notebook's frozen rows, verdict and sensitivity, bit-identical."""
    _, tagrow, res = s1c
    devs = compare(stage1c.rows_json(tagrow), ref("rows_1c.json"), skip=SKIP_1C)
    got = stage1c.to_json(stage1c.strip({k: v for k, v in res.items() if k != "sensitivity"}))
    devs += compare(got, ref("verdict_1c.json"), skip=SKIP_1C)
    devs += compare(stage1c.to_json(res["sensitivity"]), ref("sensitivity_1c.json"), skip=SKIP_1C)
    w = worst(devs)
    print(f"stage-1c rows, verdict, sensitivity: {len(devs)} numbers, max relative deviation {w[0]:.1e} ({w[1]})")
    assert len(devs) > 3000 and w[0] <= TOL, w
    assert all(
        r["file"] == ref("verdict_1c.json")["integrity"][t]["file"]
        for t, r in got["integrity"].items()
        if t != "all_ok"
    )
