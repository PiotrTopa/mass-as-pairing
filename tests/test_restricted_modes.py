"""The light block of the source: the matrix-free plane-wave resolution equals the closed form (K5.15), and the
corner-algebra identities of the two four-link strings (K5.14)."""

from masspairing.analysis import corner_invariants as CI
from masspairing.analysis import restricted_modes as R


def test_light_block_closed_form():
    for shape in ((4, 4, 4, 4), (4, 4, 4, 8)):
        live, form = R.light_block(shape), R.light_block_formula(shape)
        assert live["max_formula_dev"] < 1e-12
        assert abs(live["frac"] - form["frac"]) < 1e-12
        assert (live["n_split"], live["n_source_free"]) == (form["n_split"], form["n_source_free"])
    assert R.light_block_formula((8, 8, 8, 8))["n_source_free"] == 1536
    assert abs(R.light_block_formula((8, 8, 8, 8))["frac"] - 0.36840643955) < 1e-9


def test_ir_class_on_6_is_the_split():
    assert R.ir_class_mask((6, 6, 6, 6)).sum() == 256 == R.light_block_formula((6, 6, 6, 6))["n_split"]


def test_corner_facts():
    f = CI.corner_facts()
    assert f["X02Z0123_anticommutes_all"] and f["X02Z0123_prop_G0G1G2G3"] and f["X02Z0123_symmetric"]
    assert f["X13Z0123_commutes_all"] and f["X13Z0123_symmetric"]


def test_sign_field_parity_4():
    assert CI.sign_field_parity((4,) * 4, (1,) * 4) == dict(maps=384, parity=384, parity_times_boundary=0)
    assert CI.sign_field_parity((4,) * 4, (-1,) * 4) == dict(maps=384, parity=24, parity_times_boundary=360)
