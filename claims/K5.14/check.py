"""K5.14: assumption (A4) of the general proposition by enumeration of the whole light corner algebra on 4^4.

(1) 4^4 pppp (light sector = the 16 corner momenta): under the 32-element point-group stabiliser of C_chi, 52 corner-
    string bilinears keep a light part, 20 of them antisymmetric, even in p and not mass-type; under its Z4 extension
    (64 elements) 20 survive, 12 antisymmetric even non-mass-type (4 LL+RR self-dual tensor, 8 LR): (A4) fails with
    the point group alone.
(2) with the one-site shifts (stabiliser 8192) only the identity and X13 Z0123 survive on pppp (symmetric kernels).
(3) 4^4 aaaa, full stabiliser: 10 of 496 survive -- identity, X13 Z0123 and eight antisymmetric strings, all odd in p
    (the four Gamma_mu and four three-link strings); no mass-type survivor on either box.
(4) corner algebra: X02 Z0123 = +-Gamma0Gamma1Gamma2Gamma3 anticommutes with every Gamma_mu (a mass string);
    X13 Z0123 commutes with every Gamma_mu (the taste chirality).
(5) sign fields of the 384 hypercubic maps: x mod 2 functions on periodic 4^4, 6^4; on aaaa 4^4, 6^4 the 360 maps with a
    reflection carry the boundary-slice factor (8^4 recorded from the notebook run).
About 4 min, 0.8 GB.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from masspairing.analysis import corner_invariants as CI
from masspairing.claimcheck import Check

c = Check("K5.14")
TOL = CI.TOL


def lab(r):
    return (r["S"], r["D"], "anti" if r["anti"] else "sym")


# ---- (1), (2) 4^4 pppp
info, rp = CI.enumerate_box((1, 1, 1, 1))
c.item(
    "4^4 pppp: bilinears; point group / stabiliser / flips; with shifts / stabiliser; max |[T_g, P_L]|",
    (len(rp), info["G_pg"], info["St_pg"], info["Fl_pg"], info["G_all"], info["St_all"], float(info["PL_commutes"])),
    len(rp) == 496
    and (info["G_pg"], info["St_pg"], info["Fl_pg"], info["G_all"], info["St_all"]) == (384, 32, 32, 98304, 8192)
    and info["PL_commutes"] < 1e-12,
)
s_pg, s_z4, s_all = (CI.survivors(rp, k) for k in ("light_pg", "light_pgz4", "light_all"))
a_pg = [r for r in s_pg if r["anti"]]
mass_pg = sorted(lab(r) for r in s_pg if r["mass_type"])
c.item(
    "pppp, point-group stabiliser (32): light survivors; antisymmetric (all even in p, not mass-type); "
    "mass-type survivors",
    (len(s_pg), len(a_pg), mass_pg),
    len(s_pg) == 52
    and len(a_pg) == 20
    and all(r["even_p"] and not r["mass_type"] for r in a_pg)
    and mass_pg == [((0, 1, 2, 3), (), "sym"), ((0, 2), (0, 1, 2, 3), "sym")],
)
a_z4 = [r for r in s_z4 if r["anti"]]
llrr = sorted(lab(r)[:2] for r in a_z4 if r["taste"] == "LL+RR")
c.item(
    "pppp, point group + Z4 (64): survivors; antisymmetric even-in-p non-mass-type; their LL+RR strings; LR count",
    (len(s_z4), len(a_z4), llrr, sum(r["taste"] == "LR" for r in a_z4)),
    len(s_z4) == 20
    and len(a_z4) == 12
    and all(r["even_p"] and not r["mass_type"] for r in a_z4)
    and llrr == [((0, 1, 2), (0, 3)), ((0, 2, 3), (1, 2)), ((1,), (1, 2)), ((3,), (0, 3))]
    and sum(r["taste"] == "LR" for r in a_z4) == 8,
)
c.item(
    "pppp, full stabiliser with shifts (8192): survivors (symmetric kernels only); "
    "the antisymmetric invariants of (1) vanish",
    [lab(r) for r in s_all],
    sorted(lab(r) for r in s_all) == [((), (), "sym"), ((1, 3), (0, 1, 2, 3), "sym")]
    and all(r["light_all"] < TOL for r in a_pg),
)

# ---- (3) 4^4 aaaa
info, ra = CI.enumerate_box((-1, -1, -1, -1))
s_pg, s_z4, s_all = (CI.survivors(ra, k) for k in ("light_pg", "light_pgz4", "light_all"))


def cnt(ss):
    return len(ss), sum(r["anti"] and r["even_p"] and not r["mass_type"] for r in ss)


c.record(
    "aaaa: light survivors (all, antisymmetric even non-mass): point group, + Z4, full",
    (cnt(s_pg), cnt(s_z4), cnt(s_all)),
)
anti = [r for r in s_all if r["anti"]]
gam = sorted(lab(r)[:2] for r in anti if r["orbit_all"] == 4 and r["n_anti_gamma"] == 3)
three = sorted(lab(r)[:2] for r in anti if r["orbit_all"] == 16 and r["n_anti_gamma"] == 1 and len(r["D"]) == 3)
c.item(
    "aaaa, full stabiliser: survivors of 496; symmetric ones; "
    "antisymmetric (all odd in p, not mass-type): Gamma_mu, three-link",
    (len(s_all), sorted(lab(r) for r in s_all if not r["anti"]), gam, three),
    len(ra) == 496
    and info["St_all"] == 8192
    and len(s_all) == 10
    and sorted(lab(r) for r in s_all if not r["anti"]) == [((), (), "sym"), ((1, 3), (0, 1, 2, 3), "sym")]
    and len(anti) == 8
    and all(not r["even_p"] and not r["mass_type"] for r in anti)
    and gam == sorted(CI.corner_facts()["gamma_strings"])
    and len(three) == 4,
)
c.item(
    "no mass-type string survives the full stabiliser on pppp or aaaa",
    sum(r["mass_type"] for r in CI.survivors(rp, "light_all") + s_all),
    not any(r["mass_type"] for r in CI.survivors(rp, "light_all") + s_all),
)

# ---- (4) corner algebra
f = CI.corner_facts()
c.item(
    "X02 Z0123: anticommutes with every Gamma_mu, = +-Gamma0Gamma1Gamma2Gamma3, symmetric",
    (f["X02Z0123_anticommutes_all"], bool(f["X02Z0123_prop_G0G1G2G3"]), f["X02Z0123_symmetric"]),
    f["X02Z0123_anticommutes_all"] and f["X02Z0123_prop_G0G1G2G3"] and f["X02Z0123_symmetric"],
)
c.item(
    "X13 Z0123: commutes with every Gamma_mu, symmetric (the taste chirality)",
    (f["X13Z0123_commutes_all"], f["X13Z0123_symmetric"]),
    f["X13Z0123_commutes_all"] and f["X13Z0123_symmetric"],
)

# ---- (5) sign fields
sf = {(L, tag): CI.sign_field_parity((L,) * 4, (b,) * 4) for L in (4, 6) for b, tag in ((1, "pppp"), (-1, "aaaa"))}
c.item(
    "sign fields (maps, x mod 2, x mod 2 times boundary slice) on 4^4, 6^4 pppp and aaaa",
    {f"{L}^4 {t}": tuple(v.values()) for (L, t), v in sf.items()},
    all(tuple(v.values()) == ((384, 384, 0) if t == "pppp" else (384, 24, 360)) for (_, t), v in sf.items()),
)
c.record("8^4 aaaa sign fields (notebook run, not recomputed here)", (384, 24, 360))
c.done()
