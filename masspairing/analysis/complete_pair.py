"""Taste-projector coverage and the light pair susceptibility on volume pairs with a complete taste split (K5.13).

The light/heavy split of the taste projector exists on the momenta where every cos p_mu != 0
(``patterns.TasteProjector.n_boundary`` counts the others). On all-antiperiodic boxes p_mu = (2n+1) pi / L, so it fails
iff L = 2 mod 4: 4^4, 4^3 x 8, 8^4, 12^4 are complete, 6^4 is split on (4/6)^4 of the momenta.

``pair_rows`` reads chi_L = ts_chi_L_sum (= V <phi_L^2>) of the y = 3.0 chains at h in {0, 0.5, 1, 2}: 4^4 from stage 0
(cut: trajectories >= 150, the stage-0 reader's first 75 of 300 measurements), 8^4 pooled over stage 1 (>= 100),
stage-1b new trajectories (>= 1000) and the stage-1c replicas rA, rB (>= 100), 6^4 from stage 1 and stage 1b; and
forms e = d ln chi_L / d ln V on (4^4, 8^4) and on (6^4, 8^4), each against the free exponent of the same pair.
Errors: 10 blocks per chain (10 per member for the pooled 8^4 series), inflated to sigma_naive sqrt(2 tau_int) where
that is larger.
"""

from __future__ import annotations

import numpy as np

from ..lattice import Lattice
from ..patterns import TasteProjector
from .n1stage import N1DIR, exponent, free_lookup, free_tables, open_chain, tau_int

COVERAGE_BOXES = (
    (4, 4, 4, 4),
    (6, 6, 6, 6),
    (8, 8, 8, 8),
    (6, 6, 6, 12),
    (4, 4, 4, 8),
    (10, 10, 10, 10),
    (12, 12, 12, 12),
)
HS = ((0.0, ""), (0.5, "_h0.5"), (1.0, "_h1"), (2.0, "_h2"))
V4, V6, V8 = 256, 1296, 4096
FREE_FILES = ["free_L4_aaaa.json", "free_L8_aaaa.json", "free_baselines.json", "free_L4x8_aaaa.json"]


def coverage():
    """{box label: dict(V, boundary, split)} on all-antiperiodic boxes."""
    out = {}
    for shape in COVERAGE_BOXES:
        lat = Lattice(shape, bc=(-1,) * 4)
        tp = TasteProjector(lat)
        out["x".join(map(str, shape)) + " aaaa"] = dict(
            V=lat.V, boundary=tp.n_boundary, split=1 - tp.n_boundary / lat.V
        )
    return out


def blocked(x, nb):
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nb))


def chain_chi(path, tmin, nb=10):
    d, _ = open_chain(path)
    keep = np.asarray(d["ts_cfg_traj"]) >= tmin
    chi = np.asarray(d["ts_chi_L_sum"], float)[keep]
    m, s = blocked(chi, nb)
    tau = tau_int(chi)
    s = max(s, float(chi.std(ddof=1) / np.sqrt(len(chi)) * np.sqrt(2 * max(tau, 0.5))))
    return dict(n=int(keep.sum()), chi=m, err=s, tau=float(tau), series=chi)


def chain_path(stage, sub, L, tag, lt=None):
    name = f"L{L}" + (f"x{lt}" if lt else "") + f"_y3_k-0.01_g0_g60{tag}_chiral_bcaaaa.npz"
    return N1DIR / stage / sub / name


def pair_rows(inject_ssb=False):
    """Rows per h of chi_L/free on 4^4, 8^4 (pooled), 6^4 and the exponents e - e_free on (4,8) and (6,8).

    ``inject_ssb``: sabotage control -- the pooled 8^4 chi_L series is rescaled to chi_L(4^4) V_8/V_4, the SSB-scale
    growth e = 1 on (4^4, 8^4)."""
    free = free_tables(FREE_FILES)
    rows = {}
    for h, tag in HS:
        r4 = chain_chi(chain_path("S0", "L4", 4, tag), tmin=150)
        s8 = {"S1": chain_chi(chain_path("S1", "L8", 8, tag), tmin=100)}
        for name, path, tmin in (
            ("S1b_new", chain_path("S1b", "L8", 8, tag), 1000),
            ("S1c_rA", chain_path("S1c", f"L8_h{int(h)}_rA", 8, tag), 100),
            ("S1c_rB", chain_path("S1c", f"L8_h{int(h)}_rB", 8, tag), 100),
        ):
            if path.exists():
                s8[name] = chain_chi(path, tmin=tmin)
        r6 = {"S1": chain_chi(chain_path("S1", "L6", 6, tag), tmin=100)}
        p6 = chain_path("S1b", "L6", 6, tag)
        if p6.exists():
            r6["S1b_new"] = chain_chi(p6, tmin=1000)
        f4, f6, f8 = (free_lookup(free, L, L, "aaaa", h)["chi_L_sum"] for L in (4, 6, 8))
        ef48, ef68 = np.log(f8 / f4) / np.log(V8 / V4), np.log(f8 / f6) / np.log(V8 / V6)
        allser = np.concatenate([r["series"] for r in s8.values()])
        if inject_ssb:
            allser = allser * (r4["chi"] * V8 / V4) / allser.mean()
        mp, sp = blocked(allser, 10 * len(s8))
        taumax = max(r["tau"] for r in s8.values())
        sp = max(sp, float(allser.std(ddof=1) / np.sqrt(len(allser)) * np.sqrt(2 * max(taumax, 0.5))))
        e48, ee48 = exponent(r4["chi"], r4["err"], mp, sp, V4, V8)
        row = dict(
            h=h,
            free=dict(chi4=f4, chi6=f6, chi8=f8, e48=ef48, e68=ef68),
            L4=dict(n=r4["n"], chi=r4["chi"], err=r4["err"], over_free=r4["chi"] / f4),
            L8=dict(
                members={k: dict(n=r["n"], chi=r["chi"], err=r["err"], over_free=r["chi"] / f8) for k, r in s8.items()},
                n=len(allser),
                chi=mp,
                err=sp,
                over_free=mp / f8,
            ),
            L6={k: dict(n=r["n"], chi=r["chi"], err=r["err"], over_free=r["chi"] / f6) for k, r in r6.items()},
            e48=dict(e=e48, err=ee48, minus_free=e48 - ef48),
            e68={
                k: dict(zip(("e", "err"), exponent(r["chi"], r["err"], mp, sp, V6, V8), strict=True))
                for k, r in r6.items()
            },
        )
        for k in row["e68"]:
            row["e68"][k]["minus_free"] = row["e68"][k]["e"] - ef68
        rows[h] = row
    return rows


def l4x8_h2_over_free():
    """chi_L/free on the stage-0 4^3 x 8 aaaa chain at (3.0, h = 2) (the third complete box)."""
    r = chain_chi(chain_path("S0", "L4x8", 4, "_h2", lt=8), tmin=150)
    f = free_lookup(free_tables(FREE_FILES), 4, 8, "aaaa", 2.0)["chi_L_sum"]
    return r["chi"] / f, r["n"]
