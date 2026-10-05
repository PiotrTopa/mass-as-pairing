#!/usr/bin/env python3
"""K5.10 -- 0+1D exact check of "no light mass term without symmetry breaking": 8 light + 8 heavy Majoranas with
Fidkowski-Kitaev quartics and a heavy mass M (256 states). The pre-registered verdict is printed as written (VOID: the
zero-sabotage control S1 did not reach its absolute threshold), then the facts: with the residual symmetry T' exact
every light mass-type Green's-function part vanishes; one T'-breaking term creates one; a light gap induced only
through the heavy half is symmetric yet scales like 1/M. Pure computation (about 20 s)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.algebra import ed01 as E
from masspairing.claimcheck import Check

c = Check("K5.10")
res = E.analyse()
v, s = res["verdict"], res["sym"]

# 0. model and symmetries
e = np.linalg.eigvalsh(E.fk(E.majoranas(4), E.w_sign()))
c.item(
    "Cayley (Fidkowski-Kitaev) quartic on 8 Majoranas: ground energy, gap (spectrum -14, 0 x 8, 2 x 7)",
    (e[0], e[1] - e[0]),
    abs(e[0] + 14) < 1e-10 and e[1] - e[0] > 13.9,
    "{0[0]:.0f}, gap {0[1]:.0f}",
)
c.item(
    "T (antiunitary, Majoranas even) and R_b leave W, X invariant; M is odd under both",
    True,
    all(
        s[k] for k in ("T_majoranas_even", "T_Wa", "T_X", "T_Mb_odd", "R_comm_Wa", "R_comm_Wb", "R_comm_X", "R_anti_Mb")
    ),
)
c.item(
    "T' = R_b T keeps M, keeps every light bilinear odd, is broken by Phi",
    True,
    s["Tp_Mb_even"] and s["Tp_light_bilinears_odd"] and s["Tp_Phi_broken"],
)

# 1. the pre-registered clauses, as written
w = v["Z"]["worst"]
c.item(
    "Z PASS: min ground-state split; max |<i a_k a_l>|; max |mass-type part of light G| (< 1e-10); min "
    "Delta_L(100)/Delta_L(10)",
    (w["split"], w["bil"], w["even"], min(v["Z"]["ratio_100_10"].values())),
    v["Z"]["ok"],
    "{0[0]:.2f}; {0[1]:.1e}; {0[2]:.1e}; {0[3]:.3f}",
)
c.item(
    "Z' (no light quartic) FAILS its absolute threshold 1e-10: max |mass-type part| (at M = 100)",
    v["Zp"]["worst"]["even"],
    not v["Zp"]["ok"] and 1e-10 < v["Zp"]["worst"]["even"] < 1e-6,
    "{:.1e}",
)
s1 = min(v["S1"]["min_even_M_pos"].values())
c.item(
    "S1 (Phi = 0.3) does NOT fire at its threshold 1e-3: min mass-type part (the FK gap is 14 U, not O(U))",
    s1,
    not v["S1"]["ok"] and 1e-5 < s1 < 1e-3,
    "{:.1e}",
)
c.item(
    "S2 PASS: Higgs-like Phi with U_a = 0 gives a seesaw light gap, d ln Delta_L / d ln M in [-1.2, -0.8]",
    v["S2"]["slope"],
    v["S2"]["ok"],
    "{:.3f}",
)
c.item("pre-registered verdict: VOID (Z/Z' are void unless S1 and S2 fire; S1 did not)", "VOID", not v["PASS"])

# 2. post hoc, labelled
zr = w["even"]
c.item(
    "post hoc: zeros vs sabotage, max mass-type part with T' exact vs min with Phi = 0.3 (ratio)",
    (zr, s1, s1 / zr),
    s1 / zr > 1e9,
    "{0[0]:.1e} vs {0[1]:.1e} ({0[2]:.1e})",
)
eo = E.even_odd_ratio(E.hamiltonian(Ua=0, Ub=1, lam=1.0, M=100.0))
c.item(
    "post hoc: the Z' excess is round-off -- even/odd part at its worst point (lambda = 1, M = 100)",
    eo,
    eo < 1e-8,
    "{:.1e}",
)
zp = res["Zp"]
for lam, rows in zp.items():
    c.record(
        f"Z' light gap Delta_L(M), lambda = {lam}",
        ", ".join(f"{float(M):g} -> {r['gap_L']:.4f}" for M, r in rows.items()),
    )
sl = v["Zp"]["slopes"]
c.item(
    "post hoc: a SYMMETRIC light gap with seesaw-like scaling (mass-type part zero): Delta_L(0) -> Delta_L(100) at "
    "lambda = 0.5, 1; d ln Delta_L / d ln M over M = 10 ... 100",
    (zp["0.5"]["0.0"]["gap_L"], zp["0.5"]["100.0"]["gap_L"], zp["1.0"]["0.0"]["gap_L"], zp["1.0"]["100.0"]["gap_L"])
    + tuple(sl.values()),
    all(-1.25 < x < -0.95 for x in sl.values()),
    "{0[0]:.3f} -> {0[1]:.4f}, {0[2]:.2f} -> {0[3]:.3f}; slopes {0[4]:.2f}, {0[5]:.2f}",
)
d = v["direction"]
lo = min(min(r.values()) for r in d.values())
c.item(
    "post hoc: with the light quartic on (Z), Delta_L(M)/Delta_L(0) stays in [min, 1] -- the light symmetric gap "
    "moves by <= 6 %, downward, for either sign of lambda",
    lo,
    lo > 0.93 and all(r["100.0"] <= 1.0 + 1e-12 for r in d.values()),
    "{:.3f}",
)

# 3. the zeros need T' only
n_unique, worst_sym, best_brk = E.random_tprime()
c.item(
    "post hoc: 12 random T'-invariant Hamiltonians -- number with a unique ground state; max light bilinear / "
    "mass-type part",
    (n_unique, worst_sym),
    n_unique >= 10 and worst_sym < 1e-10,
    "{0[0]}/12; {0[1]:.1e}",
)
c.item(
    "post hoc: one T'-odd bilinear 0.3 i a_1 b_1 added -- min mass-type part over the trials",
    best_brk,
    best_brk > 1e-6,
    "{:.1e}",
)
c.done()
