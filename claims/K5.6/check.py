#!/usr/bin/env python3
"""K5.6 -- control S3: one 6^4 chain at (y = 3.0, h = 0.5) with the phase-flipped pattern (the un-phased plane pair),
read next to the free operator with the same pattern and the chiral chain at the same point.
Reads data/derived/n1stage/S3 and the stage-1 chiral chain (a few seconds)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import stage1
from masspairing.analysis.n1stage import N1DIR, free_lookup, free_tables, open_chain
from masspairing.claimcheck import Check

c = Check("K5.6")
S3 = N1DIR / "S3"
bl = free_tables(stage1.FREE_S1)
f = json.loads((S3 / "free_flipped_L6_aaaa_h0.5.json").read_text())
fc = free_lookup(bl, 6, 6, "aaaa", 0.5)
f0 = free_lookup(bl, 6, 6, "aaaa", 0.0)
c.record(
    "free flipped h=0.5: mL_eff, mR_eff", f"{np.round(f['mL_eff'], 4).tolist()}, {np.round(f['mR_eff'], 4).tolist()}"
)
c.record(
    "free chiral h=0.5: mL_eff, mR_eff", f"{np.round(fc['mL_eff'], 4).tolist()}, {np.round(fc['mR_eff'], 4).tolist()}"
)
c.item(
    "1. free K - 0.5 C_flip: light = heavy exactly; m(t*) and G_p0 (free massless m(t*))",
    (f["mL_eff"][2], f["mR_eff"][2], f["GL_p0"], f["GR_p0"], f0["mL_eff"][2]),
    abs(f["mL_eff"][2] - f["mR_eff"][2]) < 1e-9
    and abs(f["GL_p0"] - f["GR_p0"]) < 1e-9
    and abs(f["mL_eff"][2] - 0.4093) < 1e-3
    and abs(f["GL_p0"] - 0.9885) < 1e-3,
    "m {0[0]:.4f} = {0[1]:.4f}, G_p0 {0[2]:.4f} = {0[3]:.4f} (massless {0[4]:.4f})",
)
c.item(
    "1. free chiral pattern: light massless, heavy gapped (GL_p0, GR_p0, m_L(t*), m_R(t*))",
    (fc["GL_p0"], fc["GR_p0"], fc["mL_eff"][2], fc["mR_eff"][2]),
    abs(fc["GL_p0"] - 1) < 1e-9 and fc["GR_p0"] < 0.97 and fc["mR_eff"][2] > 0.42,
    "{0[0]:.4f}, {0[1]:.4f}; {0[2]:.4f}, {0[3]:.4f}",
)
c.item(
    "1. free chi_L_sum flipped / chiral",
    (f["chi_L_sum"], fc["chi_L_sum"], f["chi_L_sum"] / fc["chi_L_sum"]),
    3.0 < f["chi_L_sum"] / fc["chi_L_sum"] < 3.4,
    "{0[0]:.4f} / {0[1]:.4f} = {0[2]:.2f}",
)

fn = S3 / "L6_y3_h0.5_flipped.npz"
_, meta = open_chain(fn)
c.item(
    "2. flipped chain complete (trajectories)",
    meta["ntraj_done"],
    meta["ntraj_done"] >= 400 and meta["h10"] == 0.5 and meta["y"] == 3.0,
)
r = stage1.read_chain(fn, bl)
r0 = stage1.read_chain(N1DIR / "S1" / "L6" / "L6_y3_k-0.01_g0_g60_h0.5_chiral_bcaaaa.npz", bl)
c.item(
    "2. measurements after the cut, acceptance per 50 (min)",
    (r["n"], r["acc_min50"]),
    r["n"] >= 75 and r["acc_min50"] >= 0.5,
    "{0[0]}, {0[1]:.2f}",
)
sig = (f["mL_eff"][2] - r["mL"][0]) / r["mL"][1]
c.item(
    "2. flipped m_L(t*) below the free flipped light value (significance); chiral chain m_L(t*)",
    (r["mL"][0], r["mL"][1], sig, r0["mL"][0], r0["mL"][1]),
    sig > 10,
    "{0[0]:.4f}({0[1]:.4f}), {0[2]:.0f} sigma; chiral {0[3]:.4f}({0[4]:.4f})",
)
dlr = abs(r["mL"][0] - r["mR"][0]) / np.hypot(r["mL"][1], r["mR"][1])
c.item("2. flipped m_L = m_R (no selectivity on SMG configurations), distance", dlr, dlr < 3, "{:.1f} sigma")
dpat = abs(r["mL"][0] - r0["mL"][0]) / np.hypot(r["mL"][1], r0["mL"][1])
c.item("2. flipped vs chiral m_L (recorded)", dpat, True, "{:.1f} sigma")
c.item(
    "2. chi_L/free and GL_p0/free: flipped (own free) vs chiral",
    (
        r["chi_L"][0] / f["chi_L_sum"],
        r0["chi_L"][0] / fc["chi_L_sum"],
        r["GL_p0"][0] / f["GL_p0"],
        r0["GL_p0"][0] / fc["GL_p0"],
    ),
    True,
    "{0[0]:.4f} vs {0[1]:.4f}; {0[2]:.4f} vs {0[3]:.4f}",
)
c.item(
    "2. flipped A, phi_R",
    (r["A"], r["phi_R"]),
    True,
    "A {0[0][0]:+.3f}({0[0][1]:.3f}), phi_R {0[1][0]:+.2e}({0[1][1]:.1e})",
)
dphi = abs(r["phi_R"][0] - r0["phi_R"][0]) / np.hypot(r["phi_R"][1], r0["phi_R"][1])
dT = abs(r["phi_T_R"][0] - r0["phi_T_R"][0]) / np.hypot(r["phi_T_R"][1], r0["phi_T_R"][1])
c.item(
    "3. composite response phi_T_R flipped vs chiral (distance); phi_R distance",
    (r["phi_T_R"], r0["phi_T_R"], dT, dphi),
    dphi > 5 or dT > 5,
    "{0[0][0]:+.6f}({0[0][1]:.6f}) vs {0[1][0]:+.6f}({0[1][1]:.6f}): {0[2]:.0f} sigma; phi_R {0[3]:.1f} sigma",
)
c.done()
