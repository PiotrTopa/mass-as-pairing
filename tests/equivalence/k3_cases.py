"""Ensembles and chain prefixes compared between the K3 analysis code and the notebook (make_k3_reference.py,
test_k3.py). Ensembles are lists of (archive chain path, first kept trajectory)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from masspairing.analysis import k3  # noqa: E402

NCOPY = {("compl", 0.01): 1268, ("compl", 0.02): 1071, ("wedge", 0.01): 1500, ("wedge", 0.02): 1500}
K0_STOCH = {("compl", 0.01): 300, ("compl", 0.02): 200, ("wedge", 0.01): 400, ("wedge", 0.02): 300}


def ensembles():
    E = {}
    for p in k3.P1_POINTS:
        for L in (4, 6, 8):
            E[f"P1_{p}_{L}"] = k3.p1_spec(p, L)
    for p in k3.X_POINTS:
        for L in (4, 6, 8):
            E[f"X_{p}_{L}"] = k3.x_spec(p, L)
    for g6 in (0.05, 0.0):
        for L in (4, 6, 8):
            for h in (0.005, 0.01, 0.02):
                E[f"P2_{g6}_{L}_{h}"] = [(k3.p2_rel(L, g6, h), 0)]
    for L in (4, 6):
        E[f"eps_{L}"] = [(f"{k3.XI}/F8_P3_eps/{k3.chain_name(L, 2.41, 0, 0)}", 0)]
        for m in ("compl", "wedge"):
            for g in (0.01, 0.02):
                E[f"P3_{m}_{g}_{L}"] = [(k3.p3_rel(m, g, L), 450 if (m, g, L) == ("wedge", 0.01, 6) else 0)]
    for y, g in ((2.41, 0.05), (3.0, 0.1), (2.41, 0.1)):
        for L in (4, 6, 8):
            E[f"C073_{y}_{g}_{L}"] = [(k3.c073_rel(L, y, g), 0)]
    for s in ("afmdimer", "hothot"):
        E[f"C073_{s}"] = [(k3.c073_rel(6, 2.41, 0.1, s), 0)]
    return E


def stochastic8():
    """The stochastic 8^4 ensembles before the copy point: notebook = the original chain files, clean = the
    continuation files (whose first NCOPY trajectories are those chains) cut at NCOPY."""
    out = {}
    for (m, g), k0 in K0_STOCH.items():
        orig = f"{k3.XI}/F8_P3_{m}/{k3.chain_name(8, 2.41, g, 0.0, '_links' if m == 'compl' else '')}"
        reps = []
        if m == "compl":
            reps = [(orig.replace("F8_P3_compl", f"F8_P3_compl_{r}"), 1100 if g == 0.01 else 900) for r in ("rA", "rB")]
        out[f"stoch8_{m}_{g}"] = dict(
            notebook=[(orig, k0)] + reps,
            clean=[(k3.p3_rel(m, g, 8), k0, NCOPY[(m, g)])] + [(r, k, None) for r, k in reps],
        )
    return out


# keys of the analysis results compared (mean and error)
KEYS = (
    "chi10",
    "chi10_corner_max",
    "xi10",
    "xi10_cmax",
    "chi6_corner_max",
    "E10",
    "dimer_sq_par",
    "dimer_sq_perp",
    "phi_stag_sq",
    "V_dimer_sq_par",
    "V_dimer_sq_perp",
    "V_phi_stag_sq",
    "phi_src",
    "dphi_src_dh",
    "O4",
    "xi2_stag",
    "xi2_s_dimer_0",
    "xi2_s_dimer_1",
)
