#!/usr/bin/env python3
"""K8.3 -- the small-h onset of the light pair-channel gap at P_c = (2.41, -0.01) on 6^4 aaaa, by the pre-registered
fit F1: m^2 - m0^2 = (c h^p)^2 on h in {0.25 (pooled replicas), 0.5 (pooled replicas), 0.75, 1}, m0 = the stored h = 0
chain, gives p = 1.14(37); the interval p +- 2 sigma meets both the quadratic band [1.6, 2.4] and the linear band
[0.7, 1.3] -> UNDECIDED. Power-limit control: a synthetic quadratic onset with the real errors also returns UNDECIDED
(the fit cannot certify a quadratic onset at these errors); the linear synthetic returns "linear", the saturating one
is not called quadratic. Reads data/derived/n1stage/{S1,S1b,S1c} and the free baselines (about 20 s)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from masspairing.analysis import stage1c as C
from masspairing.claimcheck import Check

c = Check("K8.3")
rows1b, tagrow, res = C.analyse(with_sensitivity=True)
fmt = C.fmt
Bres = res["B"]

# 1. the points
for pt in Bres["curve"]:
    c.record(f"6^4 P_c h = {pt['h']:g}: midpoint cosh mass m(t = 2) on CL_t", f"{fmt(pt['m'])} [{pt['origin']}]")
c.item(
    "6^4 replica tests CONSISTENT (K5.9) -> the pooled points enter F1 at h = 0.25, 0.5",
    Bres["used_pooled"],
    Bres["used_pooled"] == {0.25: True, 0.5: True},
)
cv = {(pt["h"], pt["origin"]): pt["m"] for pt in Bres["curve"]}
m025, m05, m075 = cv[(0.25, "stage 1c, pooled")], cv[(0.5, "stage 1c, pooled")], cv[(0.75, "stage 1c")]
c.item(
    "new points m(0.25), m(0.5) (pooled), m(0.75)",
    (m025, m05, m075),
    abs(m025[0] - 0.3921) < 5e-5 and abs(m05[0] - 0.4041) < 5e-5 and abs(m075[0] - 0.4270) < 5e-5,
    "{0[0][0]:.4f}({0[0][1]:.4f}), {0[1][0]:.4f}({0[1][1]:.4f}), {0[2][0]:.4f}({0[2][1]:.4f})",
)

# 2. F1, the verdict fit
F = Bres["F1"]
c.item(
    "F1: p, chi^2/dof, I = p +- 2 sigma_p",
    (F["p"], F["chi2_dof"], F["I"]),
    abs(F["p"][0] - 1.140) < 0.01 and abs(F["p"][1] - 0.366) < 0.01 and F["chi2_dof"] < 3,
    "{0[0][0]:.3f}({0[0][1]:.3f}), {0[1]:.2f}, [{0[2][0]:.2f}, {0[2][1]:.2f}]",
)
c.item(
    "F1 -> UNDECIDED: I meets the quadratic band [1.6, 2.4] and the linear band [0.7, 1.3], sigma_p > 0.3",
    F["verdict"],
    F["verdict"] == "UNDECIDED" and F["meets_quad"] and F["meets_lin"] and F["p"][1] > 0.3,
)

# 3. labelled fits (no verdict)
for k in ("F1prime", "F2", "F1f"):
    f = Bres[k]
    c.record(f"labelled {k}", f"p = {fmt(f['p'], 3)}, chi^2/dof {f['chi2_dof']:.2f} -> {f['verdict']}")
c.item(
    "labelled: F1' (+ stored h = 0.5), F2 (h <= 0.75), F1f (free-normalised) all UNDECIDED; F1f = F1 (free midpoint "
    "mass at 6^4 t = 2 is h-independent)",
    tuple(Bres[k]["verdict"] for k in ("F1prime", "F2", "F1f")),
    all(Bres[k]["verdict"] == "UNDECIDED" for k in ("F1prime", "F2", "F1f"))
    and all(abs(v - 1) < 1e-9 for v in Bres["free_factors"].values())
    and abs(Bres["F1f"]["p"][0] - F["p"][0]) < 1e-9,
)
pn = Bres["pull_new_vs_stored_h05"]
c.item(
    "new pooled h = 0.5 point vs the stored stage-1 one (pull, 3 sigma rule)",
    (pn["new"], pn["stored"], pn["pull"]),
    not pn["flag"],
    "{0[0][0]:.4f}({0[0][1]:.4f}) vs {0[1][0]:.4f}({0[1][1]:.4f}), {0[2]:+.2f} sigma",
)
mo = {x["step"]: x["pull"] for x in Bres["monotonicity"]}
c.item(
    "monotonicity h = 0 -> 0.25 -> 0.5 -> 0.75 -> 1 (pulls): no resolvable rise at 0.25, then rising",
    tuple(mo.values()),
    abs(mo["0->0.25"]) < 1 and mo["0.5->0.75"] > 2.5 and mo["0.75->1"] > 3,
    "{0[0]:.1f}, {0[1]:.1f}, {0[2]:.1f}, {0[3]:.1f} sigma",
)
loc = [x["kappa_local"] for x in Bres["local_exponents"]]
c.record("local quadrature exponents (0.25->0.5, 0.5->0.75, ..., 2->3)", ", ".join(f"{k:.2f}" for k in loc))

# 4. controls: the power of F1 (pre-registered; the quadratic one fails = power limit, disclosed)
ct = Bres["F1_controls"]
for k in ("quadratic", "linear", "saturating"):
    c.record(
        f"synthetic {k}",
        f"p = {fmt(ct[k]['p'], 2)}, I = [{ct[k]['I'][0]:.2f}, {ct[k]['I'][1]:.2f}] -> {ct[k]['verdict']}",
    )
c.item(
    "control: linear synthetic -> 'linear' (p); saturating synthetic not 'quadratic-like onset'",
    (ct["linear"]["p"], ct["saturating"]["verdict"]),
    ct["linear"]["verdict"] == "linear" and ct["saturating"]["verdict"] != "quadratic-like onset",
    "{0[0][0]:.2f}({0[0][1]:.2f}); {0[1]}",
)
c.item(
    "power limit: quadratic synthetic with the real errors -> UNDECIDED (p): F1 cannot certify a quadratic onset at "
    "these errors (the pre-registered quadratic control fails; reported, not retuned)",
    ct["quadratic"]["p"],
    ct["quadratic"]["verdict"] == "UNDECIDED" and not ct["ok"] and ct["quadratic"]["p"][1] > 0.3,
    "{0[0]:.2f}({0[1]:.2f})",
)
s = res["sensitivity"]["F1_mc"]
c.record(
    "recorded (post hoc): Gaussian-MC sigma_p; P(p >= 1.6); P(0.7 <= p <= 1.3)",
    f"{s['mc_std']:.2f}; {s['frac_p_ge_1p6']:.2f}; {s['frac_p_in_lin']:.2f}",
)
c.done()
