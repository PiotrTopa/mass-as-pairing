"""K7.1: the finite-size-scaling crossing of xi_2,stag / L at P_c from L = 6 and 8.

On the kappa = -0.01 chains (the analysed prefixes of masspairing.analysis.fss.K71_PREFIX; all trajectories of the
prefix, blocked jackknife with block = ceil(2 tau_int(m^2))):
  (1) xi_2,stag/L at y = 2.41: L = 6 and L = 8 agree within 1 sigma (combined), both in [0.35, 0.45];
  (2) at y = 2.537 xi_2,stag/L falls from L = 6 to L = 8 by more than 2.5 sigma; at y = 2.028 and 2.792 it is < 0.15
      on both volumes;
  (3) the two-volume slope ratio between y = 2.41 and 2.537 gives a finite 1/nu_eff = ln(s_8/s_6)/ln(8/6) (reported,
      not a measurement of nu);
  R4 = <m^4>/<m^2>^2 and tau_int(m^2) at y = 2.41 are reported.
"""

import os
import pathlib
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from masspairing.analysis.fss import K71_PREFIX, ratio_row, slope_ratio
from masspairing.analysis.t3a import load_records
from masspairing.claimcheck import Check

c = Check("K7.1")
rows = {}
for r in load_records("pilot"):
    if r["dir"] not in ("F_L6_k-0.01", "F_L8_k-0.01"):
        continue
    n = K71_PREFIX[r["L"]][r["y"]]
    rows[(r["L"], r["y"])] = row = ratio_row(r["L"], r["Sigma_stag"][:n], r["S_pi"][:n], r["S_pi_pmin"][:n])
    flag = "" if row["ok"] and row["tau"] < row["n"] / 50 else "  (tau_int > N/50)"
    c.record(
        f"L = {r['L']}, y = {r['y']}: n, tau_int(m^2), |Sigma_stag|, R4, xi_2,stag/L",
        f"{row['n']}, {row['tau']:.1f}, {row['mabs']:.4f}({row['emabs']:.4f}), {row['R4']:.3f}({row['eR4']:.3f}),"
        f" {row['xiL']:.3f}({row['exiL']:.3f}){flag}",
    )
c.item("chains analysed (L = 6: 7, L = 8: 5)", len(rows), len(rows) == 12)
a, b = rows[(6, 2.41)], rows[(8, 2.41)]
diff, sig = b["xiL"] - a["xiL"], np.hypot(a["exiL"], b["exiL"])
c.item(
    "(1) xi_2/L at y = 2.41: L = 6 | L = 8 | difference (sigma)",
    f"{a['xiL']:.3f}({a['exiL']:.3f}) | {b['xiL']:.3f}({b['exiL']:.3f}) | {diff:+.3f} ({diff / sig:+.1f})",
    abs(diff) < sig and 0.35 <= a["xiL"] <= 0.45 and 0.35 <= b["xiL"] <= 0.45,
)
cc, e = rows[(6, 2.537)], rows[(8, 2.537)]
drop = (cc["xiL"] - e["xiL"]) / np.hypot(cc["exiL"], e["exiL"])
c.item(
    "(2) xi_2/L at y = 2.537: L = 6 -> L = 8 (drop in sigma, > 2.5)",
    f"{cc['xiL']:.3f}({cc['exiL']:.3f}) -> {e['xiL']:.3f}({e['exiL']:.3f}) ({drop:.1f})",
    drop > 2.5,
)
far = {(L, y): rows[(L, y)]["xiL"] for L in (6, 8) for y in (2.028, 2.792)}
c.item(
    "(2) xi_2/L at y = 2.028 and 2.792 (L = 6, 8) < 0.15",
    ", ".join(f"({L}, {y}): {v:.3f}" for (L, y), v in far.items()),
    all(v < 0.15 for v in far.values()),
)
sr = slope_ratio(a, cc, b, e, 2.537 - 2.41)
inv, einv = sr["inv_nu"], sr["e_inv_nu"]
c.item(
    "(3) slopes d(xi/L)/dy on the SMG side L = 6 | L = 8; 1/nu_eff; nu_eff (+/-)",
    f"{sr['s6']:.2f}({sr['es6']:.2f}) | {sr['s8']:.2f}({sr['es8']:.2f}); {inv:.2f} +- {einv:.2f};"
    f" {1 / inv:.2f} (+{1 / (inv - einv) - 1 / inv:.2f} -{1 / inv - 1 / (inv + einv):.2f})",
    np.isfinite(inv) and sr["s6"] > 0 and sr["s8"] > 0,
)
c.item(
    "R4 at y = 2.41: L = 6 | L = 8, more than 3 sigma from the Gaussian 5/3 and from the ordered 1",
    f"{a['R4']:.3f}({a['eR4']:.3f}) | {b['R4']:.3f}({b['eR4']:.3f})",
    all(abs(x["R4"] - 5 / 3) > 3 * x["eR4"] and abs(x["R4"] - 1) > 3 * x["eR4"] for x in (a, b)),
)
c.record("tau_int(m^2) at y = 2.41: L = 6 | L = 8", f"{a['tau']:.1f} | {b['tau']:.1f}")
c.done()
