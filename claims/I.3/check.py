"""I.3: calibration of the Yukawa axis against the published phase diagram: y_ours = sqrt(2) y_ref with kappa and
lambda unchanged, hence P_c = (sqrt(2) 1.706, -0.01) = (2.41, -0.01).

From the stored chains (data/derived/k7/calib: L = 8, 6, 12 at kappa = 0.2, the kappa scan at L = 6, 8, the 4^4 / 6^4
runs at kappa = 0.2):
 (1) L = 8, kappa = 0.2: the maximum of chi_Sigma,stag over y <= 3.2 is interior and in (2.3, 2.6); y_1(L = 8) from
     parabolas through the maximum and its neighbours; every point with y <= 1.9 is disordered (|Sigma_stag| <= 0.035)
     and at y = 1.7 |Sigma_stag| falls from L = 6 to 8 like 1/sqrt(V) within 10 %.
 (2) every L = 8 point with 2.6 <= y <= 3.6 is ordered (|Sigma_stag| >= 0.2); y_2(L = 8) is bracketed above y = 3.
 (3) rescaling y_ours = s y_ref: only s = sqrt(2) classifies every L = 8, kappa = 0.2 point correctly; s = 1 and
     s = 1/sqrt(2) misclassify >= 3 points each (control).
 (4) |Sigma_stag| at y ~ 3 does not decay with L (L = 4, 6, 8 at 3.0, L = 12 at 2.9): within 10 % of the mean.
 (5) kappa scan (y = 2.6, 3.0; kappa = 0.1 ... -0.03): the ordered window shrinks onto P_c as predicted by the
     rescaled published window, y_2 linear in kappa; points within 0.1 of an edge or with acceptance < 0.2 not scored.
"""

import os
import pathlib
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from masspairing.analysis.calibration import ORDERED, kappa_window, misclassified, point_summary, y1_parabola
from masspairing.claimcheck import Check
from masspairing.data import derived, load_chain

c = Check("I.3")
R = {}
for p in sorted(derived("k7", "calib").glob("*.npz")):
    r = point_summary(*load_chain(p))
    R[(r["L"], round(r["kappa"], 4), round(r["y"], 4))] = r


def S(L, k, y):
    return R[(L, k, y)]["Sigma_stag_abs"]["mean"]


for key in sorted(R):
    r = R[key]
    flag = "  (tau_int > N/50)" if r["tau_max"] > r["ntraj"] / 50 else ""
    c.record(
        f"point (L, kappa, y) = {key}: N, acceptance, max tau_int, |Sigma_stag|, chi_Sigma,stag, xi_2,stag",
        f"{r['ntraj']}, {r['acceptance']:.2f}, {r['tau_max']:.1f}, {r['Sigma_stag_abs']['mean']:.4f}"
        f"({r['Sigma_stag_abs']['err']:.4f}), {r['chi_Sigma_stag']['mean']:.3f}({r['chi_Sigma_stag']['err']:.3f}),"
        f" {r['xi2_stag']['mean']:.2f}({r['xi2_stag']['err']:.2f}){flag}",
    )

# (1) lower transition at L = 8
k8 = sorted(y for (L, k, y) in R if L == 8 and k == 0.2)
low = [y for y in k8 if y <= 3.2]
chi = np.array([R[(8, 0.2, y)]["chi_Sigma_stag"]["mean"] for y in low])
err = np.array([R[(8, 0.2, y)]["chi_Sigma_stag"]["err"] for y in low])
i = int(np.argmax(chi))
c.item(
    "(1) L = 8, kappa = 0.2: chi_Sigma,stag maximum at grid point y (interior, in (2.3, 2.6))",
    low[i],
    2.3 < low[i] < 2.6 and 0 < i < len(low) - 1,
)
c.record(
    "(1) chi_Sigma,stag at y = " + ", ".join(f"{y}" for y in low),
    ", ".join(f"{a:.2f}({b:.2f})" for a, b in zip(chi, err, strict=True)),
)
y1, y1lo, y1hi = y1_parabola(low[i - 1 : i + 2], chi[i - 1 : i + 2], err[i - 1 : i + 2])
c.item(
    "(1) y_1(L = 8) from resampled parabolas through the maximum and its neighbours (median, 16-84 %) vs sqrt(2) 1.706",
    f"{y1:.3f} (+{y1hi - y1:.3f} -{y1 - y1lo:.3f}) vs {np.sqrt(2) * 1.706:.3f}",
    abs(y1 - np.sqrt(2) * 1.706) < 0.02,
)
dis = [y for y in low if y <= 1.9]
c.item(
    "(1) L = 8, y <= 1.9 disordered: |Sigma_stag|",
    ", ".join(f"{S(8, 0.2, y):.4f}" for y in dis),
    all(S(8, 0.2, y) <= 0.035 for y in dis),
)
ratio = S(6, 0.2, 1.7) / S(8, 0.2, 1.7)
c.item(
    "(1) y = 1.7: |Sigma_stag|(L = 6)/|Sigma_stag|(L = 8) vs sqrt(V_8/V_6) (pure fluctuation)",
    f"{ratio:.3f} vs {(8 / 6) ** 2:.3f}",
    abs(ratio / (8 / 6) ** 2 - 1) < 0.1,
)

# (2) the ordered window at L = 8
win = [y for y in k8 if 2.6 <= y <= 3.6]
c.item(
    "(2) L = 8 ordered for 2.6 <= y <= 3.6: |Sigma_stag|",
    ", ".join(f"{y}: {S(8, 0.2, y):.3f}" for y in win),
    len(win) >= 3 and all(S(8, 0.2, y) >= 0.2 for y in win),
)
hi = [y for y in k8 if y > 3.0]
y2lo = max(y for y in hi if S(8, 0.2, y) >= ORDERED)
y2hi = min(y for y in hi if S(8, 0.2, y) < ORDERED)
c.item(
    "(2) y_2(L = 8) bracket (largest ordered, smallest disordered y above 3) vs sqrt(2) 2.69;"
    " |Sigma_stag| at the bracket",
    f"({y2lo}, {y2hi}) vs {np.sqrt(2) * 2.69:.3f}; {S(8, 0.2, y2lo):.3f}, {S(8, 0.2, y2hi):.3f}",
    y2hi > y2lo and y2lo < np.sqrt(2) * 2.69 < y2hi,
)
c.record(
    "(2) tau_int at the y_2 bracket (N = 1000)",
    f"{R[(8, 0.2, y2lo)]['tau_max']:.1f}, {R[(8, 0.2, y2hi)]['tau_max']:.1f}",
)

# (3) convention control
pts = [(y, S(8, 0.2, y) >= ORDERED) for y in k8]
bad = {name: misclassified(pts, s) for name, s in (("1/sqrt(2)", 1 / np.sqrt(2)), ("1", 1.0), ("sqrt(2)", np.sqrt(2)))}
for name, wrong in bad.items():
    c.record(f"(3) s = {name}: misclassified L = 8 points", wrong)
c.item(
    "(3) misclassified points for s = sqrt(2) | 1 | 1/sqrt(2) (only sqrt(2) reproduces the phase sequence)",
    f"{len(bad['sqrt(2)'])} | {len(bad['1'])} | {len(bad['1/sqrt(2)'])}",
    len(bad["sqrt(2)"]) == 0 and len(bad["1"]) >= 3 and len(bad["1/sqrt(2)"]) >= 3,
)

# (4) order at y ~ 3 independent of L
vals = {4: S(4, 0.2, 3.0), 6: S(6, 0.2, 3.0), 8: S(8, 0.2, 3.0), 12: S(12, 0.2, 2.9)}
mean = np.mean(list(vals.values()))
c.item(
    "(4) |Sigma_stag| at y ~ 3 (L = 4, 6, 8 at 3.0; L = 12 at 2.9), within 10 % of the mean",
    ", ".join(f"L = {L}: {v:.3f}" for L, v in vals.items()),
    all(abs(v / mean - 1) < 0.1 for v in vals.values()),
)

# (5) kappa scan
ks = sorted({k for (L, k, y) in R if k != 0.2})
for L, nmin in ((8, 5), (6, 8)):
    wrong, n, skipped = [], 0, []
    for k in ks:
        a, b = kappa_window(k)
        for y in (2.6, 3.0):
            if (L, k, y) not in R:
                continue
            o = S(L, k, y) >= ORDERED
            acc = R[(L, k, y)]["acceptance"]
            c.record(
                f"(5) L = {L}, kappa = {k:g}, y = {y}: |Sigma_stag|, acceptance, predicted window",
                f"{S(L, k, y):.3f} ({'ordered' if o else 'disordered'}), {acc:.2f}, ({a:.2f}, {b:.2f})",
            )
            if min(abs(y - a), abs(y - b)) > 0.1 and acc >= 0.2:
                n += 1
                if o != (a < y < b):
                    wrong.append((k, y))
            else:
                skipped.append((k, y))
    c.item(
        f"(5) L = {L} kappa scan: scored points, misclassified, not scored (edge or acceptance < 0.2)",
        f"{n}, {wrong}, {skipped}",
        n >= nmin and not wrong,
    )
c.done()
