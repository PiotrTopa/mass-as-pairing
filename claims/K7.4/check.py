"""K7.4: xi(y) at L <= 12 cannot separate walking, xi = a exp(c/sqrt(t)), from a power law, xi = a t^-nu
(t = |y - y_c|), unless y_c is known exactly; the separation needs L >= 16 with 1 % errors on xi, or L = 24.

Deterministic mock (masspairing.analysis.fss.discrimination_table, seconds of CPU): matched dynamic range,
xi(t_max) = 1.5, nu = 0.61; only points with xi <= L_max/3; the wrong model is fitted with (log a, exponent) and
optionally a shift of y_c; its minimum chi^2 is the expected Delta chi^2 (optimistic: Gaussian errors, no corrections
to scaling).
  (a) pilot grid (t = 0.042, 0.127, 0.38): no fit at any L_max <= 16 (<= 2 usable points), none with y_c free at
      L_max <= 24 (3 points for 3 parameters);
  (b) 7 log-spaced points in [0.01, 0.4], y_c free: no fit or Delta chi^2 <= 1.5 for L_max <= 12 at every eps >= 1 %;
      >= 7 at L_max = 16 only with eps = 1 % (<= 1.5 at 3 %); >= 30 at L_max = 24 with 1 %;
  (c) y_c fixed: L_max = 12 at 3 % would give Delta chi^2 >= 4 both ways.
"""

import math
import os
import pathlib
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from masspairing.analysis.fss import discrimination_table
from masspairing.claimcheck import Check

c = Check("K7.4")
rows = discrimination_table()


def get(grid, Lmax, eps, free):
    for r in rows:
        if r["grid"].startswith(grid) and r["Lmax"] == Lmax and abs(r["eps"] - eps) < 1e-9 and r["free_shift"] == free:
            return r
    raise KeyError((grid, Lmax, eps, free))


nan = lambda x: x is None or (isinstance(x, float) and math.isnan(x))
for L in (8, 12, 16):
    r = get("pilot", L, 0.03, False)
    c.item(
        f"(a) pilot grid, y_c fixed, L_max = {L}: usable points (no fit)",
        r["n_use"],
        r["n_use"] <= 2 and nan(r["dchi2_WP"]),
    )
for L in (8, 12, 16, 24):
    r = get("pilot", L, 0.01, True)
    c.item(f"(a) pilot grid, y_c free, L_max = {L}: usable points (no fit)", r["n_use"], nan(r["dchi2_WP"]))
g = "7 log-spaced in [0.01, 0.4]"
for L in (8, 12):
    for eps in (0.01, 0.03, 0.05):
        r = get(g, L, eps, True)
        v = "no fit" if nan(r["dchi2_WP"]) else f"{max(r['dchi2_WP'], r['dchi2_PW']):.2f}"
        ok = nan(r["dchi2_WP"]) or max(r["dchi2_WP"], r["dchi2_PW"]) <= 1.5
        c.item(
            f"(b) {g}, y_c free, L_max = {L}, eps = {eps}: usable points, max Delta chi^2 (<= 1.5)",
            f"{r['n_use']}, {v}",
            ok,
        )
r = get(g, 16, 0.01, True)
c.item(
    "(b) L_max = 16, eps = 1 %: Delta chi^2 walking->power | power->walking (>= 7)",
    f"{r['dchi2_WP']:.1f} | {r['dchi2_PW']:.1f}",
    min(r["dchi2_WP"], r["dchi2_PW"]) >= 7,
)
r = get(g, 16, 0.03, True)
c.item(
    "(b) L_max = 16, eps = 3 %: Delta chi^2 (<= 1.5)",
    f"{r['dchi2_WP']:.2f} | {r['dchi2_PW']:.2f}",
    max(r["dchi2_WP"], r["dchi2_PW"]) <= 1.5,
)
r = get(g, 24, 0.01, True)
c.item(
    "(b) L_max = 24, eps = 1 %: Delta chi^2 (>= 30)",
    f"{r['dchi2_WP']:.1f} | {r['dchi2_PW']:.1f}",
    min(r["dchi2_WP"], r["dchi2_PW"]) >= 30,
)
r = get(g, 12, 0.03, False)
c.item(
    "(c) y_c fixed, L_max = 12, eps = 3 %: Delta chi^2 (>= 4)",
    f"{r['dchi2_WP']:.1f} | {r['dchi2_PW']:.1f}",
    min(r["dchi2_WP"], r["dchi2_PW"]) >= 4,
)
c.done()
