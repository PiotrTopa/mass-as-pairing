#!/usr/bin/env python3
"""figures/k6_alpha.{pdf,png}: odd-part exponent alpha of the corner-block loop per stored configuration against y
(claim K6.2). Left L = 6, right L = 8. SYM and SMG configurations (filled: scored and not near-critical; open: the
other labelled files); P_c and AFM files as the third series. alpha = -1: corner pole; alpha > 0: zero-type gap.
Exact per configuration (no error bars)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import derived
from masspairing.plotting import MARKERS, NEUTRAL, SERIES, figure, save


def main():
    recs = json.loads(derived("n1inst", "K6_winding_stored.json").read_text())["records"]
    fig, ax = figure(ncols=2, width=3.8)
    groups = [("SYM", ("SYM",)), ("SMG", ("SMG",)), ("$P_c$ / AFM", ("critical", "AFM"))]
    for j, L in enumerate((6, 8)):
        a = ax[0, j]
        for i, (label, phases) in enumerate(groups):
            sel = [r for r in recs if r["L"] == L and r["phase"] in phases]
            core = [r for r in sel if r["scored"] and not r["near_critical"]]
            rest = [r for r in sel if r not in core]
            kw = dict(color=SERIES[i], marker=MARKERS[i], ls="none")
            a.plot([r["y"] for r in core], [r["alpha"] for r in core], label=label, **kw)
            a.plot([r["y"] for r in rest], [r["alpha"] for r in rest], mfc="none", **kw)
        a.axhline(-1.0, color=NEUTRAL, ls="--", lw=0.8)
        a.axhline(0.0, color=NEUTRAL, ls=":", lw=0.8)
        a.set_xlabel("y")
        a.set_ylabel("α")
        a.set_title(f"L = {L}")
        if j == 0:
            a.legend()
    save(fig, "k6_alpha")


if __name__ == "__main__":
    main()
