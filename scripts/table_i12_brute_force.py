#!/usr/bin/env python3
"""results/I12_brute_force.csv: exact-determinant Metropolis against the RHMC variants on 2^4 (claims I.1, I.2).

One row per (comparison, observable): Metropolis mean and error, RHMC mean and error (20 equal blocks), pull.
Reads the frozen runs of data/derived/k7/mc (scripts/derive_k7.py mc).
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from masspairing.data import RESULTS, derived, load_chain  # noqa: E402

KEYS = ("sigma2", "O4", "Sigma_abs", "Sigma_stag_abs", "phi_stag_sq")
COMPARISONS = [  # claim, y, reference run, compared run
    ("I.1", 0.0, "metropolis_y0", "rhmc_y0"),
    ("I.1", 1.2, "metropolis_y1.2", "rhmc_y1.2"),
    ("I.2", 1.2, "metropolis_y1.2", "hasenbusch_y1.2"),
    ("I.2", 2.5, "metropolis_y2.5", "hasenbusch_y2.5"),
    ("I.2 sabotage", 2.5, "metropolis_y2.5", "sabotage_heavy_only_y2.5"),
]


def blocked(x, nb=20):
    x = np.asarray(x, float)
    n = len(x) // nb * nb
    b = x[:n].reshape(nb, -1).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)


def main():
    rows = []
    for claim, y, ref, run in COMPARISONS:
        a, _ = load_chain(derived("k7", "mc", f"{ref}.npz"))
        b, _ = load_chain(derived("k7", "mc", f"{run}.npz"))
        for k in KEYS if y else ("sigma2",):
            vm, em = blocked(a["ts_" + k])
            vh, eh = blocked(b["ts_" + k])
            rows.append(
                dict(
                    claim=claim, y=y, reference=ref, run=run, observable=k, acceptance=f"{b['ts_accepted'].mean():.3f}"
                )
                | dict(metropolis=f"{vm:.6g}", e_metropolis=f"{em:.2g}", rhmc=f"{vh:.6g}", e_rhmc=f"{eh:.2g}")
                | dict(pull=f"{(vm - vh) / np.hypot(em, eh):+.2f}")
            )
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "I12_brute_force.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"results/I12_brute_force.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
