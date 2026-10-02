#!/usr/bin/env python3
"""results/algebra_anomaly.csv: Dai-Freed anomaly table of nu Weyl fermions of charge 1 (claim K1.4).

One row per Spin-Z_{2m} structure h on a lens space S^5/Z_n (n <= 4m, m = 2, 4, 8): xi = eta/2 as an exact fraction,
its order, and nu xi mod 1 for nu = 16, 32, 48 (0 = anomaly-free on that lens space).
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.algebra.eta import anomaly_table
from masspairing.data import RESULTS


def main():
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "algebra_anomaly.csv"
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "h", "xi", "order", "nu16_xi_mod1", "nu32_xi_mod1", "nu48_xi_mod1"])
        for m in (2, 4, 8):
            for n, h, xi, per in anomaly_table(m):
                w.writerow([f"Spin-Z{2 * m}", n, h, str(xi), xi.denominator, str(per[16]), str(per[32]), str(per[48])])
    print(dst)


if __name__ == "__main__":
    main()
