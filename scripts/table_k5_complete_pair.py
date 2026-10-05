#!/usr/bin/env python3
"""results/K5_complete_pair.csv: chi_L/free on 4^4, 8^4 (pooled) and 6^4 at y = 3.0 and the exponents e - e_free on the
complete pair (4^4, 8^4) and on (6^4, 8^4) with the same reader (claim K5.13); results/K5_taste_coverage.csv: the
fraction of momenta on which the light/heavy taste split exists, per all-antiperiodic box."""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.analysis import complete_pair as P
from masspairing.data import RESULTS


def main():
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "K5_taste_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["box", "V", "boundary_modes", "split_fraction"])
        for key, r in P.coverage().items():
            w.writerow([key, r["V"], r["boundary"], f"{r['split']:.4f}"])
    with open(RESULTS / "K5_complete_pair.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "h",
                "chiL_over_free_L4",
                "chiL_over_free_L8_pooled",
                "n_L8",
                "chiL_over_free_L6_S1",
                "chiL_over_free_L6_S1b",
                "e_free_48",
                "e_minus_free_48",
                "err_48",
                "e_minus_free_68_S1",
                "err_68_S1",
                "e_minus_free_68_S1b",
                "err_68_S1b",
            ]
        )
        for h, r in P.pair_rows().items():
            l6, e68 = r["L6"], r["e68"]
            b = "S1b_new"
            w.writerow(
                [
                    h,
                    f"{r['L4']['over_free']:.5f}",
                    f"{r['L8']['over_free']:.5f}",
                    r["L8"]["n"],
                    f"{l6['S1']['over_free']:.5f}",
                    f"{l6[b]['over_free']:.5f}" if b in l6 else "",
                    f"{r['free']['e48']:.4f}",
                    f"{r['e48']['minus_free']:.4f}",
                    f"{r['e48']['err']:.4f}",
                    f"{e68['S1']['minus_free']:.4f}",
                    f"{e68['S1']['err']:.4f}",
                    f"{e68[b]['minus_free']:.4f}" if b in e68 else "",
                    f"{e68[b]['err']:.4f}" if b in e68 else "",
                ]
            )
    print(RESULTS / "K5_complete_pair.csv")


if __name__ == "__main__":
    main()
