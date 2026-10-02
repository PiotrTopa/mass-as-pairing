#!/usr/bin/env python3
"""results/algebra_branching.csv: Spin(10) -> SU(4) x SU(2)_1 x SU(2)_2 branching of 16, 10, 126, 120, 45 (claim K1.5).

One row per constituent: the Spin(10) representation, the D3 x A1 x A1 highest weight (physical coordinates), the
dimensions (SU(4), SU(2)_1, SU(2)_2) and the multiplicity.
"""

import csv
import pathlib
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.algebra.characters import (
    branch_d5_to_ps,
    char126,
    decompose,
    ms_ext_powers,
    spinor16,
    vector10,
    weyl_dim,
)
from masspairing.data import RESULTS


def main():
    s16 = spinor16(-1)
    reps = {
        "16": s16,
        "10": vector10(),
        "126": char126(),
        "120": ms_ext_powers(s16, 2)[2],
        "45": ms_ext_powers(vector10(), 2)[2],
    }
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "algebra_branching.csv"
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spin10", "highest_weight_D3xA1xA1", "dim_su4", "dim_su2_1", "dim_su2_2", "multiplicity"])
        for name, ch in reps.items():
            dec = decompose(branch_d5_to_ps(ch), 2, 3)
            for lam, m in sorted(dec.items(), key=lambda t: (-weyl_dim(t[0][:3], 0, 3), t[0])):
                hw = "(" + ",".join(str(Fraction(x, 2)) for x in lam) + ")"
                w.writerow([name, hw, weyl_dim(lam[:3], 0, 3), lam[3] + 1, lam[4] + 1, m])
    print(dst)


if __name__ == "__main__":
    main()
