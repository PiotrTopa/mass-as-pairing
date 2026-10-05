#!/usr/bin/env python3
"""results/K5_mass_strings.csv: the 16 corner-space Lorentz-scalar mass strings of the light doublet on 4^4 pppp and
aaaa with their flavour content, taste, and group averages of the string and its light part under the point-group
stabiliser of C_chi, the full stabiliser with one-site shifts and the Z4-extended point group (claim K5.12; read from
the frozen data/derived/n1inst/K512_mass_strings.json)."""

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import RESULTS, derived


def main():
    d = json.loads(derived("n1inst", "K512_mass_strings.json").read_text())
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "K5_mass_strings.csv"
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "box",
                "S",
                "D",
                "flavour",
                "taste",
                "avg_pointgroup",
                "avg_with_shifts",
                "avg_pointgroup_Z4",
                "light_avg_pointgroup",
                "light_avg_with_shifts",
                "flip_with_shifts",
                "keep_with_shifts",
            ]
        )
        for lab, box in d["boxes"].items():
            for v in box["strings"].values():
                w.writerow(
                    [
                        lab,
                        " ".join(map(str, v["S"])),
                        " ".join(map(str, v["D"])),
                        "10" if v["antisym"] else "6",
                        v["taste"],
                        f"{v['avg_pg']:.4f}",
                        f"{v['avg_all']:.4f}",
                        f"{v['avg_pg_z4']:.4f}",
                        f"{v['avgL_pg']:.4f}",
                        f"{v['avgL_all']:.4f}",
                        v["flip_keep_all"][0],
                        v["flip_keep_all"][1],
                    ]
                )
    print(dst)


if __name__ == "__main__":
    main()
