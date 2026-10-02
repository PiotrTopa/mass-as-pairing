#!/usr/bin/env python3
"""results/algebra_democracy.json: distances of Majorana source patterns from the sign-free (tau_2 K) class on 4^4
(claim K2.4): relative distance of M from span{1, Gamma_a} (x) R^{V x V}; 0 = inside the class."""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.algebra.signfree import class_distance
from masspairing.data import RESULTS
from masspairing.lattice import Lattice
from masspairing.patterns import chiral_mass, source_pattern


def main():
    lat = Lattice((4,) * 4)
    V = lat.V
    C0, Cx = source_pattern(lat).toarray(), chiral_mass(lat).toarray()
    m = np.array([0.2, 0.5, 1.0, 1.3])
    rows = {
        "1 (x) C_chi (flavour-blind chiral mass)": (np.kron(Cx, np.eye(4)), 0.0),
        "1 (x) C0 (flavour-blind nodal source)": (np.kron(C0, np.eye(4)), 0.0),
        "C0 (x) diag(0,0,0,1) (1-of-4 split)": (np.kron(C0, np.diag([0, 0, 0, 1.0])), np.sqrt(3) / 2),
        "C_chi (x) diag(0,0,1,1) (2+2 split)": (np.kron(Cx, np.diag([0, 0, 1, 1.0])), 1 / np.sqrt(2)),
        "C_chi (x) diag(0.2,0.5,1,1.3)": (
            np.kron(Cx, np.diag(m)),
            float(np.linalg.norm(m - m.mean()) / np.linalg.norm(m)),
        ),
    }
    out = {
        "lattice": "4^4, periodic space, antiperiodic time",
        "class": "span{1, Gamma_a} (x) R^{V x V}",
        "rows": [
            {"pattern": k, "relative_distance": round(float(class_distance(M, V)), 12), "exact": round(float(e), 12)}
            for k, (M, e) in rows.items()
        ],
    }
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "algebra_democracy.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(dst)


if __name__ == "__main__":
    main()
