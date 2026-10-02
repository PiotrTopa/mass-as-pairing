#!/usr/bin/env python3
"""results/algebra_mtr.csv: Majorana-time-reversal classification of the Hubbard-Stratonovich ensembles of the
2+1d four-flavour model on the 2-site and 2x2 clusters (claim K2.2): dims of the symmetric / antisymmetric solution
spaces L, existence of T+ and T-, and the sign-free class (Majorana, Kramers) or none."""

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.algebra.hamiltonian import bdg_flavour_generator, hs_ensembles, majorana_matrix_from_bdg, mtr_classify
from masspairing.data import RESULTS


def main():
    rng = np.random.default_rng(21)
    RESULTS.mkdir(exist_ok=True)
    dst = RESULTS / "algebra_mtr.csv"
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "ensemble", "dimL_sym", "dimL_anti", "T_plus", "T_minus", "sign_free_class"])
        for nsites, bonds, lab in [(2, [(0, 1)], "2-site"), (4, [(0, 1), (1, 2), (2, 3), (3, 0)], "2x2")]:
            E = hs_ensembles(nsites, bonds)
            Tz = np.diag([1.0, -1.0, 0, 0])
            zeeman = [(majorana_matrix_from_bdg(*bdg_flavour_generator(nsites, s, Tz)), "herm") for s in range(nsites)]
            cases = {
                "eps V>0 (He et al.)": E["eps_pos"],
                "eps V<0": E["eps_neg"],
                "Sym2 scheme A (pairing)": E["pairing"],
                "Sym2 scheme B (density + exchange)": E["dens_exch"],
                "Sym2 scheme C (density + SU(4) spin)": E["spin"],
                "eps + Sym2 scheme A": E["eps_pos"] + E["pairing"],
                "eps + flavour Zeeman (control)": E["eps_pos"] + zeeman,
            }
            for name, ens in cases.items():
                r = mtr_classify(E["band"] + ens, rng)
                w.writerow(
                    [
                        lab,
                        name,
                        r["dimL_sym"],
                        r["dimL_anti"],
                        r["Tplus"] is not None,
                        r["Tminus"] is not None,
                        r["cls"] or "none",
                    ]
                )
    print(dst)


if __name__ == "__main__":
    main()
