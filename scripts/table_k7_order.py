#!/usr/bin/env python3
"""Order of the transition at P_c (claims K7.1, K7.2):

results/K7_hysteresis.csv    the start chains at P_c (6^4 and 8^4) and the pilot 8^4 P_c chain, after the cut
results/K7_verdict.json      the signals, controls and verdicts of the pilot set (K7.1) and the full set (K7.2)
results/K7_phase_labels.csv  the phase label of each of the 44 chains (archive path; configuration array sigma_final)
"""

import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.analysis import t3a  # noqa: E402
from masspairing.data import RESULTS  # noqa: E402

KEYS = ("mabs", "m2", "sig2", "O4", "Spi", "Spm", "xiL", "R4")


def g(x, d=6):
    return float(f"{x:.{d}g}")


def summary(v, keys):
    return t3a.jsonable({k: v[k] for k in keys if k in v})


def main():
    pilot, new = t3a.load_records("pilot"), t3a.load_records("sharpened")
    P = t3a.analyse_pilot(pilot)
    S = t3a.analyse_sharpened(pilot, new, P)
    RESULTS.mkdir(exist_ok=True)

    rows = []
    chains = [("6^4 start", P["verdict"]["hyst"][s]) for s in ("afm", "cold", "hot")]
    chains += [("8^4 start", S["verdict"]["hyst8"][s]) for s in ("afm", "cold", "hot")]
    chains += [("8^4 pilot P_c chain", S["verdict"]["old8"])]
    for role, s in chains:
        r = dict(role=role, L=s["L"], start=s["start"], seed=s["seed"], n=s["n"], acceptance=g(s["acc"], 3))
        r |= dict(
            tau_int_m2=g(s["tau"], 4),
            half_pull_m2=g(s["half_pulls"]["m2"], 3),
            half_pull_sig2=g(s["half_pulls"]["sig2"], 3),
        )
        r |= dict(scored=s["scored"], flags=",".join(s["flags"]))
        for k in KEYS:
            r[k], r["e_" + k] = g(s[k]), g(s["e_" + k], 2)
        rows.append(r)
    with open(RESULTS / "K7_hysteresis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    vp, vs = P["verdict"], S["verdict"]
    out = {
        "K7.1": summary(
            vp,
            (
                "verdict",
                "verdict_prereg_letter",
                "crossing_persists",
                "H1",
                "H1_max",
                "H1_pulls",
                "H2",
                "H2_bimodal",
                "H2_Ve",
                "H2_Ve_ratio",
                "H3",
                "H3_theta",
                "H4",
                "H4_gamma_nu",
                "H4_eta",
                "H4_gridmax",
                "H5",
                "H5_dR4",
                "pc_scored",
                "controls",
            ),
        ),
        "K7.2": summary(
            vs,
            (
                "verdict",
                "crossing_persists",
                "crossing",
                "H1",
                "H1_6_max",
                "H1_8",
                "H1_8_max",
                "H1_8_pulls",
                "H1_8_energy_max",
                "replica4_max",
                "old_vs_new_max",
                "new_vs_new_max",
                "H2",
                "H2_bimodal_new",
                "H2_Ve",
                "H2_Ve_ratio",
                "H3",
                "H3_power",
                "H3_theta",
                "H3_drift",
                "xiL_max_y",
                "H4",
                "H4_unpooled",
                "H4_pooled",
                "H5",
                "H5_unpooled",
                "H5_pooled",
                "pc_scored",
                "controls",
            ),
        ),
    }
    (RESULTS / "K7_verdict.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    labels = P["labels"] + S["labels"]
    cols = ["file", "config_array", "L", "kappa", "y", "start", "seed", "ntraj", "acceptance", "phase", "near_critical"]
    cols += ["Sigma_stag_abs", "e_Sigma_stag_abs", "O4", "e_O4", "xiL", "e_xiL", "tau_int_m2", "scored", "flags"]
    with open(RESULTS / "K7_phase_labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, lineterminator="\n")
        w.writeheader()
        for x in labels:
            r = {k: x[k] for k in cols if k in x}
            for k in ("Sigma_stag_abs", "O4", "xiL"):
                r[k], r["e_" + k] = x[k]
            r["flags"] = ",".join(x["flags"])
            w.writerow(r)
    print(
        f"results/K7_hysteresis.csv ({len(rows)} chains), K7_verdict.json, K7_phase_labels.csv ({len(labels)} chains)"
    )


if __name__ == "__main__":
    main()
