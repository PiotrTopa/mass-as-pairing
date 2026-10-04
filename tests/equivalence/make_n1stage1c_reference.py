#!/usr/bin/env python3
"""Copy the notebook's frozen stage-1c outputs into tests/reference/n1stage/ (needs a notebook checkout).

    MASSPAIRING_NOTEBOOK=/path/to/unhiggsed_notebook python tests/equivalence/make_n1stage1c_reference.py

results/laneS1cA/{rows_1c,verdict_1c,sensitivity}.json of the notebook (commit fd0c934) become rows_1c.json,
verdict_1c.json, sensitivity_1c.json; absolute paths are cut to archive-relative ones (from ``results/`` on).
"""

import json
import os
import pathlib
import re

OUT = pathlib.Path(__file__).resolve().parents[1] / "reference" / "n1stage"
FILES = {
    "rows_1c.json": "rows_1c.json",
    "verdict_1c.json": "verdict_1c.json",
    "sensitivity.json": "sensitivity_1c.json",
}


def relative(o):
    if isinstance(o, dict):
        return {k: relative(v) for k, v in o.items()}
    if isinstance(o, list):
        return [relative(v) for v in o]
    if isinstance(o, str) and o.startswith("/"):
        return re.sub(r"^.*?/(results/)", r"\1", o)
    return o


def main():
    src = pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"]) / "results" / "laneS1cA"
    OUT.mkdir(parents=True, exist_ok=True)
    for a, b in FILES.items():
        data = relative(json.loads((src / a).read_text()))
        (OUT / b).write_text(json.dumps(data, indent=1) + "\n")
        print("written", OUT / b)


if __name__ == "__main__":
    main()
