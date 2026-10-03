#!/usr/bin/env python3
"""Set the DOI of the raw chain archive in every place that carries it.

    python scripts/set_data_doi.py 10.5281/zenodo.1234567

Writes masspairing.data.DATA_DOI, replaces the previous DOI in README.md and CITATION.cff, and regenerates
data/MANIFEST.tsv (its column ``archive``). tests/test_data_doi.py checks that the places agree.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from masspairing.data import DATA_DOI  # noqa: E402

TEXT_FILES = ("README.md", "CITATION.cff")


def main():
    if len(sys.argv) != 2 or not re.fullmatch(r"10\.5281/zenodo\.\d+", sys.argv[1]):
        sys.exit(__doc__)
    new = sys.argv[1]
    data_py = ROOT / "masspairing" / "data.py"
    s, n = re.subn(r'^DATA_DOI = "[^"]*"$', f'DATA_DOI = "{new}"', data_py.read_text(), flags=re.M)
    assert n == 1, "DATA_DOI line not found"
    data_py.write_text(s)
    for name in TEXT_FILES:
        p = ROOT / name
        t = p.read_text()
        assert DATA_DOI in t, f"{name} does not carry {DATA_DOI}"
        p.write_text(t.replace(DATA_DOI, new))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_manifest.py")], check=True)
    print(f"data DOI: {DATA_DOI} -> {new}")


if __name__ == "__main__":
    main()
