# Mass as pairing — companion code and certified claims

This repository holds the lattice code, the data products and the machine-checked claims behind the paper
*Mass as pairing*. Fermion mass is read as pairing with a partner of conjugate charge: the Higgs builds the partner
from another elementary fermion, the seesaw uses the self-conjugate ν_R with a ℤ₄-odd Majorana mass, and symmetric mass
generation (SMG) builds it from the fermions themselves (ψ̄ψ̄ψ ∈ 16̄). The repository establishes, for one Spin(10)
generation and its lattice proxy (four reduced staggered flavours with an ε vertex):

- **K1** the structure and partner algebra (ℤ₄ the unique anomaly-free remnant, ℤ₄² = (−1)^F: no Majorana
  (self-conjugate) pole in a ℤ₄-symmetric phase, while the Dirac pole of ψ with ψ̄ψ̄ψ is allowed);
- **K2** the sign-free class is flavour-democratic;
- **K3** the ℤ₄-odd elementary (126) channel does not condense on the sign-free family (L ≤ 8);
- **K4** composite takeover: in the SMG phase the Majorana response sits in the ε-vertex composite;
- **K5** with an explicit Majorana mass on half of the generation, no light mass term — flavour-symmetric or flavour
  sextet — arises without spontaneous breaking of the symmetry group of the action (exact; the lattice group with the
  one-site shifts and the flavour SO(4), the point group alone leaving two sextet terms), and in the SMG phase the other
  half keeps a symmetric gap of nearly unchanged size, with a small drift toward free (replicated on independent
  chains); spontaneous breaking at the largest explicit mass is not resolved at L ≤ 8 — the growth of the light pair
  susceptibility on (6⁴, 8⁴) compares a box with a partial taste split (19.8 % of the momenta) with a complete one, and
  on the complete pair (4⁴, 8⁴) that susceptibility relative to free falls with the volume;
- **K6** the corner-block frequency exponent as a per-configuration pole/zero readout;
- **K7** the SMG transition shows no first-order signature at L ≤ 8;
- **K8** at the critical point the explicit mass moves the critical line to larger y (large-N sign; the light
  observables move in that direction on 6⁴ and 8⁴): the critical ε/σ channel collapses and the other half de-criticalises
  toward free, with no seesaw and no spontaneous breaking — a crossover, not a power law;
- **M** the critical-seesaw moonshot is closed as a null;
- **N** the ℤ₄ as a neutrino selection rule: no Majorana neutrino mass and no |ΔL| = 2 while it is exact.

Statements, numbers and caveats: [RESULTS.md](RESULTS.md). Every claim is a directory `claims/<ID>/` with `claim.md`
(statement, numbers, method, caveats, data) and `check.py` (which recomputes it and prints `PASS <ID>`); the status table
is [claims/STATUS.md](claims/STATUS.md).

## Install

Python ≥ 3.11, CPU only (cupy optional for GPU runs of the chain runner).

```
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'        # numpy, scipy; pytest, ruff, black, matplotlib
```

Tested with the versions in `requirements-lock.txt`.

## Reproduce

```
make test       # unit and equivalence tests (minutes)
make check      # every claim's check.py and the PASS table (claims/STATUS.md); about 1 h on 4 cores
make results    # tables in results/
make figures    # figures in figures/
```

All of these run from the committed derived data. Rebuilding the derived data needs the raw chain archive, deposited
on Zenodo (DOI 10.5281/zenodo.23117210, five bundles, about 0.7 GB; files, md5 sums and bundles in
[data/MANIFEST.tsv](data/MANIFEST.tsv); six small raw files of K8.4–K8.9, marked `pending` there, are not yet in the
deposit — their derived data are committed, so every check runs):

```
python scripts/fetch_data.py --out archive            # download, check md5 sums, unpack
make derived MASSPAIRING_ARCHIVE=archive
```

New chains: `python scripts/run_chain.py --help`.

## Layout

```
masspairing/        the package: lattice, patterns, link fields, operator, action, HMC, measures, analysis, algebra
claims/<ID>/        claim.md + check.py per certified claim; claims/run_all.py runs them all
data/derived/       derived data (time series without configurations, frozen scans); data/configs/ stored configurations
data/MANIFEST.tsv   every data file: path, md5, size, producer, users; raw archive files with their Zenodo bundle
results/  figures/  final tables and figures, one script each in scripts/
tests/              unit tests, equivalence tests against the notebook implementation (frozen references)
```

Claim identifiers: `K<n>.<m>` supports paper claim K<n>; `I.<m>` certifies an instrument; `M.1` is the moonshot null;
`N.1` holds the neutrino selection rules.

## Provenance

The research history (all runs, superseded analyses, exploratory claims) is archived in the notebook repository
`PiotrTopa/unhiggsed_notebook`, commit `3c4a02c`.

## How to cite

See [CITATION.cff](CITATION.cff) (paper DOI to be added). License: MIT (code); the data archive is CC BY 4.0.
