# Plan: the companion repository of "Mass as pairing"

This file fixes the layout, the claim identifiers, what is carried over from the research notebook and how,
the data strategy, and the audit. Provenance archive: the research notebook `PiotrTopa/unhiggsed_notebook`,
pinned at commit `3c4a02c`.

## 1. Layout

```
README.md            what the repository shows, install, quick reproduce, layout, citation
RESULTS.md           one section per paper claim K1–K8 (+ the closed moonshot M): statement, numbers, links, caveats
CITATION.cff         citation metadata (paper DOI placeholder)
LICENSE              MIT
pyproject.toml       package metadata, bounded dependencies, ruff/black/pytest configuration
Makefile             make test | make check | make results | make figures | make lint
masspairing/         the Python package (CPU reference path; cupy optional)
  lattice.py           d-dimensional hypercubic lattice, staggered phases, kinetic matrix, shift tables
  patterns.py          same-parity pair patterns: nodal C0, plane operators, taste-chiral mass C_chi, taste projector
  wedge.py             links, quads and Hubbard–Stratonovich link fields of the sum-of-squares completion
  operator.py          D_c(sigma, s) = (K - A(s) - h C) (x) 1_2 + i y sigma.tau, flavour-selective source, dense forms
  rational.py          Zolotarev partial fractions (robust on wide windows)
  solvers.py           CG, multishift CG, Lanczos matrix functions
  action.py            bosonic action, pseudofermion action/forces, Hasenbusch split
  hmc.py               Omelyan RHMC, nested multiple-time-scale RHMC, trajectory-length jitter
  measure/             observables: scalar sector, epsilon channel, complete 10/6 channel, N1 taste channels
  symmetry.py          signed-permutation lattice symmetries, group averages (symmetry zeros)
  corner.py            corner blocks of the propagator, parity split, frequency exponent alpha
  pfaffian.py          Pfaffian sign and magnitude (flavour-selective source)
  algebra/             Spin(10) gamma matrices, weights and characters, Grassmann polynomials, eta invariants,
                       fermion bags and Majorana-positivity tools of the four-flavour model
  analysis/            statistics (blocking, jackknife, tau_int) and the estimators of the claims
claims/<ID>/         claim.md (statement, numbers, method, caveats, data) + check.py (uses masspairing)
claims/run_all.py    runs every check, prints the PASS table (make check)
data/MANIFEST.tsv    every data file: path, md5, size, producer, claims that use it (raw files: archive only)
data/derived/        small derived data (time series without configurations, frozen scans) used by checks/figures
data/configs/        a few stored configurations for re-measurement tests
results/             final tables (CSV/JSON), one deterministic script each
figures/             figures, one deterministic script each
scripts/             run_chain.py (the chain runner), make_derived.py, table_*.py, fig_*.py
tests/               pytest: unit tests and new-vs-notebook equivalence tests (frozen references in tests/reference/)
docs/PLAN.md         this file
```

## 2. Claim identifiers

Scheme: `K<n>.<m>` is the m-th certified statement supporting paper claim K<n>; `I.<m>` is an instrument
(method) validation the K claims rest on; `M.1` is the closed moonshot null. Identifiers are stable: a claim
that is later withdrawn keeps its number with status "withdrawn"; new claims get new numbers. The directory of
a claim is `claims/<ID>/`. The notebook claim numbers (`Cnnn`) are kept only in this table and in the
`Provenance` line of each claim.md.

| paper claim | ID | statement (short) | notebook claims |
|---|---|---|---|
| K1 | K1.1 | 16 ⊗ 16 bilinears: Sym² = 10 ⊕ 126, Λ² = 120; ranks of 126 and 10 vevs | C001–C004 |
| K1 | K1.2 | unique holomorphic quartic Φ₁₀²; Φ₁₂₆² ≡ 0; 126 ⊗ 126; the unique octic | C005, C006 |
| K1 | K1.3 | Fierz basis of ψψψ̄ψ̄ quartics {\|Φ₁₀\|², \|Φ₁₂₆\|²} | C009 |
| K1 | K1.4 | ℤ₄ is the unique anomaly-free remnant (eta invariants; instanton vertex ψ⁴) | C007, C010 |
| K1 | K1.5 | Pati–Salam embedding; channel content of the lattice ε and Sym² terms | C008 |
| K1 | K1.6 | the four-flavour lattice model: channel algebra, Fierz identities; U(1)_ε ≠ U(1)_ψ | C020, C101 |
| K1 | K1.7 | ℤ₄² = (−1)^F; the partner algebra of a symmetric gap; composite partner ψ̄ψ̄ψ | C149 |
| K2 | K2.1 | τ₂K: det D_c > 0 configuration by configuration; on-site bilinear structure | C040 (structure part) |
| K2 | K2.2 | the pure Sym² term has no sign-free formulation found (Hamiltonian MTR, bags, moment obstruction); the completed term is sign-free | C021, C022, C023 |
| K2 | K2.3 | anatomy of the completed (wedge) term; wedge optimality theorem; sign-free same-parity source | C100 |
| K2 | K2.4 | the sign-free class is quaternion-linear, hence flavour-democratic | C165 |
| K2 | K2.5 | C₀ is a nodal pairing (no corner mass, also to O(h⁵)); C_χ is the taste-chiral Majorana mass (ε convention) | C147, C151 item 4, C152 items 1–3, C160 item 1 |
| K2 | K2.6 | exact flavour-factorised bag expansion on 2³×4; 2+2 split positive in bag/conjugate-pair form | C148 items 1, 2, 4, 5 |
| K3 | K3.1 | complete χ₁₀ does not grow with volume at five wedge points, L ≤ 8 | C104 |
| K3 | K3.2 | complete χ₁₀ on the wedge family is free-like and suppressed (≤ 0.5 × free) | C130 |
| K3 | K3.3 | pure-10 response irrelevant at P_c: R(8)/R(6) = 1.02(5), 0.97(2) (dense 8⁴) | C107 |
| K3 | K3.4 | no SSB under the same-parity pairing source | C106 |
| K3 | K3.5 | y = 0 LSM exit: bond crystal or gapless fermion, never the (10,3,1) channel | C131 |
| K3 | K3.6 | columnar bond crystal condenses on the +edge (dimer channel) | C073 (dimer content) |
| K4 | K4.1 | composite takeover across SYM → P_c → SMG (nodal source on one flavour, 4⁴/6⁴) | C150 |
| K4 | K4.2 | stage 0 (4⁴, 4³×8): SMG screens the elementary chiral mass; the composite carries it | C190 (G4), C191 |
| K4 | K4.3 | K4 at 8⁴, 6⁴, 6³×12 with C_χ: elementary ≤ 1.1 % of free, composite linear, lattice-independent to 2 % | C195 |
| K4 | K4.4 | production-bc continuity (6⁴ pppa vs aaaa) | C197 |
| K5 | K5.1 | N1 instrument definitions: taste projectors, free baselines, the light-doublet gate | C160 items 2–5, C163 items 1–2 |
| K5 | K5.2 | symmetry zeros: no Lorentz-scalar mass of the light doublet (Majorana or Dirac) without SSB; box-shape domain | C151 items 1–3, C163 item 3, C199 item 2 (+ E9 domain note) |
| K5 | K5.3 | stage 0 gates G1–G4 as pre-registered, with controls | C190 |
| K5 | K5.4 | stage 1 data integrity and error model | C193 |
| K5 | K5.5 | stage 1 by the pre-registered letter: no verdict at L ≤ 8 (recorded) | C194 items 1–3, 5 |
| K5 | K5.6 | control S3: the phase-flipped pattern | C198 |
| K5 | K5.7 | stage 1b integrity; box-shape facts (E9) | C200 |
| K5 | K5.8 | stage 1b verdict at y = 3.0: (a) the light half keeps its own gap (h = 1, 2) | C201 |
| K6 | K6.1 | no frequency winding in the sign-free class; quantisation scale above the bandwidth | C180, C181 |
| K6 | K6.2 | the odd-part exponent α: validation; SYM/SMG separated per configuration | C182, C183 |
| K6 | K6.3 | α_L ensemble readout: saturated Luttinger zero in SMG at every h | C196 |
| K7 | K7.1 | finite-size-scaling crossing of ξ₂/L at P_c from L = 6/8 | C083 |
| K7 | K7.2 | no first-order signature at L ≤ 8 (pilot set, 39 chains) | C051 |
| K7 | K7.3 | 8⁴ three-start hysteresis, inner pair, pooled P_c (2000 trajectories per start) | C171, C192 |
| K7 | K7.4 | ξ(y) at L ≤ 12 cannot separate walking from a power law | C082 |
| K8 | K8.1 | anti-seesaw tier 1: partner-mass lemma; ε/σ collapse; light pair-channel gap rises; quench | C199 items 1, 3 |
| K8 | K8.2 | tier 2 as pre-registered: rise monotone, no SSB, V-stable; not a power law (crossover) | C202 |
| M | M.1 | moonshot trigger not met: the light gap rises at P_c (no critical seesaw at L ≤ 8) | C194 item 4, C202 (C4 alert) |
| — | I.1 | RHMC exact: free limit, forces, reversibility, δτ², exact-determinant Metropolis, 4⁴ phases | C040 (code part), C041, C042, C043 |
| — | I.2 | Hasenbusch split + multiple-time-scale integrator exact | C060 |
| — | I.3 | calibration: y_ours = √2 y_BCHH, P_c = (2.41, −0.01) | C050 |
| — | I.4 | link-field (wedge) HMC exact; per-link completion model exact | C070, C111 |
| — | I.5 | link-field HMC resonance at ω_j τ ≈ nπ; trajectory-length jitter removes it | C102 |
| — | I.6 | complete SU(4)-invariant 10/6-channel estimator (dense = stochastic, boundary sign) | C110 |
| — | I.7 | same-parity source: sign-free, exact HMC, linear response | C112 |
| — | I.8 | flavour-selective source in the real basis: |Pf| RHMC and real-basis observables exact | C140, C142 |
| — | I.9 | Pfaffian sign: three routes agree; pencil flip points | C141 (method part) |
| — | I.10 | N1 RHMC exact vs exact-determinant Metropolis; flipped pattern rejected | C161 |
| — | I.11 | N1 taste observables exact (free identities, stochastic = dense) | C162 |

Where one clean claim merges several notebook claims, the clean check.py runs all the carried-over items; where a
notebook claim is split (C040, C194, C199, C160, C163), each item goes to exactly one clean claim.

## 3. Not carried over

Stays in the notebook (cited once, in README, as the provenance archive):

- **Speed and hardware facts** (no paper claim depends on them): C052 fused GPU solver, C061 integrator criterion
  miss, C080 launch ceiling, C081 solver anatomy, C090 CUDA graphs, C091 eigensolve-free Lanczos, C092 re-timing,
  C093 deflation miss.
- **Superseded instruments and estimators**: C071/C072 (bond-product estimator, one unthermalised row), C103 (the
  identification of that estimator — the clean package carries only the complete estimator), C105 (stochastic 8⁴
  linear response, superseded by the dense C107), C148 item 3 (struck).
- **Notebook-code regression facts**: C121 (fit-fallback bit identity), C122 (checkpoint resume), C143/C164
  (default-path bit identity of the notebook runners), C144 (Lanczos cap flag). Replaced here by the equivalence
  tests of §5.
- **Operator facts of the nodal source with no paper meaning**: C145 (sign wall), C146 (MD stiffness).
- **Notebook history**: lane names, folds, dates as narrative, orders and charters, compute placement and spend,
  dashboards, monitoring exporters, rented-infrastructure tooling (`scripts/compute`, `scripts/rent`,
  `scripts/monitoring`, `scripts/backup`), the original targets T2/T3b/T4 and their reframing.

## 4. Code: kept, refactored, dropped

| notebook module | fate | clean module |
|---|---|---|
| `hmc/lattice.py` + `WedgeLattice` | merged (one d-dimensional lattice) | `lattice.py` |
| `hmc/dirac.py` (CPU part), `hmc/wedge.py` `WedgeDirac` | merged into one operator; custom CUDA stencils dropped | `operator.py` |
| `hmc/majorana.py` `MajoranaSourceDirac` | flavour-selective source kept as an operator option | `operator.py` |
| `hmc/wedge.py` `WedgeGeometry`, source patterns | kept | `wedge.py`, `patterns.py` |
| `hmc/chiral_mass.py` | kept | `patterns.py` |
| `hmc/action.py` | kept; fused/graph/eigensolve-free dispatch dropped | `action.py` |
| `hmc/hmc.py` | kept; state-dependent step count (`adapt`, unused in production) dropped | `hmc.py` |
| `hmc/solvers.py` | CG, multishift CG, Lanczos kept; fused CUDA kernels and `est` hook dropped | `solvers.py` |
| `hmc/rational.py` | kept; the robust construction used as fallback where the plain one is non-finite | `rational.py` |
| `hmc/cf.py`, `hmc/graphs.py` | dropped (speed only; the measure does not depend on them) | — |
| `hmc/checkpoint.py` | dropped (the runner keeps a plain resume from its own npz) | — |
| `hmc/observables.py` | kept | `measure/scalar.py`, `measure/epsilon.py` |
| `WedgeMeasure` | complete estimator only (`chi10="full"`); bond-product estimator and sabotage hooks dropped | `measure/channels.py` |
| `MajoranaMeasure` | kept; sabotage hooks dropped (controls live in check.py) | `measure/n1.py` |
| `hmc/lattice_symmetry.py` | kept | `symmetry.py` |
| `winding.py` | kept | `corner.py` |
| `majorana.py` Pfaffian routines | kept | `pfaffian.py` |
| `spin10.py`, `liealg.py`, `grassmann.py`, `eta.py` | kept | `algebra/` |
| `staggered.py` | the parts the K1/K2 claims use (channel algebra, bags, Hamiltonian MTR classification) | `algebra/` |
| `scripts/run_scan.py`, `run_wedge.py` | merged into one runner; GPU-speed flags dropped | `scripts/run_chain.py` |
| `scripts/analyse.py`, lane analysis scripts, `claims/C051/t3a_lib.py`, `C171/c171_lib.py` | the estimators and verdict logic the claims use | `analysis/` |
| `scripts/laneX/*`, `laneF/*`, `laneE/compare_schemes.py`, `compute/`, `rent/`, `monitoring/`, `backup/`, `so10_explore.py` | dropped | — |

GPU: every array routine takes the array module of its inputs, so cupy works through `cupyx.scipy.sparse`;
there are no custom kernels. The CPU path is the reference and the only one the tests certify.

## 5. Equivalence with the notebook code

- `tests/equivalence/`: the clean package against the notebook implementation on small lattices (2⁴, 2³×4, 4⁴,
  4³×8, 6⁴): operators (sparse and dense), actions, forces, Lanczos values, partial fractions, every observable
  key of the three measures (dense and stochastic with the same seed), HMC trajectories (plain, Hasenbusch/MTS,
  link fields, chiral source, selective source) — bit-identical, or within a stated tolerance ≤ 1e-12 where the
  arithmetic order changed. The notebook outputs are frozen in `tests/reference/` by
  `tests/equivalence/make_reference.py` (needs a notebook checkout, `MASSPAIRING_NOTEBOOK=/path`); the tests
  compare against the frozen files and, when the notebook is present, against a live run too.
- Re-measurement from stored configurations: dense N1 measurements of stored stage-1 configurations reproduce the
  stored time-series entries; stored ε-model configurations reproduce their stored scalar observables.
- Production chains (8⁴, GPU, CUDA-graph and eigensolve-free Lanczos path) are not re-run: that path differs from
  the CPU reference only in solver arithmetic inside the molecular dynamics, not in the measure. Their stored
  outputs are verified as frozen inputs (md5) and by the re-measurements above.

## 6. Data strategy

- **Raw data** (chain npz files with configurations, 0.6 GB; the largest 172 MB) are not committed. Each is listed
  in `data/MANIFEST.tsv` with its path in the archive, md5, size, the run that produced it and the claims that
  use it.
- **Derived data** (committed, each file < 5 MB): per chain, the time series the checks read (no configurations,
  no unused keys); frozen outputs of long computations (free baselines at 8⁴, α-scan block means, symmetry scans);
  a handful of stored configurations for the re-measurement tests. Produced from the raw files by
  `scripts/make_derived.py` (needs the archive) and listed in the manifest with md5.
- Every check.py and every figure runs from the derived data alone. Where a check needs a raw configuration
  set beyond the committed subset, it states so and verifies the frozen derived product instead.
- **Archive proposal (decision for the user):** deposit the raw chain files (the `results/` tree of the notebook
  at `3c4a02c`, ≈ 650 MB, as one tarball per data set) on Zenodo with a DOI, and record that DOI in
  `data/MANIFEST.tsv` and `CITATION.cff`. Alternatives: a GitHub release asset of the tagged notebook (2 GB limit
  per file, no DOI), or Git LFS on the companion repository (quota, no DOI). Nothing is uploaded by this lane.

## 7. Audit (filled in at the end)

See §audit below.

## audit

(pending)
