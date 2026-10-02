PY ?= .venv/bin/python
JOBS ?= 4
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1

.PHONY: venv test check results figures lint derived reference

venv:
	python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

test:            ## unit and equivalence tests (minutes, CPU)
	$(PY) -m pytest

check:           ## every claim's check.py, PASS table (claims/STATUS.md)
	$(PY) claims/run_all.py -j $(JOBS)

results:         ## final tables in results/
	for s in scripts/table_*.py; do $(PY) $$s || exit 1; done

figures:         ## figures in figures/
	for s in scripts/fig_*.py; do $(PY) $$s || exit 1; done

trace:          ## every number in RESULTS.md found in the tagged check outputs (after make check)
	$(PY) scripts/trace_results.py

lint:
	$(PY) -m ruff check masspairing scripts tests claims
	$(PY) -m black --check masspairing scripts tests claims

derived:         ## rebuild data/derived from the raw archive (MASSPAIRING_ARCHIVE=/path)
	for s in scripts/derive_*.py; do $(PY) $$s || exit 1; done
	$(PY) scripts/make_manifest.py

reference:       ## refreeze the notebook outputs of the equivalence tests (MASSPAIRING_NOTEBOOK=/path)
	$(PY) tests/equivalence/make_reference.py
