"""The data DOI is the same everywhere it appears, and every raw file of the manifest lies in exactly one bundle or is
marked pending (not yet deposited)."""

import re

from masspairing.data import ARCHIVE_BUNDLES, ARCHIVE_PENDING, DATA, DATA_DOI, ROOT, archive_bundle


def test_doi_consistent():
    assert DATA_DOI in (ROOT / "README.md").read_text()
    assert f'value: "{DATA_DOI}"' in (ROOT / "CITATION.cff").read_text()
    lines = (DATA / "MANIFEST.tsv").read_text().splitlines()
    cols = lines[0].split("\t")
    rows = [dict(zip(cols, ln.split("\t"), strict=True)) for ln in lines[1:] if ln]
    raw = [r for r in rows if r["kind"] == "raw"]
    assert raw
    pending = [r for r in raw if r["archive"] == "pending"]
    deposited = [r for r in raw if r["archive"] != "pending"]
    assert all(re.match(ARCHIVE_PENDING, r["path"]) for r in pending)
    assert all(not any(re.match(rx, r["path"]) for rx in ARCHIVE_BUNDLES.values()) for r in pending)
    assert all(r["archive"] == f"{DATA_DOI}:{archive_bundle(r['path'])}" for r in deposited)
    assert all(r["archive"] == "-" for r in rows if r["kind"] != "raw")
    assert {archive_bundle(r["path"]) for r in deposited} == set(ARCHIVE_BUNDLES)
