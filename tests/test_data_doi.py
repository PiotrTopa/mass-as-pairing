"""The data DOI is the same everywhere it appears, and every raw file of the manifest lies in exactly one bundle."""

from masspairing.data import ARCHIVE_BUNDLES, DATA, DATA_DOI, ROOT, archive_bundle


def test_doi_consistent():
    assert DATA_DOI in (ROOT / "README.md").read_text()
    assert f'value: "{DATA_DOI}"' in (ROOT / "CITATION.cff").read_text()
    lines = (DATA / "MANIFEST.tsv").read_text().splitlines()
    cols = lines[0].split("\t")
    rows = [dict(zip(cols, ln.split("\t"), strict=True)) for ln in lines[1:] if ln]
    raw = [r for r in rows if r["kind"] == "raw"]
    assert raw
    assert all(r["archive"] == f"{DATA_DOI}:{archive_bundle(r['path'])}" for r in raw)
    assert all(r["archive"] == "-" for r in rows if r["kind"] != "raw")
    assert {archive_bundle(r["path"]) for r in raw} == set(ARCHIVE_BUNDLES)
