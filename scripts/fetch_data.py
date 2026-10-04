#!/usr/bin/env python3
"""Download the raw chain archive from Zenodo (DOI masspairing.data.DATA_DOI), verify it and unpack it.

    python scripts/fetch_data.py --out archive                     # all five bundles (about 0.6 GB download)
    python scripts/fetch_data.py --out archive --bundle eps-model  # only the bundles whose name contains the argument
    python scripts/fetch_data.py --out archive --verify-only       # no network: check an unpacked archive
    make derived MASSPAIRING_ARCHIVE=archive                       # then rebuild data/derived from it

Each bundle is checked against the md5 that Zenodo publishes for it; after unpacking, every raw file of the chosen
bundles is checked against its md5 and size in data/MANIFEST.tsv. Exit 1 on any mismatch or missing file.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import tarfile
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import ARCHIVE_BUNDLES, DATA, DATA_DOI, DATA_DOI_PLACEHOLDER, md5  # noqa: E402

API = "https://zenodo.org/api/records/"


def manifest_rows():
    """Raw rows of data/MANIFEST.tsv: (path, md5, bytes, bundle)."""
    lines = (DATA / "MANIFEST.tsv").read_text().splitlines()
    cols = lines[0].split("\t")
    rows = [dict(zip(cols, ln.split("\t"), strict=True)) for ln in lines[1:] if ln]
    return [(r["path"], r["md5"], int(r["bytes"]), r["archive"].rsplit(":", 1)[1]) for r in rows if r["kind"] == "raw"]


def record_files(doi):
    """{file name: (download url, md5, size)} of the Zenodo record of ``doi`` (10.5281/zenodo.<record id>)."""
    recid = doi.rsplit("zenodo.", 1)[1]
    with urllib.request.urlopen(API + recid, timeout=60) as r:
        rec = json.load(r)
    out = {}
    for f in rec["files"]:
        algo, digest = f["checksum"].split(":", 1)
        assert algo == "md5", f["checksum"]
        out[f["key"]] = (f["links"]["self"], digest, int(f["size"]))
    return out


def download(url, dst, want_md5):
    h = hashlib.md5()
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
            h.update(chunk)
    if h.hexdigest() != want_md5:
        tmp.unlink()
        sys.exit(f"md5 mismatch for {dst.name}: {h.hexdigest()} != {want_md5}")
    tmp.rename(dst)


def unpack(bundle, out):
    with tarfile.open(bundle) as tf:
        for m in tf.getmembers():
            p = pathlib.PurePosixPath(m.name)
            if not (m.isfile() or m.isdir()) or p.is_absolute() or ".." in p.parts or p.parts[0] != "results":
                sys.exit(f"{bundle.name}: unexpected member {m.name}")
        tf.extractall(out, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def verify(out, rows):
    bad = 0
    for rel, want, size, _ in rows:
        p = out / rel
        if not p.is_file():
            print("MISSING", rel)
            bad += 1
        elif p.stat().st_size != size or md5(p) != want:
            print("MISMATCH", rel)
            bad += 1
    print(f"{len(rows) - bad}/{len(rows)} raw files verified against data/MANIFEST.tsv")
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=pathlib.Path, help="archive root to create (holds results/)")
    ap.add_argument("--bundle", action="append", default=[], help="substring of the bundle names to fetch")
    ap.add_argument("--verify-only", action="store_true", help="only check an unpacked archive (no network)")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded bundles in <out>/bundles")
    a = ap.parse_args()
    bundles = [b for b in ARCHIVE_BUNDLES if not a.bundle or any(s in b for s in a.bundle)]
    if not bundles:
        sys.exit(f"no bundle matches {a.bundle}; bundles: {list(ARCHIVE_BUNDLES)}")
    rows = [r for r in manifest_rows() if r[3] in bundles]
    if not a.verify_only:
        if DATA_DOI == DATA_DOI_PLACEHOLDER:
            sys.exit("the data DOI is not set yet (masspairing.data.DATA_DOI; scripts/set_data_doi.py)")
        files = record_files(DATA_DOI)
        store = a.out / "bundles"
        store.mkdir(parents=True, exist_ok=True)
        for b in bundles:
            url, want, _ = files[b]
            print(f"downloading {b}", flush=True)
            download(url, store / b, want)
            unpack(store / b, a.out)
            if not a.keep:
                (store / b).unlink()
        if not a.keep:
            store.rmdir()
    sys.exit(0 if verify(a.out, rows) else 1)


if __name__ == "__main__":
    main()
