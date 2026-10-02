#!/usr/bin/env python3
"""Assemble data/MANIFEST.tsv from the per-group fragments data/manifest/*.tsv.

Fragment columns (tab-separated, header line first): kind, path, md5, bytes, produced_by, used_by
  kind      raw (archive file, not committed) | derived | config | reference (committed files)
  path      archive-relative for raw files, repository-relative otherwise
  md5,bytes may be "-" for committed files: they are recomputed here; for raw files they are required.
Every committed file under data/derived and data/configs must appear in some fragment.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import DATA, ROOT, md5  # noqa: E402

COLS = ["kind", "path", "md5", "bytes", "produced_by", "used_by"]


def main():
    rows = {}
    for frag in sorted((DATA / "manifest").glob("*.tsv")):
        lines = frag.read_text().splitlines()
        assert lines[0].split("\t") == COLS, (frag, lines[0])
        for ln in lines[1:]:
            if not ln.strip():
                continue
            r = dict(zip(COLS, ln.split("\t"), strict=True))
            if r["kind"] != "raw":
                p = ROOT / r["path"]
                assert p.exists(), r["path"]
                r["md5"], r["bytes"] = md5(p), str(p.stat().st_size)
            key = (r["kind"], r["path"])
            if key in rows:  # same file used by several groups: merge the users
                old = rows[key]
                users = sorted(set(old["used_by"].split(",")) | set(r["used_by"].split(",")))
                old["used_by"] = ",".join(u for u in users if u)
            else:
                rows[key] = r
    committed = {
        p.relative_to(ROOT).as_posix() for d in ("derived", "configs") for p in (DATA / d).rglob("*") if p.is_file()
    }
    listed = {path for (kind, path) in rows if kind != "raw"}
    missing = sorted(committed - listed)
    assert not missing, f"committed data files without a manifest entry: {missing[:10]}"
    big = [path for (kind, path), r in rows.items() if kind != "raw" and int(r["bytes"]) > 5_000_000]
    assert not big, f"committed files above 5 MB: {big}"
    out = ["\t".join(COLS)] + ["\t".join(r[c] for c in COLS) for _, r in sorted(rows.items())]
    (DATA / "MANIFEST.tsv").write_text("\n".join(out) + "\n")
    print(f"{len(rows)} entries ({sum(1 for k in rows if k[0] == 'raw')} raw)")


if __name__ == "__main__":
    main()
