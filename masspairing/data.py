"""Locations and formats of the repository's data.

* ``data/derived/<group>/<name>.npz`` -- derived data committed with the repository: chain time series without
  configurations (keys ``ts_*``), frozen scan outputs, and a JSON ``meta`` record with the chain parameters, the source
  file (path in the raw archive), its md5 and size.
* ``data/configs/`` -- a few stored configurations for re-measurement tests.
* the raw archive -- the chain files with configurations, not committed; ``MASSPAIRING_ARCHIVE`` points at its root
  (the directory that contains ``results/``). Only ``scripts/derive_*.py`` and tests marked ``archive`` need it. It is
  deposited on Zenodo under ``DATA_DOI`` as the bundles of ``ARCHIVE_BUNDLES``; ``scripts/fetch_data.py`` downloads,
  verifies and unpacks it.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = DATA / "derived"
CONFIGS = DATA / "configs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# DOI of the raw chain archive on Zenodo. The placeholder is replaced everywhere (here, README.md, CITATION.cff,
# data/MANIFEST.tsv) by `python scripts/set_data_doi.py <doi>`.
DATA_DOI_PLACEHOLDER = "10.5281/zenodo.XXXXXXX"
DATA_DOI = "10.5281/zenodo.23117210"

# the bundles of the deposit: name -> archive paths it holds (regular expression on the archive-relative path)
ARCHIVE_BUNDLES = {
    "mass-as-pairing-data_eps-model.tar.gz": r"results/xi_scan/(F_|F2_|F5_|F12_|L6_k|L8_k)"
    r"|results/(calib|hmc_validation|laneE|laneK2)/",
    "mass-as-pairing-data_wedge.tar.gz": r"results/xi_scan/(F8_|F9_)|results/(laneW2|laneR)/",
    "mass-as-pairing-data_nodal-source.tar.gz": r"results/xi_scan/S_prod/|results/laneS/",
    "mass-as-pairing-data_n1.tar.gz": r"results/xi_scan/K_N1_(?!S1c/)|results/(laneK5S1A|laneK5S1bA|laneK4Z)/"
    r"|results/laneK1/(?!free_L6_aaaa_h0\.25_0\.5_0\.75\.json$)",
    "mass-as-pairing-data_n1-stage1c.tar.gz": r"results/xi_scan/K_N1_S1c/"
    r"|results/laneK1/free_L6_aaaa_h0\.25_0\.5_0\.75\.json$",
    "mass-as-pairing-data_n1-k8.tar.gz": r"results/(laneTH|laneTHP)/",
}


# raw archive files that no bundle of the deposit holds yet (archive column "pending" in data/MANIFEST.tsv); none
ARCHIVE_PENDING = r"(?!)"


def archive_bundle(rel):
    """The bundle of the deposit that holds the archive file ``rel`` (exactly one)."""
    hits = [b for b, rx in ARCHIVE_BUNDLES.items() if re.match(rx, rel)]
    assert len(hits) == 1, (rel, hits)
    return hits[0]


def archive_column(rel):
    """The manifest's archive entry of a raw file: "<DATA_DOI>:<bundle>", or "pending" (not yet deposited)."""
    if re.match(ARCHIVE_PENDING, rel):
        assert not any(re.match(rx, rel) for rx in ARCHIVE_BUNDLES.values()), rel
        return "pending"
    return f"{DATA_DOI}:{archive_bundle(rel)}"


def archive_root():
    """Root of the raw archive (env MASSPAIRING_ARCHIVE) or None."""
    p = os.environ.get("MASSPAIRING_ARCHIVE")
    return pathlib.Path(p) if p else None


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def strip_chain(src, dst, keys=None, extra_meta=None, rel=None):
    """Write the ts_ series ``keys`` (all ts_ keys except ts_cfg if None) of chain file ``src`` to ``dst`` (compressed).

    The meta record keeps the chain's own meta and adds source=rel (archive-relative path), source_md5, source_bytes.
    """
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    d = np.load(src, allow_pickle=True)
    if keys is None:
        keys = [k for k in d.files if k.startswith("ts_") and k != "ts_cfg"]
    out = {k: d[k] for k in keys if k in d.files}
    missing = [k for k in keys if k not in d.files]
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    meta.update(source=str(rel or src), source_md5=md5(src), source_bytes=src.stat().st_size, missing_keys=missing)
    if extra_meta:
        meta.update(extra_meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, meta=json.dumps(meta, default=str), **out)
    return dst


def load_chain(path):
    """(series dict, meta dict) of a derived chain file."""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return {k: d[k] for k in d.files if k != "meta"}, meta


def derived(*parts):
    """Path inside data/derived."""
    return DERIVED.joinpath(*parts)
