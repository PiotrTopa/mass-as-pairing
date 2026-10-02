#!/usr/bin/env python3
"""Derived data of the K3 claims and of the instruments I.4-I.7 (writes data/derived/k3/, data/configs/k3/).

    MASSPAIRING_ARCHIVE=/path/to/archive python scripts/derive_k3.py series     # chain series from the raw archive
    python scripts/derive_k3.py chains [name ...]                              # small exactness chains (no archive)
    python scripts/derive_k3.py bags                                           # 2^3 fermion-bag references (~1 h)

``series``: for every stored chain the claims use, the ts_ series the analyses read (the summarised scalar and
fermion series, acceptance and dH, the first 8 corner structure factors where present, the fermion correlator
G_f for the y = 0 exit scan, the link-field variance for the resonance check), cut at the first trajectory the
claims keep where that removes many trajectories (the cut is recorded in the meta record as ``k0_stored`` with
the raw length ``n_stored``); a few final configurations go to data/configs/k3/. Writes data/manifest/k3.tsv.

``chains``: the exactness chains of I.4 and I.7 (exact-determinant Metropolis vs the RHMC on 2^3, 2^4, 2^3 x 4,
fixed seeds) with the clean package; each check regenerates a prefix and compares it bit by bit.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from masspairing.analysis import k3
from masspairing.data import DATA, ROOT, archive_root, md5

OUT = DATA / "derived" / "k3"
CFG = DATA / "configs" / "k3"
X = "results/xi_scan"
W2 = "results/laneW2"

BASE = ("dH", "accepted", "ferm_flag") + k3.SCALAR_KEYS + k3.FERMION_KEYS
CORNERS = ("chi10_corners", "chi10_pmin", "chi6_corners")
KEYS = {
    "full": BASE + CORNERS,
    "full+Gf": BASE + CORNERS + ("Gf",),
    "scalar": BASE,
    "s2": ("dH", "accepted", "s2"),
}

# (archive path, first trajectory kept in the derived file, key set, claims)
P1_PTS = [(2.41, 0.05, 0.05), (2.41, 0.1, 0.1), (3.0, 0.1, 0.1), (2.41, 0.05, 0.0), (2.41, 0.02, 0.0)]
X_PTS = [
    (0.0, 0.05, 0.0),
    (0.0, 0.1, 0.0),
    (0.0, 0.05, -0.05),
    (0.0, 0.1, -0.1),
    (2.41, 0.05, -0.05),
    (2.41, 0.1, -0.1),
]
NCOPY = {("compl", 0.01): 1268, ("compl", 0.02): 1071, ("wedge", 0.01): 1500, ("wedge", 0.02): 1500}
K0_STOCH8 = {("compl", 0.01): 300, ("compl", 0.02): 200, ("wedge", 0.01): 400, ("wedge", 0.02): 300}


def p1_name(L, y, g, g6):
    return f"L{L}_y{y:g}_k-0.01_g{g:g}_g6{g6:g}.npz"


def p3_name(m, g, L):
    return f"L{L}_y2.41_k-0.01_g{g:g}_g60{'_links' if m == 'compl' else ''}.npz"


def series_jobs():
    jobs = []
    for p in P1_PTS:
        for L in (4, 6, 8):
            k0 = 600 if (L, p) == (8, (2.41, 0.02, 0.0)) else 0
            jobs.append((f"{X}/F8_P1_wedge/{p1_name(L, *p)}", k0, "full", "K3.1,K3.2"))
    for r in ("rA", "rB"):
        jobs.append((f"{X}/F8_P1_wedge_{r}/{p1_name(8, 2.41, 0.02, 0.0)}", 1400, "full", "K3.1,K3.2"))
    for g6 in (0.05, 0.0):
        for L in (4, 6, 8):
            for h in (0.005, 0.01, 0.02):
                jobs.append((f"{X}/F8_P2_src/L{L}_y2.41_k-0.01_g0.05_g6{g6:g}_h{h:g}.npz", 0, "scalar", "K3.4"))
    for L in (4, 6):
        jobs.append((f"{X}/F8_P3_eps/L{L}_y2.41_k-0.01_g0_g60.npz", 0, "full", "K3.2"))
        for m in ("compl", "wedge"):
            for g in (0.01, 0.02):
                jobs.append((f"{X}/F8_P3_{m}/{p3_name(m, g, L)}", 0, "scalar", "K3.3"))
    for m in ("compl", "wedge"):
        for g in (0.01, 0.02):
            jobs.append((f"{X}/F8_P3x_{m}/{p3_name(m, g, 8)}", K0_STOCH8[(m, g)], "scalar", "K3.3"))
            jobs.append((f"{X}/F8_P3x_{m}_rA/{p3_name(m, g, 8)}", NCOPY[(m, g)] + 50, "scalar", "K3.3"))
    for g, k0 in ((0.01, 1100), (0.02, 900)):
        for r in ("rA", "rB"):
            jobs.append((f"{X}/F8_P3_compl_{r}/{p3_name('compl', g, 8)}", k0, "scalar", "K3.3"))
    for p in X_PTS:
        for L in (4, 6, 8):
            jobs.append((f"{X}/F9_X_y0exit/{p1_name(L, *p)}", 0, "full+Gf", "K3.5"))
    for y, g in ((2.41, 0.05), (3.0, 0.1), (2.41, 0.1)):
        tag = f"_y{y:g}_k-0.01_g{g:g}_g6{g:g}.npz"
        n20 = f"{W2}/c072/n20/L4{tag}"
        jobs.append((n20 if (y, g) == (2.41, 0.05) else f"{W2}/c072/L4{tag}", 0, "scalar", "K3.6"))
        for L in (6, 8):
            jobs.append((f"{W2}/c073/main/L{L}{tag}", 0, "scalar", "K3.6"))
    for sub in ("afmdimer", "hothot"):
        jobs.append((f"{W2}/c073/{sub}/L6_y2.41_k-0.01_g0.1_g60.1.npz", 0, "scalar", "K3.6"))
    for rel in C102_FILES:
        jobs.append((rel, 0, "s2", "I.5"))
    for tau in ("1.0", "0.7"):
        jobs.append((f"results/laneR/c102/tau{tau}/L4_y1_k-0.01_g0.05_g60.npz", 0, "s2", "I.5"))
    return jobs


C102_FILES = [
    f"{W2}/c072/L4_y1_k-0.01_g0.05_g60.npz",
    f"{W2}/c072/L4_y1.7_k-0.01_g0.05_g60.npz",
    f"{W2}/c072/n20/L4_y2.41_k-0.01_g0.05_g60.npz",
    f"{W2}/c072/n20/L4_y3_k-0.01_g0.05_g60.npz",
    f"{W2}/c072/L4_y2.41_k-0.01_g0.02_g60.npz",
    f"{W2}/c072/L4_y2.41_k-0.01_g0.05_g60.05.npz",
]
CONFIGS = {  # configuration name -> (archive path of the chain whose final configuration is kept, claims)
    "c072_L4_y1_g0.05_g60": (C102_FILES[0], "I.5"),
    "c072_L4_y1.7_g0.05_g60": (C102_FILES[1], "I.5"),
    "c072n20_L4_y2.41_g0.05_g60": (C102_FILES[2], "I.5"),
    "c072n20_L4_y3_g0.05_g60": (C102_FILES[3], "I.5"),
    "c072_L4_y2.41_g0.02_g60": (C102_FILES[4], "I.5"),
    "c072_L4_y2.41_g0.05_g60.05": (C102_FILES[5], "I.5,I.6"),
    "c072_L4_y2.41_g0.1_g60": (f"{W2}/c072/L4_y2.41_k-0.01_g0.1_g60.npz", "I.6"),
    "c073_L6_y2.41_g0.1_g60.1": (f"{W2}/c073/main/L6_y2.41_k-0.01_g0.1_g60.1.npz", "I.6"),
}


def strip_cut(src, dst, rel, k0, keys):
    """Write the series ``keys`` of chain ``src`` from trajectory k0 on (corner arrays: first 8 corners)."""
    d = np.load(src, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    series = {k: d[k] for k in d.files if k.startswith("ts_")}
    if "ts_chi10_corners" not in series:  # pre-complete-estimator chain: its chi10/E10 are the bond-product channel
        for k in ("ts_chi10", "ts_E10"):
            series.pop(k, None)
    nt = k3.n_traj(series)
    c = k3.cut(series, k0)
    out = {}
    for k in keys:
        a = c.get("ts_" + k)
        if a is None:
            continue
        out["ts_" + k] = a[:, : k3.NCORNER] if k in CORNERS else a
    meta.update(
        source=rel,
        source_md5=md5(src),
        source_bytes=src.stat().st_size,
        k0_stored=int(k0),
        n_stored=int(nt),
        corners_kept=k3.NCORNER if "ts_chi10_corners" in out else 0,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, meta=json.dumps(meta, default=str), **out)
    return dst


def derive_series():
    arch = archive_root()
    assert arch is not None, "set MASSPAIRING_ARCHIVE"
    rows = []
    for rel, k0, keyset, users in series_jobs():
        src = arch / rel
        dst = strip_cut(src, k3.derived_path(rel), rel, k0, KEYS[keyset])
        rows.append(("raw", rel, md5(src), str(src.stat().st_size), "chain run (archive)", users))
        rows.append(("derived", dst.relative_to(ROOT).as_posix(), "-", "-", "scripts/derive_k3.py", users))
        print(f"{dst.relative_to(ROOT)}  ({dst.stat().st_size / 1e3:.0f} kB)", flush=True)
    for name, (rel, users) in CONFIGS.items():
        src = arch / rel
        d = np.load(src, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        meta.update(source=rel, source_md5=md5(src), source_bytes=src.stat().st_size)
        dst = CFG / f"{name}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, fields=np.asarray(d["fields_final"]).reshape(-1), meta=json.dumps(meta, default=str))
        rows.append(("raw", rel, md5(src), str(src.stat().st_size), "chain run (archive)", users))
        rows.append(("config", dst.relative_to(ROOT).as_posix(), "-", "-", "scripts/derive_k3.py", users))
    write_manifest(rows, part="series")


# ---- fermion-bag references on 2^3 ---------------------------------------------------------------------------
BAG_POINTS = {"interior": (0.4, 0.1, "completed"), "edge+": (0.5, 0.5, "completed"), "pure": (0.5, 0.5, "pure")}


def derive_bags():
    """All-orders bag enumeration of the 2^3 link-field model (interior point about 1 h on one core)."""
    from masspairing.analysis import k3_exact

    out = OUT / "bags_2x2x2.json"
    ref = json.loads(out.read_text()) if out.exists() else {}
    for name, (g10, g6, kind) in BAG_POINTS.items():
        if name in ref:
            continue
        r = k3_exact.bag_reference(k3_exact.model_2x3(g10, g6).geom, kind)
        ref[name] = dict(g10=g10, g6=g6, kind=kind, **{k: float(v) for k, v in r.items()})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ref, indent=1) + "\n")
        print(name, ref[name], flush=True)
    write_manifest(
        [("derived", out.relative_to(ROOT).as_posix(), "-", "-", "scripts/derive_k3.py bags", "I.4")], part="chains"
    )


# ---- manifest -----------------------------------------------------------------------------------------------------
def write_manifest(rows, part):
    """data/manifest/k3.tsv: the rows of this part (series | chains) replace its previous rows."""
    path = DATA / "manifest" / "k3.tsv"
    head = "kind\tpath\tmd5\tbytes\tproduced_by\tused_by"
    old = path.read_text().splitlines()[1:] if path.exists() else []
    is_chain = lambda ln: "/k3/chains/" in ln.split("\t")[1] or "bags_2x2x2" in ln
    new = {}
    for r in rows:
        cur = new.setdefault((r[0], r[1]), list(r))
        cur[5] = ",".join(sorted(set(cur[5].split(",")) | set(r[5].split(","))))
    keep = [ln for ln in old if ln.strip() and tuple(ln.split("\t")[:2]) not in new]
    if part == "series":
        keep = [ln for ln in keep if is_chain(ln)]
    lines = keep + ["\t".join(r) for r in new.values()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([head] + sorted(set(lines))) + "\n")


def main(argv):
    what = argv[0] if argv else "series"
    if what == "series":
        derive_series()
    elif what == "chains":
        from masspairing.analysis import k3_exact

        rows = []
        for name in k3_exact.CHAINS if len(argv) < 2 else argv[1:]:
            dst = k3_exact.generate(name, OUT / "chains")
            rows.append(
                (
                    "derived",
                    dst.relative_to(ROOT).as_posix(),
                    "-",
                    "-",
                    "scripts/derive_k3.py",
                    k3_exact.CHAINS[name]["used_by"],
                )
            )
            print(f"{dst.relative_to(ROOT)}  ({dst.stat().st_size / 1e3:.0f} kB)", flush=True)
        write_manifest(rows, part="chains")
    elif what == "bags":
        derive_bags()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
