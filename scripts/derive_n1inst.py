#!/usr/bin/env python3
"""Derived data of the N1 instrument claims (K4.1, K5.1, K5.2, K5.12, I.8-I.11, K6.1, K6.2).

    MASSPAIRING_ARCHIVE=/path python scripts/derive_n1inst.py                 # every part
    MASSPAIRING_ARCHIVE=/path python scripts/derive_n1inst.py free configs    # selected parts

Parts (outputs under data/derived/n1inst/ unless stated):
  free      the frozen free baselines of the N1 observables, copied verbatim from the archive into
            data/derived/free/ (regenerable with masspairing.analysis.free.free_point; the checks regenerate the
            4^4, 4^3 x 8 and two 6^4 rows live)
  configs   stored configurations: data/configs/n1inst_*.npz (two 4^4 and two 6^4 equilibrium configurations)
  takeover  time series (phi_T, phi_f, Pfaffian sign) of the 23 chains with the nodal source on flavour 4
  winding   corner-block loop readouts of the 39 phase-labelled stored configurations (K6_winding_stored.json)
  boxshape  free light amplitude on L^3 x 2L and hypercubic boxes (K52_box_shape.json)
  massstrings  group averages of the 16 corner-space scalar mass strings on 4^4 pppp/aaaa (K512_mass_strings.json)
  i8        exact Metropolis (|Pf| weight) vs RHMC on 2^3 x 4, flavour-selective source (I8_chains.npz)
  i10       exact Metropolis (Woodbury) vs RHMC on 4^4 aaaa, taste-chiral mass, and the flipped pattern (I10_*.npz)
  manifest  data/manifest/n1inst.tsv (raw sources with md5/size, derived files)
Parts without archive input (boxshape, massstrings, i8, i10) run without MASSPAIRING_ARCHIVE. Single-threaded; i10
takes a few hours, i8 about half an hour, massstrings about a quarter of an hour.
"""

import json
import os
import pathlib
import shutil
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np

from masspairing.analysis import mass_strings as MS
from masspairing.analysis import n1inst as X
from masspairing.data import CONFIGS, DERIVED, ROOT, archive_root, md5, strip_chain

OUT = DERIVED / "n1inst"
FREE = DERIVED / "free"

FREE_FILES = {
    **{
        f"free_{n}.json": f"results/laneK1/free_{n}.json"
        for n in (
            "baselines",
            "L4_aaaa",
            "L4_pppa",
            "L4x8_aaaa",
            "L6_aaaa_h1.5_3",
            "L6x12_aaaa",
            "L6x12_aaaa_h1.5_3",
            "L8_aaaa",
            "L8_aaaa_h1.5",
            "L8_aaaa_h3",
        )
    },
    "free_flipped_L6_aaaa_h0.0.json": "results/laneK5S1A/s3_flipped/free_flipped_L6_aaaa_h0.0.json",
    "free_flipped_L6_aaaa_h0.5.json": "results/laneK5S1A/s3_flipped/free_flipped_L6_aaaa_h0.5.json",
}

CONFIG_FILES = {  # name: (archive path, key)
    "L4_y2.41_g0.05_pppa": ("results/xi_scan/F8_P1_wedge/L4_y2.41_k-0.01_g0.05_g60.npz", "fields_final"),
    "L4_y2.41_eps_pppa": ("results/xi_scan/F8_P3_eps/L4_y2.41_k-0.01_g0_g60.npz", "fields_final"),
    "L6_y2.028_eps_pppa": ("results/xi_scan/F_L6_k-0.01/L6_y2.028_k-0.01.npz", "sigma_final"),
    "L6_y2.792_eps_pppa": ("results/xi_scan/F_L6_k-0.01/L6_y2.792_k-0.01.npz", "sigma_final"),
}

TAKEOVER_SETS = [  # (archive glob, run label): two independent runs at some (L, y, h)
    ("results/laneS/chains_L4/L4_y*_h*_f0001.npz", "a"),
    ("results/xi_scan/S_prod/L4/L4_y*_h*_f0001.npz", "b"),
    ("results/xi_scan/S_prod/L6/L6_y*_h*_f0001.npz", "b"),
    ("results/laneS/pilot/L6/L6_y*_h*_f0001.npz", "a"),
]
LABELS = "results/laneK2/c051/phase_labels.json"


def need_archive():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the archive root"
    return root


# ---- parts -------------------------------------------------------------------------------------------------
def part_free():
    root = need_archive()
    FREE.mkdir(parents=True, exist_ok=True)
    for name, rel in FREE_FILES.items():
        shutil.copyfile(root / rel, FREE / name)
    print(f"free: {len(FREE_FILES)} files -> {FREE}", flush=True)


def part_configs():
    root = need_archive()
    (CONFIGS / "n1inst").mkdir(parents=True, exist_ok=True)
    for name, (rel, key) in CONFIG_FILES.items():
        src = root / rel
        z = np.load(src, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        info = dict(chain=meta, source=rel, source_md5=md5(src), source_bytes=src.stat().st_size, key=key)
        f = np.asarray(z[key], float)
        np.savez_compressed(CONFIGS / "n1inst" / f"{name}.npz", meta=json.dumps(info, default=str), fields=f)
    print(f"configs: {len(CONFIG_FILES)} files", flush=True)


def part_takeover():
    root = need_archive()
    d = OUT / "takeover"
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern, run in TAKEOVER_SETS:
        for src in sorted(root.glob(pattern)):
            z = np.load(src, allow_pickle=True)
            if "ts_phi_T" not in z.files:
                continue
            L, y, h = X.parse_chain_name(src.name)
            rel = src.relative_to(root).as_posix()
            dst = d / f"L{L}_y{y:g}_h{h:g}_{run}.npz"
            strip_chain(src, dst, keys=["ts_phi_T", "ts_phi_f", "ts_pf_sign"], rel=rel, extra_meta=dict(L=L, y=y, h=h))
            n += 1
    print(f"takeover: {n} chains", flush=True)


def part_winding():
    root = need_archive()
    labels = json.loads((root / LABELS).read_text())["chains"]
    recs = []
    for lab in labels:
        rel = lab["file"]
        z = np.load(root / rel, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        r = X.winding_record(np.asarray(z["sigma_final"]), meta)
        r = dict(file=rel, **r)
        r.update(phase=lab.get("phase"), near_critical=lab.get("near_critical"), scored=lab.get("scored"))
        recs.append(r)
        print(
            f"  {rel}: f_M {r['f_M']:.4f} alpha {r['alpha']:+.3f} N_odd {r['N_odd']:.2f} [{r['wall_s']:.0f}s]",
            flush=True,
        )
    for r in recs:
        r.pop("wall_s", None)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "K6_winding_stored.json"
    p.write_text(X.dumps(dict(records=recs)) + "\n")


BOX_SERIES = [(4, 8, 1.0), (4, 8, 2.0), (6, 12, 1.0), (6, 12, 2.0), (8, 16, 1.0), (8, 16, 2.0)]
BOX_HYPERCUBIC = [(4, 4, 2.0), (6, 6, 2.0), (8, 8, 2.0)]


def part_boxshape():
    out = dict(aspect2=[], hypercubic=[])
    for key, rows in (("aspect2", BOX_SERIES), ("hypercubic", BOX_HYPERCUBIC)):
        for L, Lt, h in rows:
            t0 = time.time()
            v = X.free_phi_L(L, Lt, h)
            out[key].append(dict(L=L, Lt=Lt, h=h, phi_L=v))
            print(f"  boxshape {L}^3 x {Lt} h={h}: phi_L = {v:+.3e} [{time.time() - t0:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "K52_box_shape.json"
    p.write_text(X.dumps(out) + "\n")


def part_massstrings():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "K512_mass_strings.json"
    p.write_text(X.dumps(MS.compute()) + "\n")


def save_series(path, parts):
    out = {}
    for name, r in parts.items():
        for k, v in r.items():
            out[f"{name}__{k}"] = np.asarray(v)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def part_i8():
    parts = {}
    for i, (n, s) in enumerate(X.I8_MET):
        t0 = time.time()
        parts[f"met{i}"] = X.i8_metropolis(n, s, record_trace=True)
        print(f"  i8 metropolis {i}: acc {parts[f'met{i}']['acc']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    for name, (n, s, sab) in zip(("rhmc", "sabotage"), X.I8_RHMC, strict=True):
        t0 = time.time()
        parts[name] = X.i8_rhmc(n, s, sab)
        print(f"  i8 {name}: acc {parts[name]['acc']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    p = OUT / "I8_chains.npz"
    save_series(p, parts)


def part_i10():
    for i, (n, s) in enumerate(X.I10_MET):
        t0 = time.time()
        r = X.i10_metropolis(n, s, record_trace=True)
        p = OUT / f"I10_met{i}.npz"
        save_series(p, {f"met{i}": r})
        print(f"  i10 metropolis {i}: acc {r['acc']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    for name, (n, s, flip) in zip(("rhmc", "flipped"), X.I10_RHMC, strict=True):
        t0 = time.time()
        r = X.i10_rhmc(n, s, flip)
        p = OUT / f"I10_{name}.npz"
        save_series(p, {name: r})
        print(f"  i10 {name}: acc {r['acc']:.2f} [{time.time() - t0:.0f}s]", flush=True)


# ---- manifest ------------------------------------------------------------------------------------------------
DERIVED_USERS = {
    "K6_winding_stored.json": "K6.2",
    "K52_box_shape.json": "K5.2",
    "K512_mass_strings.json": "K5.12",
    "I8_chains.npz": "I.8",
}


def part_manifest():
    """data/manifest/n1inst.tsv: every raw archive file read (md5, size) and every file written by this script."""
    root = need_archive()
    users = {}
    notes = {}

    def add(rel, who, note="chain run (archive)"):
        users.setdefault(rel, set()).update(who.split(","))
        notes[rel] = note

    for rel in FREE_FILES.values():
        add(rel, "K5.1,K5.2", "frozen free baseline (archive)")
    cfg_users = {"L4_y2.41_g0.05_pppa": "I.9", "L4_y2.41_eps_pppa": "I.9,I.10"}
    for name, (rel, _) in CONFIG_FILES.items():
        add(rel, cfg_users.get(name, "K6.1,K6.2"))
    for pattern, _ in TAKEOVER_SETS:
        for src in sorted(root.glob(pattern)):
            if "ts_phi_T" in np.load(src, allow_pickle=True).files:
                add(src.relative_to(root).as_posix(), "K4.1")
    add(LABELS, "K6.2", "phase labels of the stored chains (archive)")
    for lab in json.loads((root / LABELS).read_text())["chains"]:
        add(lab["file"], "K6.2")
    lines = ["\t".join(["kind", "path", "md5", "bytes", "produced_by", "used_by"])]
    for rel in sorted(users):
        p = root / rel
        lines.append("\t".join(["raw", rel, md5(p), str(p.stat().st_size), notes[rel], ",".join(sorted(users[rel]))]))
    rows = []
    for name in FREE_FILES:
        rows.append(("derived", FREE / name, "K5.1,K5.2"))
    for name in CONFIG_FILES:
        rows.append(("config", CONFIGS / "n1inst" / f"{name}.npz", cfg_users.get(name, "K6.1,K6.2")))
    for f in sorted((OUT / "takeover").glob("*.npz")):
        rows.append(("derived", f, "K4.1"))
    for f in sorted(OUT.glob("*")):
        if f.is_file():
            rows.append(("derived", f, DERIVED_USERS.get(f.name, "I.10" if f.name.startswith("I10_") else "")))
    for kind, f, who in rows:
        assert f.exists(), f
        lines.append("\t".join([kind, f.relative_to(ROOT).as_posix(), "-", "-", "scripts/derive_n1inst.py", who]))
    p = ROOT / "data" / "manifest" / "n1inst.tsv"
    p.write_text("\n".join(lines) + "\n")
    print(f"manifest: {len(lines) - 1} rows -> {p}")


PARTS = dict(
    free=part_free,
    configs=part_configs,
    takeover=part_takeover,
    winding=part_winding,
    boxshape=part_boxshape,
    massstrings=part_massstrings,
    i8=part_i8,
    i10=part_i10,
)

if __name__ == "__main__":
    sel = sys.argv[1:] or list(PARTS) + ["manifest"]
    for name in sel:
        if name != "manifest":
            PARTS[name]()
    if "manifest" in sel:
        part_manifest()
