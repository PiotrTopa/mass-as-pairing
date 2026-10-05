#!/usr/bin/env python3
"""Derived data of the K8 direction, large-N and pentest claims (K8.4-K8.9); needs MASSPAIRING_ARCHIVE.

Writes
  data/derived/n1stage/TH/{sym6,y3h3}/<chain>.npz  three 6^4 aaaa N1 chains at kappa = -0.01: y = 2.0 (h = 0, 3) and
                                                   y = 3.0 (h = 3). They ran without stored configurations, so they
                                                   carry no ts_cfg_traj: it is reconstructed exactly from ts_ferm_flag
                                                   (the trajectory index of every fermion measurement). No cfg_digest.
  data/derived/n1stage/TH/sym8/<chain>.npz         two 8^4 aaaa N1 chains at y = 2.0, h = 0 and 3 (K8.9)
  data/derived/n1stage/o4l_series.npz              the light-doublet restriction of the epsilon-vertex density, ts_O4L,
                                                   of the P_c chains (stage 1, 1b, 1c) and of the 6^4 y = 3.0 h = 0, 2
                                                   chains, one array per derived chain
                                                   (key: the derived path with "/" -> "__"), aligned with its rows
  data/derived/largen/mf_prod.json                 the recorded full mean-field transition y*(h) on 4^4 and 6^4 (copy of
                                                   the notebook output; the 6^4 scan takes about 16 min)
  data/manifest/th.tsv                             the manifest fragment
"""

import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from derive_n1stage import KEYS, META_DROP  # noqa: E402

from masspairing.data import DERIVED, archive_root, md5  # noqa: E402

TH = DERIVED / "n1stage" / "TH"
O4L = DERIVED / "n1stage" / "o4l_series.npz"
MFP = DERIVED / "largen" / "mf_prod.json"
TH_CHAINS = (
    "results/laneTH/sym6/L6_y2_k-0.01_g0_g60_chiral_bcaaaa.npz",
    "results/laneTH/sym6/L6_y2_k-0.01_g0_g60_h3_chiral_bcaaaa.npz",
    "results/laneTH/y3h3/L6_y3_k-0.01_g0_g60_h3_chiral_bcaaaa.npz",
)
SYM8_CHAINS = (  # the 8^4 y = 2.0 pair of K8.9 (stored configurations: ts_cfg_traj present, cfg_digest kept)
    "results/laneTHP/sym8/h0/L8_y2_k-0.01_g0_g60_chiral_bcaaaa.npz",
    "results/laneTHP/sym8/h3/L8_y2_k-0.01_g0_g60_h3_chiral_bcaaaa.npz",
)
USERS_SYM8 = "K8.9"
USERS_TH = "K8.4,K8.5,K8.7"
USERS_O4L = "K8.7"
USERS_MFP = "K8.6"
XS = "results/xi_scan"
RAW = []  # (archive path, used_by)


def hname(h):
    return "" if h == 0 else f"_h{h:g}"


def o4l_chains():
    """(derived path, archive path) of the P_c chains whose O4L series the epsilon-vertex claim reads."""
    out = []
    for lat in ("L8", "L6", "L6x12"):
        for h in (0, 0.5, 1, 2):
            name = f"{lat}/{lat}_y2.41_k-0.01_g0_g60{hname(h)}_chiral_bcaaaa.npz"
            out.append((f"n1stage/S1/{name}", f"{XS}/K_N1_S1/{name}"))
        for h in (1.5, 3):
            name = f"{lat}/{lat}_y2.41_k-0.01_g0_g60{hname(h)}_chiral_bcaaaa.npz"
            out.append((f"n1stage/S1b/{name}", f"{XS}/K_N1_S1b/{name}"))
    for h in (0.25, 0.5, 0.75):
        name = f"L6/L6_y2.41_k-0.01_g0_g60_h{h:g}_chiral_bcaaaa.npz"
        out.append((f"n1stage/S1c/{name}", f"{XS}/K_N1_S1c/{name}"))
    for h in (0, 2):  # the 6^4 SMG references at y = 3.0
        name = f"L6/L6_y3_k-0.01_g0_g60{hname(h)}_chiral_bcaaaa.npz"
        out.append((f"n1stage/S1/{name}", f"{XS}/K_N1_S1/{name}"))
    return out


def th_chains(root):
    for rel in TH_CHAINS:
        src = root / rel
        RAW.append((rel, USERS_TH))
        d = np.load(src, allow_pickle=True)
        z = {k: d[k] for k in KEYS + ["ts_O4L"] if k in d.files}
        assert "ts_cfg_traj" not in z
        z["ts_cfg_traj"] = np.nonzero(np.asarray(d["ts_ferm_flag"], bool))[0]
        assert len(z["ts_cfg_traj"]) == len(d["ts_GL_p0"]), "ferm_flag count != fermion measurements"
        meta = json.loads(str(d["meta"]))
        for k in META_DROP:
            meta.pop(k, None)
        meta.update(
            source=rel,
            source_md5=md5(src),
            source_bytes=src.stat().st_size,
            missing_keys=[k for k in KEYS if k not in d.files and k != "ts_cfg_traj"],
            cfg_traj_reconstructed="ts_cfg_traj = nonzero(ts_ferm_flag)",
        )
        dst = TH / pathlib.Path(rel).parent.name / pathlib.Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, meta=json.dumps(meta, default=str), **z)


def sym8_chains(root):
    from derive_n1stage import digest

    for rel in SYM8_CHAINS:
        src = root / rel
        RAW.append((rel, USERS_SYM8))
        d = np.load(src, allow_pickle=True)
        z = {k: d[k] for k in KEYS + ["ts_O4L"] if k in d.files}
        z["cfg_digest"] = np.array([digest(c) for c in d["ts_cfg"]])
        meta = json.loads(str(d["meta"]))
        for k in META_DROP:
            meta.pop(k, None)
        meta.update(
            source=rel,
            source_md5=md5(src),
            source_bytes=src.stat().st_size,
            missing_keys=[k for k in KEYS if k not in d.files],
        )
        dst = TH / "sym8" / pathlib.Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, meta=json.dumps(meta, default=str), **z)


def o4l(root):
    z = {}
    for der, rel in o4l_chains():
        RAW.append((rel, USERS_O4L))
        d = np.load(root / rel, allow_pickle=True)
        dd = np.load(DERIVED / der)
        n = len(dd["ts_GL_p0"])
        x = np.asarray(d["ts_O4L"], float)
        if len(x) > n:  # the archive file was later extended; the derived chain is its pre-extension part
            assert np.array_equal(np.asarray(d["ts_cfg_traj"])[:n], dd["ts_cfg_traj"]), rel
            assert np.array_equal(np.asarray(d["ts_O4"], float)[:n], dd["ts_O4"]), rel
            x = x[:n]
        assert len(x) == n, (rel, len(x), n)
        z[der.replace("/", "__")] = x
    np.savez_compressed(O4L, **z)


def mf_prod(root):
    rel = "results/laneTHP/mf_prod.json"
    RAW.append((rel, USERS_MFP))
    MFP.parent.mkdir(parents=True, exist_ok=True)
    if MFP.exists():
        assert md5(MFP) == md5(root / rel), f"{MFP} differs from the archive file"
    else:
        shutil.copyfile(root / rel, MFP)


def manifest(root):
    lines = ["kind\tpath\tmd5\tbytes\tproduced_by\tused_by"]
    for rel, u in sorted(RAW):
        p = root / rel
        prod = "chain run (archive)" if rel.endswith(".npz") else "notebook output (archive)"
        lines.append(f"raw\t{rel}\t{md5(p)}\t{p.stat().st_size}\t{prod}\t{u}")
    prod = "scripts/derive_th.py"
    for p in sorted(TH.rglob("*.npz")):
        u = USERS_SYM8 if p.parent.name == "sym8" else USERS_TH
        lines.append(f"derived\t{p.relative_to(DERIVED.parents[1]).as_posix()}\t-\t-\t{prod}\t{u}")
    lines.append(f"derived\t{O4L.relative_to(DERIVED.parents[1]).as_posix()}\t-\t-\t{prod}\t{USERS_O4L}")
    lines.append(f"derived\t{MFP.relative_to(DERIVED.parents[1]).as_posix()}\t-\t-\t{prod}\t{USERS_MFP}")
    (DERIVED.parent / "manifest" / "th.tsv").write_text("\n".join(lines) + "\n")


def main():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the archive root"
    missing = [rel for rel in TH_CHAINS + SYM8_CHAINS + ("results/laneTHP/mf_prod.json",) if not (root / rel).exists()]
    if missing:  # archive files marked "pending" in data/MANIFEST.tsv (not yet in the deposit)
        print(f"derive_th: {len(missing)} archive files not present (pending deposit); committed derived data kept")
        return
    th_chains(root)
    sym8_chains(root)
    o4l(root)
    mf_prod(root)
    manifest(root)
    print(f"TH chains, O4L series, mf_prod: {sum(p.stat().st_size for p in TH.rglob('*.npz')) / 1e6:.2f} MB (TH)")


if __name__ == "__main__":
    main()
