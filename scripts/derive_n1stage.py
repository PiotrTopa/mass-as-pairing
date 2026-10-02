#!/usr/bin/env python3
"""Derived data of the N1 stage-0/1/1b claims (K4.2-K4.4, K5.3-K5.8, K6.3, K8.1, K8.2, M.1); needs MASSPAIRING_ARCHIVE.

Writes
  data/derived/n1stage/S0/<lattice>/<chain>.npz    the 26 stage-0 chains (4^4, 4^3x8, 4^4 pppa)
  data/derived/n1stage/S1/<lattice>/<chain>.npz    the 26 stage-1 chains as read by the stage-1 analysis (the four
                                                   chains extended later are taken from their pre-extension copies)
  data/derived/n1stage/S1b/<lattice>/<chain>.npz   the ten stage-1b chains (four extensions to 2000 trajectories, 6 new)
  data/derived/n1stage/S3/                         the phase-flipped-pattern control chain and its free values
  data/derived/n1stage/alpha/{S1,S1b}/<chain>.npz  alpha readout inputs: per-block means of the corner blocks projected
                                                   on the light and heavy corner subspaces; alpha/ref_*.npz the free and
                                                   Dirac-mass references
  data/derived/n1stage/e3_quench_L8_Pc.npz         8^4 P_c h = 0 configurations re-measured at h = 1, 2 (fixed sigma)
  data/derived/n1stage/e9_box_shape.json           box-shape facts of the light Majorana pattern (frozen scan)
  data/configs/n1stage_*.npz                       the stored configurations the quench and alpha re-scan checks use
  data/derived/free/free_*.json                    free baselines (copied when absent; identical to the archive files)
Chain files keep only the time series the analyses read (no configurations) plus `cfg_digest`, the first 16 hex digits
of the md5 of every stored configuration (duplicate and seam checks without the configurations).
"""

import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from masspairing.data import CONFIGS, DERIVED, archive_root, md5, strip_chain  # noqa: E402

OUT = DERIVED / "n1stage"
FREE = DERIVED / "free"
XS = "results/xi_scan"
KEYS = [
    "ts_cfg_traj",
    "ts_ferm_flag",
    "ts_dH",
    "ts_accepted",
    "ts_sigma2",
    "ts_Sigma_stag_abs",
    "ts_S_pi",
    "ts_CL_t",
    "ts_CR_t",
    "ts_chi_L_sum",
    "ts_chi_L_sum_pmin",
    "ts_GL_p0",
    "ts_GR_p0",
    "ts_chi_f_sum",
    "ts_phi_f",
    "ts_phi_L",
    "ts_phi_T_R",
    "ts_phi_T_L",
    "ts_phi_stag_sq",
    "ts_O4",
]
EXTENDED = (
    "L8/L8_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz",
    "L8/L8_y3_k-0.01_g0_g60_h1_chiral_bcaaaa.npz",
    "L6/L6_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz",
    "L6/L6_y3_k-0.01_g0_g60_h1_chiral_bcaaaa.npz",
)
FREE_FILES = (
    "free_L4_aaaa.json",
    "free_L4x8_aaaa.json",
    "free_L4_pppa.json",
    "free_L8_aaaa.json",
    "free_L8_aaaa_h1.5.json",
    "free_L8_aaaa_h3.json",
    "free_L6x12_aaaa.json",
    "free_L6x12_aaaa_h1.5_3.json",
    "free_L6_aaaa_h1.5_3.json",
    "free_baselines.json",
)
META_DROP = ("out", "wall_s", "hmc_stats", "device", "graphs", "lanczos_cf", "fast", "resume")
RAW = []  # (archive path, used_by) of every raw file read


def digest(a):
    return hashlib.md5(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()[:16]


def chain(root, rel, dst, used_by):
    src = root / rel
    RAW.append((rel, used_by))
    strip_chain(src, dst, keys=KEYS, rel=rel)
    z = dict(np.load(dst, allow_pickle=True))
    meta = json.loads(str(z.pop("meta")))
    for k in META_DROP:
        meta.pop(k, None)
    d = np.load(src, allow_pickle=True)
    z["cfg_digest"] = np.array([digest(c) for c in d["ts_cfg"]])
    np.savez_compressed(dst, meta=json.dumps(meta, default=str), **z)


def chains(root):
    for sub in ("L4", "L4x8", "L4pppa"):
        for f in sorted((root / XS / "K_N1_S0" / sub).glob("*.npz")):
            chain(root, f"{XS}/K_N1_S0/{sub}/{f.name}", OUT / "S0" / sub / f.name, "K4.2,K5.3,K5.5")
    for sub in ("L8", "L6x12", "L6", "L6pppa"):
        for f in sorted((root / XS / "K_N1_S1" / sub).glob("*.npz")):
            rel = f"{sub}/{f.name}"
            src = f"{XS}/K_N1_S1b/pre/{rel}" if rel in EXTENDED else f"{XS}/K_N1_S1/{rel}"
            chain(root, src, OUT / "S1" / sub / f.name, "K4.3,K4.4,K5.4,K5.5,K5.7,K5.8,K8.1,K8.2,M.1")
    for rel in EXTENDED:
        chain(root, f"{XS}/K_N1_S1/{rel}", OUT / "S1b" / rel, "K5.7,K5.8,K8.2")
    for sub in ("L8", "L6x12", "L6"):
        for f in sorted((root / XS / "K_N1_S1b" / sub).glob("*.npz")):
            chain(root, f"{XS}/K_N1_S1b/{sub}/{f.name}", OUT / "S1b" / sub / f.name, "K5.7,K8.2,M.1")
    s3 = "results/laneK5S1A/s3_flipped"
    chain(root, f"{s3}/L6_y3_k-0.01_g0_g60_h0.5_chiral_bcaaaa.npz", OUT / "S3" / "L6_y3_h0.5_flipped.npz", "K5.6")
    for h in ("0.0", "0.5"):
        name = f"free_flipped_L6_aaaa_h{h}.json"
        RAW.append((f"{s3}/{name}", "K5.6"))
        shutil.copyfile(root / s3 / name, OUT / "S3" / name)


def isometry(P):
    """Orthonormal basis (columns) of the range of an orthogonal projector P (eigenvalue 1)."""
    w, U = np.linalg.eigh((P + P.conj().T) / 2)
    return U[:, w > 0.5]


def alpha(root):
    """Per chain: block means of the four corner blocks G_B(+-p1), G_B(+-p3) compressed to the light and heavy corner
    subspaces U^dag G U (U an orthonormal basis of the range of P_L,B (x) 1_2 or P_R,B (x) 1_2; Frobenius norms equal
    those of P G P) and the per-configuration full-block norms. The free (sigma = 0) and Dirac-mass reference blocks
    depend only on (L, Lt, bc, h): one file per lattice and h, alpha/ref_<L>x<Lt>_<bc>_h<h>.npz."""
    srcs = [("S1", f) for f in sorted((root / "results/laneK5S1A/alpha").glob("*.npz"))]
    srcs += [("S1b", f) for f in sorted((root / "results/laneK5S1bA/alpha").glob("*__new.npz"))]
    refs = {}
    for tag, f in srcs:
        rel = f.relative_to(root).as_posix()
        RAW.append((rel, "K6.3,K5.8"))
        z = np.load(f, allow_pickle=True)
        U = {side: isometry(np.kron(z[f"P{side}B"], np.eye(2))) for side in ("L", "R")}
        comp = lambda G, side: np.einsum("ia,...ij,jb->...ab", U[side].conj(), G, U[side])
        out = {f"block_means_{s}": comp(z["block_means"], s) for s in ("L", "R")}
        for k in ("counts", "n", "sel_traj", "PLB", "PRB", "p0s", "base", "shape", "bc", "y", "h", "L", "Lt"):
            out[k] = z[k]
        for k in ("hook_identity", "sigma2_traj", "meas_every", "all_norms"):
            out[k] = z[k]
        bc = "".join("a" if b == -1 else "p" for b in z["bc"])
        key = f"ref_{int(z['L'])}x{int(z['Lt'])}_{bc}_h{float(z['h']):g}"
        ref = dict(masses=z["masses"], PLB=z["PLB"], PRB=z["PRB"])
        for s in ("L", "R"):
            ref[f"free_{s}"] = comp(z["free"], s)
            ref[f"massed_{s}"] = comp(z["massed"], s)
        if key in refs:  # the references of equal (L, Lt, bc, h) are identical in the archive
            assert all(np.array_equal(refs[key][k], ref[k]) for k in ref), key
        refs[key] = ref
        out["ref"] = key
        meta = dict(source=rel, source_md5=md5(f), source_bytes=f.stat().st_size)
        dst = OUT / "alpha" / tag / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, meta=json.dumps(meta), **out)
    for key, ref in refs.items():
        np.savez_compressed(OUT / "alpha" / f"{key}.npz", **ref)


def e3(root):
    rel = "results/laneK4Z/e3_quench_L8_Pc.npz"
    RAW.append((rel, "K8.2"))
    z = np.load(root / rel, allow_pickle=True)
    rec = json.loads(str(z["rows_json"]))
    keys = list(rec)  # the archive's order
    out = dict(
        traj=np.array([int(k.split("|")[0]) for k in keys]),
        h=np.array([float(k.split("|")[1]) for k in keys]),
        CL_t=np.array([rec[k]["CL_t"] for k in keys], float),
        GL_p0=np.array([rec[k]["GL_p0"] for k in keys], float),
        identity_CL_t=np.array([rec[k].get("_identity_CL_t", np.nan) for k in keys], float),
    )
    meta = dict(source=rel, source_md5=md5(root / rel), chain=str(z["chain"]).replace(f"{root}/", ""))
    np.savez_compressed(OUT / "e3_quench_L8_Pc.npz", meta=json.dumps(meta), **out)


def e9(root):
    rel = "results/laneK5S1bA/e9_box_shape.json"
    RAW.append((rel, "K5.7"))
    shutil.copyfile(root / rel, OUT / "e9_box_shape.json")


def configs(root):
    """Stored configurations: the stage-0 quench (10 per chain), the 6^4 P_c h = 0 quench (4), the alpha re-scan (2)."""
    sets = {
        "n1stage_quench_L4_y2_h2": (
            f"{XS}/K_N1_S0/L4/L4_y2_k-0.01_g0_g60_h2_chiral_bcaaaa.npz",
            list(range(100, 300, 20)),
        ),
        "n1stage_quench_L4_y3_h2": (
            f"{XS}/K_N1_S0/L4/L4_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz",
            list(range(100, 300, 20)),
        ),
        "n1stage_quench_L4x8_y3_h2": (
            f"{XS}/K_N1_S0/L4x8/L4x8_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz",
            list(range(100, 300, 20)),
        ),
        "n1stage_quench_L6_y2.41_h0": (f"{XS}/K_N1_S1/L6/L6_y2.41_k-0.01_g0_g60_chiral_bcaaaa.npz", None),
        "n1stage_alpha_L6_y3_h2": (f"{XS}/K_N1_S1b/pre/L6/L6_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz", "alpha"),
    }
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for name, (rel, idx) in sets.items():
        RAW.append((rel, "K4.2" if "L4" in name else ("K8.1" if "quench" in name else "K6.3")))
        d = np.load(root / rel, allow_pickle=True)
        ct = np.asarray(d["ts_cfg_traj"])
        if idx is None:  # every 56th stored configuration after the cut, the first four
            idx = [i for i in range(len(ct)) if ct[i] >= 100][::56][:4]
        elif idx == "alpha":
            a = np.load(root / "results/laneK5S1A/alpha/L6__L6_y3_k-0.01_g0_g60_h2_chiral_bcaaaa.npz")
            idx = [int(np.where(ct == a["sel_traj"][j])[0][0]) for j in (0, 5)]
        meta = json.loads(str(d["meta"]))
        for k in META_DROP:
            meta.pop(k, None)
        out = dict(
            fields=np.asarray(d["ts_cfg"][idx], float),
            index=np.array(idx),
            cfg_traj=ct[idx],
            CL_t=d["ts_CL_t"][idx],
            CR_t=d["ts_CR_t"][idx],
            phi_f=d["ts_phi_f"][idx],
        )
        info = dict(chain=meta, source=rel, source_md5=md5(root / rel), source_bytes=(root / rel).stat().st_size)
        np.savez_compressed(CONFIGS / f"{name}.npz", meta=json.dumps(info, default=str), **out)


def free(root):
    FREE.mkdir(parents=True, exist_ok=True)
    for name in FREE_FILES:
        rel = f"results/laneK1/{name}"
        RAW.append((rel, "K4.2,K4.3,K4.4,K5.3,K5.4,K5.5,K5.7,K5.8,K8.1,K8.2"))
        dst = FREE / name
        if dst.exists():
            assert md5(dst) == md5(root / rel), f"{dst} differs from the archive file"
        else:
            shutil.copyfile(root / rel, dst)


def manifest(root):
    lines = ["kind\tpath\tmd5\tbytes\tproduced_by\tused_by"]
    users = {}
    for rel, u in RAW:
        users.setdefault(rel, set()).update(u.split(","))
    for rel in sorted(users):
        p = root / rel
        chain = rel.endswith(".npz") and ("/xi_scan/" in rel or "/s3_flipped/" in rel)
        prod = "chain run (archive)" if chain else "notebook output (archive)"
        lines.append(f"raw\t{rel}\t{md5(p)}\t{p.stat().st_size}\t{prod}\t{','.join(sorted(users[rel]))}")
    prod = "scripts/derive_n1stage.py"
    use = "K4.2,K4.3,K4.4,K5.3,K5.4,K5.5,K5.6,K5.7,K5.8,K6.3,K8.1,K8.2,M.1"
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            lines.append(f"derived\t{p.relative_to(OUT.parents[2]).as_posix()}\t-\t-\t{prod}\t{use}")
    for name in FREE_FILES:
        lines.append(f"derived\tdata/derived/free/{name}\t-\t-\t{prod}\t{use}")
    for p in sorted(CONFIGS.glob("n1stage_*.npz")):
        lines.append(f"config\tdata/configs/{p.name}\t-\t-\t{prod}\t{use}")
    (OUT.parents[1] / "manifest" / "n1stage.tsv").write_text("\n".join(lines) + "\n")


def main():
    root = archive_root()
    assert root is not None, "set MASSPAIRING_ARCHIVE to the archive root"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S3").mkdir(parents=True, exist_ok=True)
    chains(root)
    alpha(root)
    e3(root)
    e9(root)
    configs(root)
    free(root)
    manifest(root)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"data/derived/n1stage: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
