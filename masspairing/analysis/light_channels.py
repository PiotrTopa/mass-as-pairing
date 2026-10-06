"""Every inequivalent light mass-type channel at h = 2 on 4^4, 6^4, 8^4, at y = 3.0 and at P_c (K5.16).

The pair susceptibility chi = V <phi^2> of the light part (P_L O P_L for taste-diagonal strings, P_L O P_R + h.c. for
LR ones) of each of the sixteen corner mass strings O, plus the three self-dual planes as controls, per stored
configuration (data/derived/k5channels/channels_h2.npz, 22 channels; produced in the research notebook from the
stored sigma configurations, the 8^4 rows on one rented GPU gated against the CPU code). Channels:

  asd0j        the anti-self-dual (0,j) plane (chiral_mass(lat, (0, j), -1)); asd03 is the measured chi_L
  sd0j         the self-dual (0,j) planes (sd03 = the direction of C_chi itself): controls, zero light part on 4^4, 6^4
  link0..3     the four one-link LR strings (Dirac light-heavy), one orbit
  onsite_31/13, fourlink_31/13   the on-site and four-link sextet strings, (3,1) and (1,3) flavour parts
  three{a..d}_31/13              the four three-link LR sextet strings, one orbit, both flavour parts

Statistics as in ``complete_pair`` (K5.13): 10 blocks per member, 20 for a pooled multi-member series, raised to
sigma_naive sqrt(2 tau_int) where larger; chi/free against the free value of the same channel and box;
e = d ln chi / d ln V on (4^4, 8^4) and (6^4, 8^4) minus the free exponent of the same channel. Orbit rows average the
member values and combine their errors in quadrature divided by the number of members (as recorded with the
measurement). ``orbits`` recomputes the orbits (modulo sign) of the sixteen strings under the point-group stabiliser
and the full stabiliser with shifts on 4^4 aaaa.
"""

from __future__ import annotations

import json

import numpy as np

from .. import symmetry as S
from ..lattice import Lattice, kinetic_matrix
from ..patterns import chiral_mass, scalar_mass_strings
from . import corner_invariants as CI
from .complete_pair import blocked, chain_path
from .mass_strings import string_op
from .n1stage import DERIVED, N1DIR, open_chain, tau_int

DIR = DERIVED / "k5channels"
V = {4: 256, 6: 1296, 8: 4096}
CH = [
    "asd03",
    "asd01",
    "asd02",
    "link0",
    "link1",
    "link2",
    "link3",
    "onsite_31",
    "onsite_13",
    "fourlink_31",
    "fourlink_13",
    "threea_31",
    "threeb_31",
    "threec_31",
    "threed_31",
    "threea_13",
    "threeb_13",
    "threec_13",
    "threed_13",
    "sd01",
    "sd02",
    "sd03",
]
SETS = {
    "y3": {4: ["L4_y3"], 6: ["L6_y3"], 8: ["L8_y3_rA", "L8_y3_rB", "L8_y3_S1"]},
    "Pc": {4: ["L4_Pc"], 6: ["L6_Pc"], 8: ["L8_Pc"]},
}
ORBIT_ROWS = [
    ("asd(0,3)", ["asd03"]),
    ("asd(0,1)", ["asd01"]),
    ("asd(0,2)", ["asd02"]),
    ("one-link LR", ["link0", "link1", "link2", "link3"]),
    ("on-site sextet (3,1)", ["onsite_31"]),
    ("four-link sextet (3,1)", ["fourlink_31"]),
    ("three-link LR sextet (3,1)", ["threea_31", "threeb_31", "threec_31", "threed_31"]),
    ("three-link LR sextet (1,3)", ["threea_13", "threeb_13", "threec_13", "threed_13"]),
    ("four-link sextet (1,3)", ["fourlink_13"]),
    ("on-site sextet (1,3)", ["onsite_13"]),
]


def load():
    z = np.load(DIR / "channels_h2.npz", allow_pickle=False)
    cols = list(z["columns"])
    data = {t: {c: z[t][:, i] for i, c in enumerate(cols)} for t in z.files if t not in ("columns", "meta")}
    fg = json.loads((DIR / "free_and_gates_h2.json").read_text())
    return data, fg


def stat(x, nb):
    x = np.asarray(x, float)
    if len(x) < 2 * nb:
        nb = max(2, len(x) // 2)
    m, s = blocked(x, nb)
    tau = tau_int(x) if len(x) > 4 else 0.5
    return float(m), max(float(s), float(x.std(ddof=1) / np.sqrt(len(x)) * np.sqrt(2 * max(tau, 0.5)))), float(tau)


def _exponent(c1, e1, c2, e2, V1, V2):
    return float(np.log(c2 / c1) / np.log(V2 / V1)), float(np.sqrt((e1 / c1) ** 2 + (e2 / c2) ** 2) / np.log(V2 / V1))


def analyse():
    """{point: {L: dict(n, members, stored_dev, chan={ch: dict(chi, err, free, ratio, ratio_err, e48/e68)})}}."""
    data, fg = load()
    free = {int(k[1:]): v for k, v in fg["free"].items()}
    rep = {}
    for point, boxes in SETS.items():
        per = {}
        for L, tags in boxes.items():
            rows = {c: np.concatenate([data[t][c] for t in tags]) for c in ["traj", "stored"] + CH}
            n = len(rows["traj"])
            per[L] = dict(
                n=n,
                members={t: len(data[t]["traj"]) for t in tags},
                stored_dev=float(np.max(np.abs(rows["asd03"] / rows["stored"] - 1))),
                chan={},
            )
            for ch in CH:
                m, s, tau = stat(rows[ch], 20 if len(tags) > 1 else 10)
                fr = free[L][ch]
                per[L]["chan"][ch] = dict(
                    chi=m, err=s, tau=tau, free=fr, ratio=m / fr if fr else None, ratio_err=s / fr if fr else None
                )
        for La in (4, 6):
            for ch in CH:
                a, b = per[La]["chan"][ch], per[8]["chan"][ch]
                if a["free"] and b["free"] and a["chi"] > 0 and b["chi"] > 0:
                    e, err = _exponent(a["chi"], a["err"], b["chi"], b["err"], V[La], V[8])
                    ef = float(np.log(b["free"] / a["free"]) / np.log(V[8] / V[La]))
                    b[f"e{La}8"] = (e - ef, err)
        rep[point] = per
    return rep


def compact(rep=None):
    """{point: [(label, {L: (ratio, err)}, e48 (val, err) or None, e68 or None)]} with orbit means."""
    rep = rep or analyse()
    out = {}
    for point, per in rep.items():
        rows = []
        for label, members in ORBIT_ROWS:
            k = len(members)
            cells = {}
            for L in (4, 6, 8):
                r = [per[L]["chan"][c]["ratio"] for c in members]
                e = [per[L]["chan"][c]["ratio_err"] for c in members]
                cells[L] = (float(np.mean(r)), float(np.sqrt(np.sum(np.square(e))) / k))
            ee = {}
            for La in (4, 6):
                vals = [per[8]["chan"][c].get(f"e{La}8") for c in members]
                ee[La] = (
                    None
                    if any(v is None for v in vals)
                    else (float(np.mean([v[0] for v in vals])), float(np.sqrt(sum(v[1] ** 2 for v in vals)) / k))
                )
            rows.append((label, cells, ee[4], ee[6]))
        out[point] = rows
    return out


def chain_match():
    """Rows of the derived chains with the same trajectory: (matched, max rel. deviation of the stored chi_L)."""
    data, _ = load()
    paths = {
        "L4_y3": [chain_path("S0", "L4", 4, "_h2")],
        "L6_y3": [chain_path("S1b", "L6", 6, "_h2")],
        "L8_y3_rA": [chain_path("S1c", "L8_h2_rA", 8, "_h2")],
        "L8_y3_rB": [chain_path("S1c", "L8_h2_rB", 8, "_h2")],
        "L8_y3_S1": [chain_path("S1", "L8", 8, "_h2"), chain_path("S1b", "L8", 8, "_h2")],
        "L4_Pc": [N1DIR / "S0" / "L4" / "L4_y2.41_k-0.01_g0_g60_h2_chiral_bcaaaa.npz"],
        "L6_Pc": [N1DIR / "S1" / "L6" / "L6_y2.41_k-0.01_g0_g60_h2_chiral_bcaaaa.npz"],
        "L8_Pc": [N1DIR / "S1" / "L8" / "L8_y2.41_k-0.01_g0_g60_h2_chiral_bcaaaa.npz"],
    }
    res = {}
    for tag, ps in paths.items():
        ref = {}
        for p in ps:
            if p.exists():
                d, _ = open_chain(p)
                ref.update(
                    zip(
                        np.asarray(d["ts_cfg_traj"]).tolist(),
                        np.asarray(d["ts_chi_L_sum"], float).tolist(),
                        strict=True,
                    )
                )
        t, st = data[tag]["traj"], data[tag]["stored"]
        hits = [(ref[int(a)], b) for a, b in zip(t, st, strict=True) if int(a) in ref]
        dev = max((abs(r - b) / abs(r) for r, b in hits), default=float("inf"))
        res[tag] = (len(hits), len(t), dev)
    return res


def gates():
    """Max relative CPU/GPU differences over the 22 channels: free 8^4 at h = 2, one stored 8^4 configuration."""
    _, fg = load()
    g = fg["gate"]

    def rel(a, b):
        return max(abs(a[k] - b[k]) / abs(a[k]) for k in a if a[k] != 0)

    return dict(
        free=rel(g["free_L8_cpu"], g["free_L8_gpu"]),
        cfg=rel(g["cfg_S1_traj400_cpu"], g["cfg_S1_traj400_gpu"]),
        cfg_stored=abs(g["cfg_S1_traj400_cpu"]["asd03"] / g["cfg_S1_traj400_stored_chi_L"] - 1),
    )


def orbits():
    """Orbits (modulo sign) of the sixteen mass strings under the point-group stabiliser of C_chi (32) and the full
    stabiliser with shifts (8192) on 4^4 aaaa: {group: sorted list of orbit classes (tuples of (S, D))}."""
    lat = Lattice((4,) * 4, bc=(-1,) * 4)
    K = kinetic_matrix(lat)
    grp = CI.groups(lat, K, chiral_mass(lat))
    strings = [(Sx, Dx) for Sx, Dx, _ in scalar_mass_strings(4)]
    ops = {k: string_op(lat, *k, anti) for k, (*_, anti) in zip(strings, scalar_mass_strings(4), strict=True)}

    def key(A):
        B = np.round(A, 8) + 0.0
        return min(B.tobytes(), (-B + 0.0).tobytes())

    index = {key(O): k for k, O in ops.items()}
    out = {}
    for name, gens in (("pg", grp["St_pg"]), ("all", grp["st_gens"])):
        classes = set()
        for O in ops.values():
            seen = {key(O): O}
            queue = [O]
            while queue:
                A = queue.pop()
                for g, s in gens:
                    B = S.transform(A, g, s)
                    kb = key(B)
                    if kb not in seen:
                        seen[kb] = B
                        queue.append(B)
            classes.add(tuple(sorted(index[x] for x in seen if x in index)))
        out[name] = sorted(classes)
    out["sizes"] = (len(grp["St_pg"]), len(grp["St_all"]))
    return out
