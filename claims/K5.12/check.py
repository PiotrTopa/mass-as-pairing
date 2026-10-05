"""K5.12: every light Lorentz-scalar mass string -- all 16 corner-space strings, the flavour-symmetric 10 and the
flavour sextet -- is an exact zero under the full symmetry group of the h C_chi-sourced action (lattice stabiliser of
C_chi with the one-site shifts; flavour SO(4)); the point-group stabiliser alone leaves two taste-diagonal sextet
strings.

On 4^4 pppp and 4^4 aaaa (h = 0.5, C_chi = (0,3)+):
(1) 16 strings, 10 antisymmetric site kernels (flavour-symmetric 10), 6 symmetric (sextet); all anticommute with K; on
    the periodic box every one lifts the 8 light zero modes of K - h C_chi; 8 taste-diagonal (LL + RR), 8 LR.
(2) group sizes: point group 384, stabiliser of C_chi 32; with the shifts 98 304 / 8192, C_chi-flipping elements 8192.
(3) point-group stabiliser: the light parts of the 10 flavour-symmetric strings vanish; the on-site eps(x) chi Sigma chi
    and the four-link sextet strings average to 1.000 and 0.354 (not zeros).
(4) full stabiliser with shifts: every string's light part (and every sextet string as a whole) averages to zero; the
    on-site string is flipped by exactly the odd-shift half; every sextet string is flipped by as many elements as keep
    it; controls C_chi -> 1, C_asd(0,3) -> 0. The Z4 extension of the point group also kills every light part.
(5) flavour: SO(4) has one invariant in 4 x 4 (delta_ab, symmetric) and none in Lambda^2(4); control: SU(2)_sigma alone
    leaves four (antisymmetric) invariants. The Yukawa triplet is antisymmetric and commutes with the hidden triplet.
(6) sabotage: with the shifts dropped (point-group stabiliser) the full-group zero assertion fails on exactly the two
    taste-diagonal sextet strings.
(7) live recomputation = frozen data/derived/n1inst/K512_mass_strings.json.
About 14 min (the two closures with shifts dominate).
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.analysis import mass_strings as MS
from masspairing.claimcheck import Check
from masspairing.data import derived
from masspairing.flavour import GAMMA, GAMMA2
from masspairing.patterns import scalar_mass_strings

TOL = MS.TOL
c = Check("K5.12")

STR = scalar_mass_strings(4)
c.item(
    "scalar_mass_strings(4): strings, antisymmetric (flavour-symmetric 10), symmetric (flavour sextet)",
    (len(STR), sum(a for _, _, a in STR), sum(not a for _, _, a in STR)),
    len(STR) == 16 and sum(a for _, _, a in STR) == 10,
)
live = MS.compute()
for lab, box in live["boxes"].items():
    st = box["strings"]
    c.record(
        f"{lab}: (S, D) kernel taste |P_L O P_L|/|O| | avg pg, all, pg+Z4 | light part pg, all | flip/keep (all)",
        "",
    )
    for k, v in st.items():
        print(
            f"      {k:24s} {'anti' if v['antisym'] else 'sym ':4s} {v['taste']:6s} {v['PL_frac']:.4f} | "
            f"{v['avg_pg']:.3f} {v['avg_all']:.3f} {v['avg_pg_z4']:.3f} | {v['avgL_pg']:.3f} {v['avgL_all']:.3f} | "
            f"{tuple(v['flip_keep_all'])}",
            flush=True,
        )
    onsite, fourlink = st[str(MS.ONSITE)], st[str(MS.FOURLINK)]
    c.item(
        f"{lab}: point group, stabiliser of C_chi; with one-site shifts, stabiliser, C_chi-flipping elements",
        (box["G_pg"], box["St_pg"], box["G_all"], box["St_all"], box["Anti_all"]),
        box["G_pg"] == 384
        and box["St_pg"] == 32
        and box["G_all"] == 384 * box["V"]
        and box["St_all"] == 32 * box["V"]
        and box["Anti_all"] == box["St_all"],
    )
    c.item(
        f"{lab}: all 16 strings anticommute with K (Lorentz-scalar masses), max |{{O, K}}|",
        max(v["anticomm_K"] for v in st.values()),
        all(v["anticomm_K"] < 1e-12 for v in st.values()),
        "{:.1e}",
    )
    if lab.endswith("pppp"):
        lifts = [v["light_lift"] for v in st.values()]
        c.item(
            f"{lab}: every string lifts the 8 light zero modes of K - h C_chi (lowest singular value: min, max)",
            (round(min(lifts), 3), round(max(lifts), 3)),
            min(lifts) > 0.1,
        )
    tastes = [v["taste"] for v in st.values()]
    c.item(
        f"{lab}: taste-diagonal (LL+RR) / LR strings; on-site and four-link sextet strings taste-diagonal",
        (tastes.count("LL+RR"), tastes.count("LR")),
        tastes.count("LL+RR") == 8
        and tastes.count("LR") == 8
        and onsite["taste"] == "LL+RR"
        and fourlink["taste"] == "LL+RR"
        and all(v["taste"] == "LR" for v in st.values() if v["parity"] == 1),
    )
    sym10 = [v for v in st.values() if v["antisym"]]
    c.item(
        f"{lab}: point-group stabiliser: light parts of the 10 flavour-symmetric strings (asd planes, one-link), max",
        max(v["avgL_pg"] for v in sym10),
        max(v["avgL_pg"] for v in sym10) < TOL,
        "{:.1e}",
    )
    c.item(
        f"{lab}: point-group stabiliser: on-site eps(x) chi Sigma chi (light part) and four-link sextet (light part) "
        "-- not zeros",
        f"{onsite['avg_pg']:.3f} ({onsite['avgL_pg']:.3f}), {fourlink['avg_pg']:.3f} ({fourlink['avgL_pg']:.3f})",
        abs(onsite["avg_pg"] - 1) < 1e-9
        and abs(onsite["avgL_pg"] - 1) < 1e-9
        and abs(fourlink["avg_pg"] - np.sqrt(2) / 4) < 1e-3
        and fourlink["avgL_pg"] > 0.3,
    )
    mx = max(max(v["avgL_all"] for v in st.values()), max(v["avg_all"] for v in st.values() if not v["antisym"]))
    c.item(
        f"{lab}: full stabiliser with shifts: light parts of all 16 strings and the 6 sextet strings as wholes, max",
        mx,
        mx < TOL,
        "{:.1e}",
    )
    c.item(
        f"{lab}: on-site eps string flipped by / kept by (= the odd-shift elements, of the stabiliser)",
        (*onsite["flip_keep_all"], box["St_all_odd_shift"]),
        tuple(onsite["flip_keep_all"]) == (box["St_all"] // 2, box["St_all"] // 2)
        and box["St_all_odd_shift"] == box["St_all"] // 2,
    )
    c.item(
        f"{lab}: every sextet string flipped by as many stabiliser elements as keep it (flip, keep)",
        [tuple(v["flip_keep_all"]) for v in st.values() if not v["antisym"]],
        all(v["flip_keep_all"][0] == v["flip_keep_all"][1] > 0 for v in st.values() if not v["antisym"]),
    )
    c.item(
        f"{lab}: controls C_chi invariant, C_asd(0,3) zero (full stabiliser)",
        (round(box["C_chi"]["avg_all"], 3), box["C_asd(0,3)"]["avg_all"]),
        abs(box["C_chi"]["avg_all"] - 1) < 1e-9 and box["C_asd(0,3)"]["avg_all"] < TOL,
    )
    z4 = max(v["avgL_pg_z4"] for v in st.values())
    c.item(f"{lab}: Z4 extension of the point group (no shifts): light parts, max", z4, z4 < TOL, "{:.1e}")
    # ---- (6) sabotage: drop the shifts
    surv = sorted(k for k, v in st.items() if v["avgL_pg"] > TOL or (not v["antisym"] and v["avg_pg"] > TOL))
    c.item(
        f"{lab}: sabotage -- shifts dropped: strings failing the zero assertion (expected: on-site and four-link)",
        surv,
        surv == sorted([str(MS.ONSITE), str(MS.FOURLINK)]),
    )

# ---- (5) flavour
inv = live["flavour_invariants"]
c.item(
    "flavour SO(4): invariants in 4 x 4 (of them symmetric) -- delta_ab only, none in Lambda^2(4)",
    inv["so(4)"],
    tuple(inv["so(4)"]) == (1, 1),
)
c.item(
    "control: SU(2)_sigma alone leaves invariants in 4 x 4 (symmetric): all four in Lambda^2(4)",
    inv["su(2)_sigma (GAMMA)"],
    tuple(inv["su(2)_sigma (GAMMA)"]) == (4, 0),
)
ok = np.allclose([G @ G2 - G2 @ G for G in GAMMA for G2 in GAMMA2], 0) and all(np.allclose(G.T, -G) for G in GAMMA)
c.item("Yukawa triplet antisymmetric (sextet) and commuting with the hidden triplet: SO(4) = SU(2) x SU(2)", ok, ok)

# ---- (7) live = frozen
frozen = json.loads(derived("n1inst", "K512_mass_strings.json").read_text())
lf, ff = dict(MS.flat(live)), dict(MS.flat(frozen))
dev = max(abs(lf[k] - ff.get(k, np.nan)) for k in lf)
c.item(
    "live recomputation = frozen data/derived/n1inst/K512_mass_strings.json (max deviation)",
    dev,
    set(lf) == set(ff) and dev < 1e-9,
    "{:.1e}",
)
c.done()
