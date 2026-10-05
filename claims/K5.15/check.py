"""K5.15: light block of the source per box, restricted-mode control at (3.0, h = 2), (4^4, 8^4) pair at P_c.

(1) ||P_L C_chi P_L||_F / ||C_chi||_F on aaaa boxes, momentum-resolved and matrix-free: 0 (4^4), 0 (6^4), 0.3684 (8^4),
    0.1951 (4^3 x 8), 0.1730 (6^3 x 12); on every momentum = 2^{-3/2} ||cos p0 cos p3| - |cos p1 cos p2||; 1536 of the
    4096 light momenta of 8^4 source-free, every split momentum of 4^4 and 6^4 source-free, none of 4^3 x 8, 6^3 x 12.
(2) estimator gate: the reduced-basis chi_L with the full light basis (and on 6^4 with the IR-class restriction, which
    there is the whole split) = the stored chi_L of stored 4^4 / 6^4 configurations; the 90 control rows carry the
    stored chi_L of the derived stage-1c chains at the same trajectories.
(3) the control: chi'/free and e - e_free on (4,8), (6,8) with all light modes, the source-free modes and the IR class;
    with the source-free modes the (4,8) fall is steeper than with all modes and negative at 2 sigma.
(4) P_c: e - e_free on (4^4, 8^4) at h = 0, 0.5, 1, 2 and g(8)/g(4); |e - e_free| < 0.15 on both pairs.
About 25 s.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.analysis import restricted_modes as R
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice

ROOT = pathlib.Path(__file__).resolve().parents[2]
c = Check("K5.15")


def rnd(x, d):
    return float(f"{x:.{d}f}")


# ---- (1) light block
boxes = {(4, 4, 4, 4): 0.0, (6, 6, 6, 6): 0.0, (8, 8, 8, 8): 0.3684, (4, 4, 4, 8): 0.1951, (6, 6, 6, 12): 0.1730}
LB, FM = {}, {}
for shape in boxes:
    LB[shape], FM[shape] = R.light_block(shape), R.light_block_formula(shape)
    r = LB[shape]
    c.record(
        f"{'x'.join(map(str, shape))} aaaa: fraction (closed form); split / source-free / dressed; "
        "max |live - formula|",
        f"{r['frac']:.4f} ({FM[shape]['frac']:.4f}); {r['n_split']} / {r['n_source_free']} / {r['n_dressed']}; "
        f"{r['max_formula_dev']:.1e}",
    )
c.item(
    "||P_L C_chi P_L|| / ||C_chi|| on 4^4, 6^4, 8^4, 4^3 x 8, 6^3 x 12 aaaa "
    "(live = closed form, per momentum and in total)",
    [rnd(LB[s]["frac"], 4) for s in boxes],
    all(
        rnd(LB[s]["frac"], 4) == q
        and abs(LB[s]["frac"] - FM[s]["frac"]) < 1e-12
        and LB[s]["max_formula_dev"] < 1e-12
        and (LB[s]["n_split"], LB[s]["n_source_free"]) == (FM[s]["n_split"], FM[s]["n_source_free"])
        for s, q in boxes.items()
    ),
)
sf = {s: (LB[s]["n_source_free"], LB[s]["n_split"]) for s in boxes}
c.item(
    "source-free light momenta / momenta with a split: 4^4, 6^4, 8^4, 4^3 x 8, 6^3 x 12",
    list(sf.values()),
    list(sf.values()) == [(256, 256), (256, 256), (1536, 4096), (0, 512), (0, 768)],
)
r8 = LB[(8, 8, 8, 8)]
key = np.round(np.sort(r8["abs_cos"], axis=1), 3)
two = np.all(key == [0.383, 0.383, 0.924, 0.924], axis=1)
eq = np.all(key == key[:, :1], axis=1)
c.item(
    "8^4 source-free = the momenta with all |cos p_mu| equal (256 + 256) and the two-and-two ones with "
    "|cos p0 cos p3| = |cos p1 cos p2| (of 1536); IR class size",
    (
        int((eq & (r8["mv"] < 1e-9)).sum()),
        int((two & (r8["mv"] < 1e-9)).sum()),
        int(two.sum()),
        int(R.ir_class_mask((8,) * 4).sum()),
    ),
    int((eq & (r8["mv"] < 1e-9)).sum()) == 512
    and int((two & (r8["mv"] < 1e-9)).sum()) == 1024
    and int(two.sum()) == 1536
    and int(R.ir_class_mask((8,) * 4).sum()) == 256,
)

# ---- (2) gates
gate = []
for name, L, rows, restrict in (
    ("n1_L4_y3_h2_aaaa", 4, (0, 1), False),
    ("n1_L6_y3_h2_aaaa", 6, (0,), False),
    ("n1_L6_y3_h2_aaaa", 6, (0,), True),
):
    z = np.load(ROOT / "data/configs" / f"{name}.npz", allow_pickle=False)
    m = json.loads(str(z["meta"]))["chain"]
    lat = Lattice((L,) * 4, bc=(-1,) * 4)
    W, cc = R.light_basis(lat, R.ir_class_mask(lat.shape) if restrict else None)
    for i in rows:
        chi, _ = R.chi_reduced(lat, m["y"], z[f"fields_{i}"], m["h10"], W, cc)
        gate.append(abs(chi / float(z[f"row{i}_chi_L_sum"]) - 1))
c.item(
    "estimator gate: reduced-basis chi_L = stored chi_L_sum (4^4 x 2, 6^4, 6^4 with the IR-class basis); max rel. dev.",
    max(gate),
    max(gate) < 1e-12,
    fmt="{:.1e}",
)
dev, nmatch = R.control_vs_chains()
c.item(
    "control rows: stored full chi_L = the derived stage-1c chains at the same trajectory "
    "(rows matched of 90, max rel. dev.)",
    (nmatch, dev),
    nmatch == 90 and dev < 1e-12,
)

# ---- (3) the control
K = R.control()
c.record(
    "free 8^4 h = 2: chi_L all / IR class / source-free; 4^4, 6^4 chi_L/free (n)",
    f"{K['free']['full']:.5e} / {K['free']['ir']:.5e} / {K['free']['sf']:.5e}; "
    f"{K['L4']['over_free']:.4f} ({K['L4']['n']}), "
    f"{K['L6']['over_free']:.4f} ({K['L6']['n']})",
)
quote = {
    "full": (0.0156, 0.0021, -0.306, 0.058, 0.540, 0.134),
    "sf": (0.0056, 0.0006, -0.676, 0.049, -0.352, 0.110),
    "ir": (0.0024, 0.0003, -0.976, 0.050, -1.076, 0.113),
}
for lbl, q in quote.items():
    x = K["rows"][lbl]
    got = (
        rnd(x["pooled"]["over_free"], 4),
        rnd(x["pooled"]["over_free_err"], 4),
        rnd(x["e48"]["minus_free"], 3),
        rnd(x["e48"]["err"], 3),
        rnd(x["e68"]["minus_free"], 3),
        rnd(x["e68"]["err"], 3),
    )
    c.record(
        f"{lbl}: chi'/free rA, rB",
        f"{x['rA']['over_free']:.4f}({x['rA']['over_free_err']:.4f}), "
        f"{x['rB']['over_free']:.4f}({x['rB']['over_free_err']:.4f})",
    )
    c.item(
        f"{lbl}: chi'/free pooled (err); e - e_free (4,8) (err); e - e_free (6,8) (err)",
        f"{x['pooled']['over_free']:.4f}({x['pooled']['over_free_err']:.4f}); "
        f"{x['e48']['minus_free']:+.3f}({x['e48']['err']:.3f}); "
        f"{x['e68']['minus_free']:+.3f}({x['e68']['err']:.3f})",
        got == q,
    )
for lbl in ("ir", "sf"):
    s = K["rows"][lbl]["share"]
    c.record(f"per configuration {lbl}/all: mean, median; free", f"{s['mean']:.3f}, {s['median']:.3f}; {s['free']:.3f}")
sf48, all48, ir68 = K["rows"]["sf"]["e48"], K["rows"]["full"]["e48"], K["rows"]["ir"]["e68"]
c.item(
    "source-free modes: (4,8) e - e_free + 2 sigma < 0 and below the all-modes value; IR class: (6,8) + 2 sigma < 0",
    f"{sf48['minus_free']:+.3f}({sf48['err']:.3f}) vs {all48['minus_free']:+.3f}; "
    f"{ir68['minus_free']:+.3f}({ir68['err']:.3f})",
    sf48["minus_free"] + 2 * sf48["err"] < 0
    and sf48["minus_free"] < all48["minus_free"]
    and ir68["minus_free"] + 2 * ir68["err"] < 0,
)

# ---- (4) P_c
P = R.pc_pair()
qpc = {
    0.0: (-0.114, 0.014, 0.878),
    0.5: (-0.080, 0.008, 0.920),
    1.0: (-0.024, 0.004, 0.926),
    2.0: (0.025, 0.004, 0.933),
}
got = {h: (rnd(r["e48"]["minus_free"], 3), rnd(r["e48"]["err"], 3), rnd(r["e48"]["g_ratio"], 3)) for h, r in P.items()}
c.item(
    "P_c (2.41): e - e_free (4,8) (err), g(8)/g(4) at h = 0, 0.5, 1, 2",
    got,
    got == qpc,
)
c.record(
    "P_c (2.41): e - e_free (6,8) (err), g(8)/g(6) at h = 0, 0.5, 1, 2",
    {h: (rnd(r["e68"]["minus_free"], 3), rnd(r["e68"]["err"], 3), rnd(r["e68"]["g_ratio"], 3)) for h, r in P.items()},
)
c.item(
    "P_c: no SSB-scale growth, |e - e_free| < 0.15 on (4,8) and (6,8) at every h",
    max(abs(r[k]["minus_free"]) for r in P.values() for k in ("e48", "e68")),
    all(abs(r[k]["minus_free"]) < 0.15 for r in P.values() for k in ("e48", "e68")),
    fmt="{:.3f}",
)
c.done()
