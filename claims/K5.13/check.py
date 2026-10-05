"""K5.13: taste-projector coverage and the light pair susceptibility on the complete volume pair (4^4, 8^4) at y = 3.0.

(1) coverage of the light/heavy split on all-antiperiodic boxes: complete on 4^4, 4^3 x 8, 8^4, 12^4; 6^4 split on
    (4/6)^4 = 19.8 % of the momenta, 6^3 x 12 on 29.6 %, 10^4 on 41 %; on L = 6 the surviving momenta per axis are the
    four nearest a corner.
(2) e - e_free on (4^4, 8^4) at h = 1 and 2: negative at 2 sigma (chi_L/free falls with V); h = 0, 0.5 recorded.
(3) the (6^4, 8^4) pair with the same reader at h = 2: positive (the K5.8/K5.11 growth), many sigma from (4,8).
    R_peak at h = 2 on 6^4 and 8^4 with the stage-1c reader (recorded).
(4) chi_L/free at h = 2 on the three complete boxes 4^4 > 4^3 x 8 > 8^4; 6^4 below all three.
(5) sabotage: the pooled 8^4 chi_L rescaled to the SSB-scale growth chi_L(4^4) V_8/V_4 gives e - e_free(4,8) > 0 at
    every h, and the clause of (2) fails on it.
About 10 s.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.analysis import complete_pair as P
from masspairing.analysis import stage1c as C1c
from masspairing.claimcheck import Check

c = Check("K5.13")

# ---- (1) coverage
cv = P.coverage()
for key, r in cv.items():
    c.record(f"{key}: boundary modes / V, split exists on (%)", (r["boundary"], r["V"], round(100 * r["split"], 1)))
complete = ("4x4x4x4 aaaa", "4x4x4x8 aaaa", "8x8x8x8 aaaa", "12x12x12x12 aaaa")
c.item(
    "complete split (no boundary modes) on 4^4, 4^3 x 8, 8^4, 12^4 aaaa (L = 0 mod 4)",
    [cv[k]["boundary"] for k in complete],
    all(cv[k]["boundary"] == 0 for k in complete),
)
s6 = cv["6x6x6x6 aaaa"]
c.item(
    "6^4 aaaa: boundary modes of 1296; split on (4/6)^4 of the momenta (%)",
    (s6["boundary"], round(100 * s6["split"], 1)),
    s6["boundary"] == 1040 and abs(s6["split"] - (4 / 6) ** 4) < 1e-12,
)
s612 = cv["6x6x6x12 aaaa"]
c.item(
    "6^3 x 12 aaaa: boundary modes of 2592; split on (4/6)^3 (%, the L_t = 12 axis is complete)",
    (s612["boundary"], round(100 * s612["split"], 1)),
    s612["boundary"] == 1824 and abs(s612["split"] - (4 / 6) ** 3) < 1e-12,
)
s10 = cv["10x10x10x10 aaaa"]
c.item("10^4 aaaa: split on (8/10)^4 (%)", round(100 * s10["split"], 1), abs(s10["split"] - 0.8**4) < 1e-12)
c6 = np.cos(2 * np.pi * (np.arange(6) + 0.5) / 6)
surv = sorted({float(v) for v in np.round(np.abs(c6[np.abs(c6) > 1e-12]), 6)})
c.item(
    "L = 6 aaaa: |cos p| of the surviving momenta per axis (= cos(pi/6) only: the 4 of 6 nearest a corner)",
    surv,
    surv == [round(np.cos(np.pi / 6), 6)],
)

# ---- (2) the complete pair (4^4, 8^4)
R = P.pair_rows()
for h, r in R.items():
    c.record(
        f"h = {h}: chi_L/free 4^4 (n) | 8^4 pooled (n; members) | 6^4",
        f"{r['L4']['over_free']:.4f} ({r['L4']['n']}) | {r['L8']['over_free']:.4f} ({r['L8']['n']}; "
        + ", ".join(f"{k} {v['over_free']:.4f}" for k, v in r["L8"]["members"].items())
        + ") | "
        + ", ".join(f"{k} {v['over_free']:.4f}" for k, v in r["L6"].items()),
    )
    c.record(
        f"h = {h}: e(4,8), e_free(4,8), e - e_free(4,8); (6,8) same reader e - e_free [e_free(6,8)]",
        f"{r['e48']['e']:+.3f}({r['e48']['err']:.3f}), {r['free']['e48']:.3f}, "
        f"{r['e48']['minus_free']:+.3f}({r['e48']['err']:.3f}); "
        + ", ".join(f"{k} {v['minus_free']:+.3f}({v['err']:.3f})" for k, v in r["e68"].items())
        + f" [{r['free']['e68']:.3f}]",
    )
for h in (1.0, 2.0):
    x = R[h]["e48"]
    c.item(
        f"h = {h:g}: e - e_free on the complete pair (4^4, 8^4) < 0 at 2 sigma (chi_L/free falls with V)",
        f"{x['minus_free']:+.3f}({x['err']:.3f})",
        x["minus_free"] + 2 * x["err"] < 0,
    )
# ---- (3) (6,8) with the same reader
x48, x68 = R[2.0]["e48"], R[2.0]["e68"]["S1b_new"]
sep = (x68["minus_free"] - x48["minus_free"]) / np.hypot(x68["err"], x48["err"])
c.item(
    "h = 2: (6^4 stage 1b, 8^4 pooled) e - e_free with the same reader; distance from the (4,8) value (sigma)",
    f"{x68['minus_free']:+.3f}({x68['err']:.3f}); {sep:.1f}",
    x68["minus_free"] > 0.4 and sep > 5,
)
c.record(
    "h = 2: (6^4 stage 1, 8^4 pooled) e - e_free, same reader",
    f"{R[2.0]['e68']['S1']['minus_free']:+.3f}({R[2.0]['e68']['S1']['err']:.3f})",
)
p2 = C1c.analyse()[2]["A_scores"]["pooled"]["rows"]["2.0"]
(r6, e6), (r8, e8) = p2["R_peak6"], p2["R_peak8"]
c.record(
    "h = 2, stage-1c reader (K5.11): R_peak 6^4 -> 8^4 pooled, rise (sigma); both below 1",
    f"{r6:.2f}({e6:.2f}) -> {r8:.2f}({e8:.2f}), {(r8 - r6) / np.hypot(e6, e8):.1f}; {r6 < 1 and r8 < 1}",
)
# ---- (4) three complete boxes at h = 2
r48, n48 = P.l4x8_h2_over_free()
seq = [R[2.0]["L4"]["over_free"], r48, R[2.0]["L8"]["over_free"]]
low6 = max(v["over_free"] for v in R[2.0]["L6"].values())
c.item(
    "h = 2: chi_L/free on the complete boxes 4^4 (V 256) > 4^3 x 8 (V 512) > 8^4 (V 4096); 6^4 (V 1296) below all",
    f"{seq[0]:.4f} > {seq[1]:.4f} > {seq[2]:.4f}; 6^4 {low6:.4f}",
    seq[0] > seq[1] > seq[2] > low6,
)
# ---- (5) sabotage: injected SSB-scale growth on the 8^4 side
S = P.pair_rows(inject_ssb=True)
sab = {h: S[h]["e48"] for h in (1.0, 2.0)}
c.item(
    "sabotage (8^4 chi_L set to chi_L(4^4) V_8/V_4): e(4,8), e - e_free(4,8) at h = 1, 2; the clause of (2) fails",
    "; ".join(f"{v['e']:.3f}, {v['minus_free']:+.3f}({v['err']:.3f})" for v in sab.values()),
    all(v["minus_free"] - 2 * v["err"] > 0 and not (v["minus_free"] + 2 * v["err"] < 0) for v in sab.values()),
)
c.done()
