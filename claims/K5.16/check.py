"""K5.16: every inequivalent light mass-type channel at h = 2, y = 3.0 and P_c, on 4^4, 6^4, 8^4.

(1) orbits (modulo sign) of the sixteen mass strings on 4^4 aaaa: the same six classes under the 32-element point-group
    stabiliser and under the 8192-element stabiliser with shifts.
(2) sample: the estimator's (0,3) channel = the stored chi_L on every configuration; the stored chi_L = the derived
    chains at the same trajectory; the 8^4 GPU rows gated against the CPU code (free values and one configuration).
(3) chi/free and e - e_free on (4,8) and (6,8) per channel (orbit means), y = 3.0 and P_c, as tabulated.
(4) (4^4, 8^4): no channel grows faster than free at the SSB scale (e - e_free + 2 sigma < 0.5 everywhere; at y = 3.0
    every channel <= +0.012); the anti-self-dual channels fall at y = 3.0.
(5) (6^4, 8^4) at y = 3.0: where e - e_free > 0.5 the 6^4 chi/free lies below both 4^4 and 8^4 (the 6^4 mode set).
(6) self-dual controls on 8^4.
About 30 s.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from masspairing.analysis import light_channels as LC
from masspairing.claimcheck import Check

c = Check("K5.16")

# ---- (1) orbits
o = LC.orbits()
c.item(
    "orbit classes (sizes) of the 16 mass strings under the stabiliser 32 and 8192 on 4^4 aaaa; identical",
    (o["sizes"], [len(k) for k in o["pg"]]),
    o["pg"] == o["all"] and sorted(len(k) for k in o["pg"]) == [1, 1, 2, 4, 4, 4] and o["sizes"] == (32, 8192),
)
for k in o["pg"]:
    c.record("  class", k)

# ---- (2) sample and gates
rep = LC.analyse()
dev = max(rep[p][L]["stored_dev"] for p in rep for L in rep[p])
c.item(
    "estimator asd(0,3) = stored chi_L on every configuration (max rel. dev.); n per box at y = 3.0 and P_c",
    (f"{dev:.1e}", {p: {L: rep[p][L]["n"] for L in rep[p]} for p in rep}),
    dev < 1e-12
    and {L: rep["y3"][L]["n"] for L in (4, 6, 8)} == {4: 225, 6: 250, 8: 139}
    and {L: rep["Pc"][L]["n"] for L in (4, 6, 8)} == {4: 225, 6: 225, 8: 45},
)
cm = LC.chain_match()
c.item(
    "stored chi_L of every row = the derived chain at the same trajectory (rows matched / rows, max rel. dev.)",
    {t: (m, n, float(d)) for t, (m, n, d) in cm.items()},
    all(m == n and d < 1e-12 for m, n, d in cm.values()),
)
g = LC.gates()
c.item(
    "8^4 GPU vs CPU over 22 channels: free (h = 2), one stored configuration; its (0,3) channel vs stored chi_L",
    f"{g['free']:.1e}, {g['cfg']:.1e}; {g['cfg_stored']:.1e}",
    g["free"] < 2e-14 and g["cfg"] < 2e-14 and g["cfg_stored"] < 1e-12,
)

# ---- (3) the tables
TAB = {
    "y3": {
        "asd(0,3)": ("0.0365", "0.00839", "0.0148", "-0.325", "+0.493"),
        "asd(0,1)": ("0.0214", "0.00406", "0.00529", "-0.504", "+0.231"),
        "asd(0,2)": ("0.0166", "0.00453", "0.00575", "-0.382", "+0.208"),
        "one-link LR": ("0.3903", "0.0970", "0.3353", "-0.055", "+1.077"),
        "on-site sextet (3,1)": ("0.9721", "0.1235", "0.9301", "-0.016", "+1.755"),
        "four-link sextet (3,1)": ("0.1410", "0.0756", "0.0763", "-0.221", "+0.009"),
        "three-link LR sextet (3,1)": ("0.0294", "0.00913", "0.0192", "-0.155", "+0.644"),
        "three-link LR sextet (1,3)": ("0.0172", "0.00485", "0.0129", "-0.101", "+0.853"),
        "four-link sextet (1,3)": ("-0.0133", "-0.0123", "-0.00519", None, None),
        "on-site sextet (1,3)": ("-0.1660", "-0.0222", "-0.1564", None, None),
    },
    "Pc": {
        "asd(0,3)": ("0.7942", "0.8006", "0.8508", "+0.025", "+0.053"),
        "asd(0,1)": ("0.7808", "0.7821", "0.8184", "+0.017", "+0.039"),
        "asd(0,2)": ("0.7815", "0.7794", "0.7995", "+0.008", "+0.022"),
        "one-link LR": ("1.1260", "1.0229", "1.2023", "+0.024", "+0.140"),
        "on-site sextet (3,1)": ("1.9584", "1.3019", "2.4346", "+0.079", "+0.544"),
        "four-link sextet (3,1)": ("0.9322", "1.0885", "1.0564", "+0.045", "-0.026"),
        "three-link LR sextet (3,1)": ("0.8069", "0.8527", "0.8408", "+0.015", "-0.012"),
        "three-link LR sextet (1,3)": ("0.8000", "0.8329", "0.8264", "+0.012", "-0.007"),
        "four-link sextet (1,3)": ("0.7453", "0.7242", "0.7623", "+0.008", "+0.045"),
        "on-site sextet (1,3)": ("0.5823", "0.6922", "0.5831", "+0.001", "-0.149"),
    },
}


def same(x, q):
    """x printed at the quoted number of decimals equals the quoted string."""
    d = len(q.split(".")[1])
    return f"{x:+.{d}f}".lstrip("+") == q.lstrip("+")


CP = LC.compact(rep)
for point, rows in CP.items():
    name = "y = 3.0" if point == "y3" else "P_c"
    for label, cells, e4, e6 in rows:
        q = TAB[point][label]
        shown = " | ".join(f"{cells[L][0]:.4g}({cells[L][1]:.2g})" for L in (4, 6, 8))
        shown += " | " + " | ".join("-" if e is None else f"{e[0]:+.3f}({e[1]:.3f})" for e in (e4, e6))
        ok = all(same(cells[L][0], q[i]) for i, L in enumerate((4, 6, 8)))
        ok &= all((e is None) == (qq is None) and (e is None or same(e[0], qq)) for e, qq in ((e4, q[3]), (e6, q[4])))
        c.item(f"{name}, {label}: chi/free 4^4 | 6^4 | 8^4 | e - e_free (4,8) | (6,8)", shown, ok)

# ---- (4) no SSB-scale growth on (4,8)
e48 = {p: [(lab, e4) for lab, _, e4, _ in CP[p] if e4 is not None] for p in CP}
top = max((v[0] + 2 * v[1], p, lab) for p in e48 for lab, v in e48[p])
c.item(
    "(4,8): max over channels and points of e - e_free + 2 sigma (< 0.5); max at y = 3.0 of e - e_free (<= +0.012)",
    f"{top[0]:+.3f} ({top[2]}, {top[1]}); {max(v[0] for _, v in e48['y3']):+.3f}",
    top[0] < 0.5 and max(v[0] for _, v in e48["y3"]) <= 0.0125,
)
c.item(
    "y = 3.0, (4,8): the three anti-self-dual Majorana channels fall (e - e_free + 2 sigma < 0)",
    [f"{v[0]:+.3f}({v[1]:.3f})" for lab, v in e48["y3"] if lab.startswith("asd")],
    all(v[0] + 2 * v[1] < 0 for lab, v in e48["y3"] if lab.startswith("asd")),
)

# ---- (5) the 6^4 outlier at y = 3.0
big = [(lab, cells) for lab, cells, _, e6 in CP["y3"] if e6 is not None and e6[0] > 0.5]
c.item(
    "y = 3.0, (6,8) e - e_free > 0.5: channels; in each the 6^4 chi/free is below both 4^4 and 8^4",
    [lab for lab, _ in big],
    len(big) == 4 and all(cells[6][0] < min(cells[4][0], cells[8][0]) for _, cells in big),
)

# ---- (6) controls
sd = {p: [rep[p][8]["chan"][ch]["ratio"] for ch in ("sd01", "sd02", "sd03")] for p in rep}
c.item(
    "8^4 self-dual controls sd(0,1), sd(0,2), sd(0,3) = C_chi direction, chi/free at y = 3.0 and P_c",
    {p: [round(x, 3) for x in v] for p, v in sd.items()},
    [round(x, 3) for x in sd["y3"]] == [0.008, 0.008, 0.0] and [round(x, 3) for x in sd["Pc"]] == [0.803, 0.802, 0.666],
)
c.done()
