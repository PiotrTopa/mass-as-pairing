"""K4.1: composite takeover -- across SYM -> P_c -> SMG the Majorana response to the nodal source C_0 on flavour 4
moves from the elementary field chi^4 to its epsilon-vertex partner Psi^4 = chi^1 chi^2 chi^3.

Reads the time series of the 23 chains (data/derived/n1inst/takeover/: phi_T, phi_f, Pfaffian sign; first quarter cut,
sign-weighted means, 8-block errors) and asserts:
 (a) SYM (y = 2.0): |phi_T|/phi_heavy < 0.02 on every row;
 (b) SMG (y = 3.0): |phi_T|/phi_heavy > 4 and phi_heavy/h < 0.05 on every row;
 (c) SMG: phi_T/h constant within 6 % over h in [0.05, 0.5] at 4^4, and 4^4 (h = 0.2, 0.5) vs 6^4 (h = 0.3) within 6 %;
 (d) P_c (y = 2.41, production bc): 0.02 < |phi_T|/phi_heavy < 0.2 on every row with h <= 0.3;
 (e) |phi_light/h| < 0.02 + 3 sigma on every row with h >= 0.05;
 control: the SMG ordering never occurs in a SYM row, the SYM ordering never in an SMG row.
About 5 s.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.analysis.n1inst import takeover_row
from masspairing.claimcheck import Check
from masspairing.data import derived, load_chain

c = Check("K4.1")
rows = []
for f in sorted(derived("n1inst", "takeover").glob("*.npz")):
    s, meta = load_chain(f)
    r = takeover_row(s, meta)
    r["name"] = f.stem
    rows.append(r)
rows.sort(key=lambda r: (r["y"], r["L"], r["h"], r["name"]))
print(f"  {'chain':22s} {'n':>4s} {'<s>':>5s} {'phi_T/h':>16s} {'phi_heavy/h':>16s} {'phi_light/h':>16s} {'ratio':>8s}")
for r in rows:
    print(
        f"  {r['name']:22s} {r['n']:4d} {r['sign']:5.2f} {r['phiT_h']:+8.4f}({r['ephiT_h']:.4f}) "
        f"{r['heavy_h']:+8.4f}({r['eheavy_h']:.4f}) {r['light_h']:+8.4f}({r['elight_h']:.4f}) {r['ratio']:8.3f}"
    )
sym = [r for r in rows if r["y"] == 2.0]
smg = [r for r in rows if r["y"] == 3.0]
pc = [r for r in rows if r["y"] == 2.41]
c.item("rows SYM / P_c / SMG (4^4 and 6^4)", (len(sym), len(pc), len(smg)), (len(sym), len(pc), len(smg)) == (6, 11, 6))
c.item(
    "(a) SYM: max |phi_T|/phi_heavy < 0.02", max(r["ratio"] for r in sym), all(r["ratio"] < 0.02 for r in sym), "{:.4f}"
)
c.item(
    "(b) SMG: min |phi_T|/phi_heavy > 4, max phi_heavy/h < 0.05",
    (round(min(r["ratio"] for r in smg), 2), round(max(r["heavy_h"] for r in smg), 4)),
    all(r["ratio"] > 4 and r["heavy_h"] < 0.05 for r in smg),
)
v4 = [r["phiT_h"] for r in smg if r["L"] == 4]
v6 = [r["phiT_h"] for r in smg if r["L"] == 6]
spread4 = (max(v4) - min(v4)) / abs(np.mean(v4))
v4b = np.mean([r["phiT_h"] for r in smg if r["L"] == 4 and r["h"] in (0.2, 0.5)])
vol = abs(np.mean(v6) - v4b) / abs(np.mean(v6))
c.record("(c) SMG phi_T/h: 4^4 mean, 6^4", (round(float(np.mean(v4)), 4), round(float(np.mean(v6)), 4)))
c.item("(c) SMG: 4^4 spread of phi_T/h over h in [0.05, 0.5] (< 6 %)", spread4, spread4 < 0.06, "{:.3f}")
c.item("(c) SMG: 4^4 (h = 0.2, 0.5) vs 6^4 (h = 0.3) (< 6 %)", vol, vol < 0.06, "{:.3f}")
pc3 = [r for r in pc if r["h"] <= 0.3]
c.item(
    "(d) P_c (production bc): |phi_T|/phi_heavy range on rows with h <= 0.3, inside (0.02, 0.2)",
    (round(min(r["ratio"] for r in pc3), 3), round(max(r["ratio"] for r in pc3), 3)),
    all(0.02 < r["ratio"] < 0.2 for r in pc3),
)
pc4 = [r["phiT_h"] for r in pc if r["L"] == 4 and r["h"] <= 0.1]
pc6 = [r["phiT_h"] for r in pc if r["L"] == 6 and r["h"] == 0.1]
c.item(
    "(d) P_c phi_T/h: 4^4 (h <= 0.1) range, 6^4 (h = 0.1): no volume growth",
    (round(max(pc4), 3), round(min(pc4), 3), round(pc6[0], 3)),
    abs(pc6[0]) < min(abs(v) for v in pc4),
)
e_rows = [r for r in rows if r["h"] >= 0.05]
c.item(
    f"(e) |phi_light/h| < 0.02 + 3 sigma on all {len(e_rows)} rows with h >= 0.05 (max |phi_light/h|)",
    max(abs(r["light_h"]) for r in e_rows),
    all(abs(r["light_h"]) < 0.02 + 3 * r["elight_h"] for r in e_rows),
    "{:.4f}",
)
c.item(
    "control: SMG ordering never in SYM rows, SYM ordering never in SMG rows",
    True,
    not any(r["ratio"] > 4 for r in sym) and not any(r["ratio"] < 0.02 for r in smg),
)
c.record("Pfaffian-sign averages below 1 (rows)", [(r["name"], round(r["sign"], 2)) for r in rows if r["sign"] < 1])
c.done()
