"""K6.2: the odd-part frequency exponent alpha -- validation on known propagators, and SYM/SMG separated per
configuration on 39 stored configurations.

(1) synthetic partner propagator G(p) = (ip + M_p)/((ip)(ip + M_p) - Delta^2), p -> 2 sin p_0, embedded in corner space:
    M_p = 0 gives f_M = 0 and alpha > 0 iff Delta^2 > 4 sin p_1 sin p_3, alpha rising to +1; M_p != 0 gives f_M > 0 and
    the reading "majorana-pole" iff the light eigen-mass exceeds 2 sin p_1; the seesaw corner reads "gapless".
(2) free operator: massless -> gapless, alpha = -1, trivially quantised; Dirac mass -> f_M = m^2/(m^2 + 4 sin^2 p_1).
(3) K - h C_chi: heavy half f_M near h^2/(h^2 + 4 sin^2 p_1), light half f_M < 0.02 and alpha = -1 +- 0.1 (L >= 6);
    sabotage: the un-phased plane pair and C_0 give f_M = 0.
(4) the 39 stored configurations (data/derived/n1inst/K6_winding_stored.json; two 6^4 records recomputed live): none
    quantised; det G_B real; f_M does not separate AFM; alpha separates SMG from SYM at L = 6 and 8 with no overlap
    among scored, non-near-critical files; controls (swapped labels; near-critical files included).
About 1 min.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.analysis.n1inst import stored_config, winding_record
from masspairing.claimcheck import Check
from masspairing.corner import (
    CornerBlocks,
    analyse,
    analyse_blocks,
    classify,
    corner_block_of,
    doublet_operator,
    p0_loop,
    quantised,
    range_projectors,
)
from masspairing.data import derived
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.patterns import chiral_mass, pauli_string, plane_op, source_pattern

c = Check("K6.2")
G0 = pauli_string(0, 1)  # Gamma_0 = Z_0
MS = pauli_string(0b1111, 0)  # X_1111: anticommutes with Gamma_0, the eps(x) mass structure


def synthetic(Lt, Delta, Mp):
    loop, _ = p0_loop((4, 4, 4, Lt), (1, 1, 1, -1))
    blocks = []
    for p0 in loop:
        s = 2 * np.sin(p0)
        g = (1j * s + Mp) / ((1j * s) * (1j * s + Mp) - Delta**2)
        blocks.append(1j * g.real * MS + 1j * g.imag * G0)
    return analyse_blocks(blocks, loop)["all"]


# ---- (1) synthetic partner propagator
for Lt in (6, 8):
    p1, p3 = np.pi / Lt, 3 * np.pi / Lt
    c.record(f"L_t={Lt}: alpha > 0 needs Delta > 2 sqrt(sin p1 sin p3)", round(2 * np.sqrt(np.sin(p1) * np.sin(p3)), 3))
    prev = -9.0
    for Delta in (0.5, 1.0, 1.5, 2.0, 4.0, 8.0):
        r = synthetic(Lt, Delta, 0.0)
        pred = Delta**2 > 4 * np.sin(p1) * np.sin(p3)
        c.item(
            f"L_t={Lt} zero-type Delta={Delta}: alpha ({classify(r)})",
            r["alpha"],
            r["f_M"] < 1e-14 and (r["alpha"] > 0) == pred and r["alpha"] > prev,
            "{:+.3f}",
        )
        prev = r["alpha"]
    c.item(f"L_t={Lt}: alpha -> +1 (Delta = 8)", prev, prev > 0.9, "{:.3f}")
    for Delta, Mp in ((4.0, 2.0), (2.0, 4.0), (1.0, 2.0), (0.5, 3.0)):
        r = synthetic(Lt, Delta, Mp)
        ml = min(abs((Mp + sg * np.sqrt(Mp**2 + 4 * Delta**2)) / 2) for sg in (1, -1))
        want = "majorana-pole" if ml > 2 * np.sin(p1) else "gapless"
        c.item(
            f"L_t={Lt} M_p={Mp}, Delta={Delta} (light mass {ml:.3f}, 2 sin p1 {2 * np.sin(p1):.3f}): f_M, reading",
            (round(r["f_M"], 3), classify(r)),
            r["f_M"] > 1e-3 and classify(r) == want,
        )
    r = synthetic(Lt, 0.5, 3.0)
    c.item(
        f"L_t={Lt} seesaw corner (M_p = 3, Delta = 0.5): reads gapless, f_M",
        r["f_M"],
        classify(r) == "gapless" and r["f_M"] < 0.02,
        "{:.3f}",
    )

# ---- (2) free operator
for L in (6, 8):
    lat = Lattice((L,) * 4)
    z = np.zeros((3,) + lat.shape)
    p1 = np.pi / L
    r = analyse(CornerBlocks(doublet_operator(lat, 0.0, z), lat.shape, 2), lat.bc)["all"]
    c.item(
        f"L={L} massless: reading, alpha, quantised",
        (classify(r), round(r["alpha"], 3), quantised(r, 32)),
        classify(r) == "gapless" and abs(r["alpha"] + 1) < 1e-10 and quantised(r, 32),
    )
    for m in (0.5, 2.0):
        r = analyse(CornerBlocks(doublet_operator(lat, 0.0, z, mass=m), lat.shape, 2), lat.bc)["all"]
        fp = m**2 / (m**2 + 4 * np.sin(p1) ** 2)
        c.item(
            f"L={L} Dirac mass m={m}: f_M (formula {fp:.4f}), reading {classify(r)}",
            r["f_M"],
            abs(r["f_M"] - fp) < 1e-8,
            "{:.4f}",
        )

# ---- (3) K - h C_chi
for L in (4, 6, 8):
    lat, latp = Lattice((L,) * 4), Lattice((L,) * 4, bc=(1, 1, 1, 1))
    K, C = kinetic_matrix(lat), chiral_mass(lat)
    Ph, Pl = range_projectors(corner_block_of(chiral_mass(latp), latp.shape, np.zeros(4)))
    c.item(
        f"L={L}: rank of the p = 0 corner block of C_chi", round(np.trace(Ph).real), abs(np.trace(Ph).real - 8) < 1e-9
    )
    p1 = np.pi / L
    tol = {4: 0.30, 6: 0.12, 8: 0.06}[L]
    for h in (0.5, 1.0):
        r = analyse(CornerBlocks((K - h * C).astype(complex).tocsc(), lat.shape, 1), lat.bc, {"heavy": Ph, "light": Pl})
        fp = h**2 / (h**2 + 4 * np.sin(p1) ** 2)
        hv, lt = r["heavy"], r["light"]
        a_ok = L == 4 or abs(lt["alpha"] + 1) < 0.1
        c.item(
            f"L={L} h={h}: heavy f_M {hv['f_M']:.4f} vs {fp:.4f}, light f_M {lt['f_M']:.4f}, alpha {lt['alpha']:.3f}",
            hv["f_M"] / fp - 1,
            abs(hv["f_M"] / fp - 1) < tol and lt["f_M"] < 0.02 and a_ok,
            "{:+.3f}",
        )
    Cbad = (plane_op(lat, 0, 3, phase=()) + plane_op(lat, 1, 2, phase=())) / 8
    r = analyse(CornerBlocks((K - Cbad).astype(complex).tocsc(), lat.shape, 1), lat.bc, {"heavy": Ph, "light": Pl})
    c.item(
        f"L={L} sabotage (un-phased plane pair, h = 1): f_M",
        r["all"]["f_M"],
        r["all"]["f_M"] < 1e-12 and r["heavy"]["f_M"] < 1e-12,
        "{:.1e}",
    )
    for h in (0.5, 2.0):
        r = analyse(CornerBlocks((K - h * source_pattern(lat)).astype(complex).tocsc(), lat.shape, 1), lat.bc)["all"]
        c.item(
            f"L={L} K - {h} C_0 (nodal): f_M, alpha",
            (float(f"{r['f_M']:.1e}"), round(r["alpha"], 3)),
            r["f_M"] < 1e-12 and (L == 4 or abs(r["alpha"] + 1) < 1e-8),
        )

# ---- (4) stored configurations
recs = json.loads(derived("n1inst", "K6_winding_stored.json").read_text())["records"]
c.item(
    "records, one per phase-labelled stored chain", len(recs), len(recs) == 39 and len({r["file"] for r in recs}) == 39
)
for name, rel in (
    ("L6_y2.028_eps_pppa", "results/xi_scan/F_L6_k-0.01/L6_y2.028_k-0.01.npz"),
    ("L6_y2.792_eps_pppa", "results/xi_scan/F_L6_k-0.01/L6_y2.792_k-0.01.npz"),
):
    lat, meta, sig = stored_config(name)
    r = winding_record(sig, meta)
    fr = next(x for x in recs if x["file"] == rel)
    dev = max(abs(r["f_M"] - fr["f_M"]), abs(r["alpha"] - fr["alpha"]), abs(r["N_odd"] - fr["N_odd"]))
    c.item(
        f"live {name}: f_M {r['f_M']:.5f}, alpha {r['alpha']:.4f}, N_odd {r['N_odd']:.3f} = frozen",
        dev,
        dev < 1e-10,
        "{:.1e}",
    )
nq = sum(r["quantised"] for r in recs)
c.item(
    f"quantised configurations (N_odd in [{min(r['N_odd'] for r in recs):.1f}, {max(r['N_odd'] for r in recs):.1f}], "
    f"f_M in [{min(r['f_M'] for r in recs):.2f}, {max(r['f_M'] for r in recs):.2f}])",
    nq,
    nq == 0,
)
for L in (6, 8):
    g = [r for r in recs if r["L"] == L and r["scored"] and not r["near_critical"] and r["phase"] in ("AFM", "SMG")]
    c.item(
        f"L={L}: scored gapped (AFM/SMG, not near-critical) configurations quantised",
        f"0/{len(g)}",
        g and not any(r["quantised"] for r in g),
    )
c.item(
    f"det G_B real on all 39 (max |Im det|/|det|); half turns on {sum(r['det_half_turns'] > 0 for r in recs)} files",
    max(r["det_max_im_ratio"] for r in recs),
    all(r["det_max_im_ratio"] < 1e-9 for r in recs)
    and all(abs(r["det_winding"] - round(2 * r["det_winding"]) / 2) < 1e-6 for r in recs),
    "{:.1e}",
)


def sel(L, phase, nc_ok=False):
    return [
        r for r in recs if r["L"] == L and r["scored"] and (nc_ok or not r["near_critical"]) and r["phase"] == phase
    ]


for L in (6, 8):
    afm = sel(L, "AFM") or [r for r in recs if r["L"] == L and r["scored"] and r["phase"] == "AFM"]
    rest = sel(L, "SYM") + sel(L, "SMG")
    lo, hi = min(r["f_M"] for r in afm), max(r["f_M"] for r in rest)
    c.item(
        f"L={L}: AFM f_M >= {lo:.3f} (near-critical) inside SYM/SMG range (<= {hi:.3f}): no separation",
        lo < hi,
        lo < hi,
    )
margins = {}
for L in (6, 8):
    smg, sym = sel(L, "SMG"), sel(L, "SYM")
    lo, hi = min(r["alpha"] for r in smg), max(r["alpha"] for r in sym)
    margins[L] = lo - hi
    c.item(
        f"L={L}: min alpha(SMG) {lo:+.3f} ({len(smg)}) > max alpha(SYM) {hi:+.3f} ({len(sym)}): margin",
        lo - hi,
        lo > hi,
        "{:.2f}",
    )
    c.item(
        f"L={L}: every scored non-near-critical SYM file has alpha = -1 +- 0.25",
        [round(r["alpha"], 2) for r in sym],
        all(abs(r["alpha"] + 1) < 0.25 for r in sym),
    )
    c.item(
        f"L={L} control: labels swapped -> no separation",
        True,
        not (min(r["alpha"] for r in sym) > max(r["alpha"] for r in smg)),
    )
    smg_nc, sym_nc = sel(L, "SMG", True), sel(L, "SYM", True)
    c.record(
        f"L={L} with near-critical files: min alpha(SMG), max alpha(SYM)",
        (round(min(r["alpha"] for r in smg_nc), 3), round(max(r["alpha"] for r in sym_nc), 3)),
    )
c.item(
    "margins L = 6 > 0.3, L = 8 > 0.1",
    (round(margins[6], 2), round(margins[8], 2)),
    margins[6] > 0.3 and margins[8] > 0.1,
)
crit = [r["alpha"] for r in recs if r["phase"] == "critical"]
c.record(f"critical files (P_c, {len(crit)}): alpha range", (round(min(crit), 2), round(max(crit), 2)))
for y in (2.537, 2.792):
    a6 = next(r["alpha"] for r in recs if r["L"] == 6 and r["kappa"] == -0.01 and abs(r["y"] - y) < 1e-6)
    a8 = next(r["alpha"] for r in recs if r["L"] == 8 and r["kappa"] == -0.01 and abs(r["y"] - y) < 1e-6)
    c.record(f"(y = {y}, kappa = -0.01): alpha L = 6 -> 8", (round(a6, 3), round(a8, 3)))
c.done()
