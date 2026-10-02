"""K5.2: symmetry zeros -- no Lorentz-scalar mass of the light doublet (Majorana or Dirac) without SSB, on
hypercubic-symmetric boxes and in infinite volume; the box-shape effect on L^3 x L_t and pppa boxes.

(1) signed-permutation symmetries of K (BFS sign fields): on hypercubic-symmetric 4^4 boxes (aaaa, pppp) elements with
    C_sd -> +C_sd, C_asd -> -C_asd exist (witness (0<->1, 2<->3), also on 6^4 aaaa); with the production bc none.
(2) free theory: <chi C_asd chi>/V = 0 on aaaa 4^4, 6^4; with the production bc V <chi C_asd chi>/<chi C_sd chi> is
    constant over L = 4, 6, 8 (a 1/V boundary effect).
(3) O(U^2) connected coefficient of the induced light amplitude on 4^4: zero (aaaa), non-zero (production bc).
(4) classification with the full group (hypercubic x one-site shifts, stabiliser of C_chi) on 4^4 aaaa, 4^4 pppa,
    4^3 x 8 aaaa; free phi_L consistent with it.
(5) every light scalar mass (three anti-self-dual plane masses and their P_L parts, four one-link Dirac masses and their
    L-R parts) averages to zero under the point-group stabiliser on 4^4 aaaa; one-link masses anticommute with Q and
    are scalar masses.
(6) box shape: on 4^4 aaaa the C_chi stabiliser splits 16/16 into elements keeping/flipping C_asd(0,3), every flipping
    element exchanges a spatial axis with time; on 4^3 x 8 all 8 keep it. Free phi_L on L^3 x L_t and pppa boxes is
    non-zero (frozen baselines; 4^3 x 8 and 6^3 x 12 at h = 2 recomputed live), zero on 4^4/6^4/8^4 aaaa.
About 6 min.
"""

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import scipy.sparse as sps

from masspairing.analysis.free import FREE_DIR, free_lookup, load_free
from masspairing.analysis.n1inst import free_phi_L, one_link_mass, stabiliser_elements
from masspairing.claimcheck import Check
from masspairing.data import derived
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.patterns import (
    TasteProjector,
    chiral_mass,
    light_pattern,
    scalar_mass_strings,
    singular_values,
    source_pattern,
)
from masspairing.symmetry import classify, sign_field, site_map

c = Check("K5.2")


def action_on(C, g, s):
    V = C.shape[0]
    T = sps.csr_matrix((s, (g, np.arange(V))), shape=(V, V))
    Cg = (T @ C @ T.T).tocsr()
    for sgn in (+1, -1):
        if abs(Cg - sgn * C).max() < 1e-12:
            return sgn
    return 0


# ---- (1) symmetries flipping C_asd while fixing C_sd
witness = ((1, 0, 3, 2), (1, 1, 1, 1))
for bc, lab in (((-1,) * 4, "all antiperiodic"), ((1,) * 4, "all periodic"), ((1, 1, 1, -1), "production")):
    lat = Lattice((4,) * 4, bc=bc)
    K = kinetic_matrix(lat)
    Csd, Casd = chiral_mass(lat), chiral_mass(lat, dual_sign=-1)
    n_sym, flip, fix = 0, [], []
    for perm in itertools.permutations(range(4)):
        for refl in itertools.product((1, -1), repeat=4):
            g = site_map(lat, perm, refl)
            s = sign_field(K, g)
            if s is None:
                continue
            n_sym += 1
            a, b = action_on(Csd, g, s), action_on(Casd, g, s)
            if a == 1:
                fix.append((perm, refl))
                if b == -1:
                    flip.append((perm, refl))
    if bc == (1, 1, 1, -1):
        c.item(
            f"4^4 {lab}: {n_sym} symmetries, {len(fix)} fix C_sd, {len(flip)} flip C_asd",
            len(flip),
            n_sym == 96 and not flip,
        )
    else:
        c.item(
            f"4^4 {lab}: {n_sym} symmetries, {len(fix)} fix C_sd, of which {len(flip)} flip C_asd (witness included)",
            len(flip),
            n_sym == 384 and len(fix) == 32 and len(flip) == 16 and witness in flip,
        )
lat6 = Lattice((6,) * 4, bc=(-1,) * 4)
g6 = site_map(lat6, *witness)
s6 = sign_field(kinetic_matrix(lat6), g6)
ok6 = (
    s6 is not None
    and action_on(chiral_mass(lat6), g6, s6) == 1
    and action_on(chiral_mass(lat6, dual_sign=-1), g6, s6) == -1
    and action_on(source_pattern(lat6), g6, s6) == 0
)
c.item("6^4 aaaa: witness maps C_sd -> +C_sd, C_asd -> -C_asd, C_0 -> neither", ok6, ok6)
c.item(
    "sign field is +-1 on every site (s^4 = 1 on the epsilon vertex, s^2 = 1 on on-site bilinears)",
    True,
    bool(np.all(np.abs(s6) == 1)),
)

# ---- (2) free theory
h = 0.5


def amplitudes(lat):
    G = np.linalg.inv(kinetic_matrix(lat).toarray() - h * chiral_mass(lat).toarray())
    light = -0.5 * np.einsum("ij,ji->", G, chiral_mass(lat, dual_sign=-1).toarray()) / lat.V
    heavy = -0.5 * np.einsum("ij,ji->", G, chiral_mass(lat).toarray()) / lat.V
    return light, heavy


for L in (4, 6):
    light, heavy = amplitudes(Lattice((L,) * 4, bc=(-1,) * 4))
    c.item(
        f"free L={L} aaaa h=0.5: <chi C_asd chi>/V (heavy {heavy:+.5f})",
        light,
        abs(light) < 1e-12 * abs(heavy),
        "{:+.1e}",
    )
ratios = []
for L in (4, 6, 8):
    lat = Lattice((L,) * 4)
    light, heavy = amplitudes(lat)
    ratios.append(light / heavy * lat.V)
r = np.array(ratios)
c.item(
    "production bc: V light/heavy at L = 4, 6, 8 (constant to 15 %)",
    np.round(r, 1).tolist(),
    abs(r).min() > 0 and abs(r).max() / abs(r).min() < 1.15,
)

# ---- (3) O(U^2) on 4^4
vals = {}
for bc, lab in (((-1,) * 4, "aaaa"), ((1, 1, 1, -1), "production")):
    lat = Lattice((4,) * 4, bc=bc)
    G = np.linalg.inv(kinetic_matrix(lat).toarray() - h * chiral_mass(lat).toarray())
    GCG = G @ chiral_mass(lat, dual_sign=-1).toarray() @ G
    iu = np.triu_indices(lat.V, 1)
    vals[lab] = 4.0 * np.sum(G[iu] ** 3 * GCG[iu]) / np.sum(G[iu] ** 4)
c.item(
    "O(U^2) connected coefficient / Z_2: aaaa (zero), production (non-zero)",
    (vals["aaaa"], vals["production"]),
    abs(vals["aaaa"]) < 1e-12 and abs(vals["production"]) > 1e-4,
)

# ---- (4) full-group classification and the free phi_L
rows = json.loads((FREE_DIR / "free_baselines.json").read_text())
ol = sorted((D, S) for S, D, anti in scalar_mass_strings() if len(D) == 1 and anti)
for shape, bc, expect in (
    ((4,) * 4, (-1,) * 4, True),
    ((4,) * 4, (1, 1, 1, -1), False),
    ((4, 4, 4, 8), (-1,) * 4, None),
):
    lat = Lattice(shape, bc=bc)
    tp = TasteProjector(lat)
    PL, PR = tp.dense_PL(), tp.dense_PR()
    pats = {
        "C_chi": chiral_mass(lat),
        "C_L": light_pattern(lat, proj=tp),
        "C_asd": chiral_mass(lat, dual_sign=-1),
        "C_0": source_pattern(lat),
    }
    if shape == (4, 4, 4, 8):
        for comp in ((0, 1), (0, 2)):
            Ca = chiral_mass(lat, comp, -1)
            pats[f"C_asd{comp}"] = Ca
            pats[f"P_L C_asd{comp} P_L"] = PL @ Ca.toarray() @ PL
        for D, S in ol:
            C1 = one_link_mass(lat, D[0], S).toarray()
            pats[f"onelink{D[0]}"] = C1
            pats[f"P_L onelink{D[0]} P_R + h.c."] = PL @ C1 @ PR + PR @ C1 @ PL
    res, nG, nS = classify(pats, lat, chiral_mass(lat))
    lab = "".join("a" if b == -1 else "p" for b in bc)
    tag = f"{shape} {lab} (|G| = {nG}, stabiliser {nS})"
    c.record(f"{tag}: group averages", {k: round(v, 3) for k, v in res.items()})
    c.item(f"{tag}: C_chi invariant (allowed)", res["C_chi"], abs(res["C_chi"] - 1) < 1e-10, "{:.3f}")
    fr = free_lookup(rows, shape[0], shape[-1], lab, 0.5)
    phiL = abs(sum(fr["phi_L"]))
    if expect is True:
        c.item(
            f"{tag}: C_L, C_asd, C_0 exact symmetry zeros",
            max(res["C_L"], res["C_asd"], res["C_0"]),
            max(res["C_L"], res["C_asd"], res["C_0"]) < 1e-10,
            "{:.1e}",
        )
    elif expect is False:
        c.item(
            f"{tag}: C_L, C_asd not zeros (production bc)",
            (round(res["C_L"], 3), round(res["C_asd"], 3)),
            res["C_L"] > 1e-3 and res["C_asd"] > 1e-3,
        )
    else:
        c.item(f"{tag}: C_L not a zero (no element flips it when L_t != L)", res["C_L"], res["C_L"] > 1e-3, "{:.3f}")
        others = [k for k in res if k.startswith(("C_asd(", "P_L C_asd(", "onelink", "P_L onelink"))]
        c.item(
            f"{tag}: the other two asd components, four one-link masses and their L/R parts: zeros (max)",
            max(res[k] for k in others),
            all(res[k] < 1e-10 for k in others),
            "{:.1e}",
        )
    c.item(
        f"{tag}: free phi_L(h = 0.5) = {phiL:.1e} vanishes iff C_L is a zero",
        phiL,
        (res["C_L"] < 1e-10) == (phiL < 1e-12),
        "{:.1e}",
    )

# ---- (5) point-group zeros of every light scalar mass
lat = Lattice((4,) * 4, bc=(-1,) * 4)
tp = TasteProjector(lat)
PL, PR, Q = tp.dense_PL(), tp.dense_PR(), tp.dense()
Cchi = chiral_mass(lat)
pats = {"C_chi": Cchi, "P_R C_chi P_R": PR @ Cchi.toarray() @ PR}
for comp in ((0, 3), (0, 1), (0, 2)):
    Ca = chiral_mass(lat, comp, -1)
    pats[f"C_asd{comp}"] = Ca
    pats[f"P_L C_asd{comp} P_L"] = PL @ Ca.toarray() @ PL
onel = {}
for D, S in ol:
    C1 = one_link_mass(lat, D[0], S)
    onel[D[0]] = C1
    pats[f"onelink{D[0]}"] = C1
    pats[f"P_L onelink{D[0]} P_R + h.c."] = PL @ C1.toarray() @ PR + PR @ C1.toarray() @ PL
res, nG, nS = classify(pats, lat, Cchi, translations=False)
c.item("4^4 aaaa point group and stabiliser of C_chi", (nG, nS), (nG, nS) == (384, 32))
c.item(
    "controls C_chi, P_R C_chi P_R invariant",
    (res["C_chi"], res["P_R C_chi P_R"]),
    abs(res["C_chi"] - 1) < 1e-10 and abs(res["P_R C_chi P_R"] - 1) < 1e-10,
)
asd = [res[f"C_asd{cc}"] for cc in ((0, 3), (0, 1), (0, 2))] + [
    res[f"P_L C_asd{cc} P_L"] for cc in ((0, 3), (0, 1), (0, 2))
]
c.item(
    "three anti-self-dual (light Majorana) masses and their P_L parts: zeros (max)",
    max(asd),
    max(asd) < 1e-12,
    "{:.1e}",
)
olz = [res[f"onelink{m}"] for m in range(4)] + [res[f"P_L onelink{m} P_R + h.c."] for m in range(4)]
c.item(f"four one-link Dirac masses {ol} and their L-R parts: zeros (max)", max(olz), max(olz) < 1e-12, "{:.1e}")
anti = all(
    abs(Q @ C1.toarray() + C1.toarray() @ Q).max() < 1e-12 and abs(Q @ C1.toarray() - C1.toarray() @ Q).max() > 0.1
    for C1 in onel.values()
)
c.item("one-link masses anticommute with Q (L-R Dirac masses)", anti, anti)
latp = Lattice((4,) * 4, bc=(1,) * 4)
sv = np.sort(singular_values(kinetic_matrix(latp).toarray() - 0.5 * one_link_mass(latp, 0, (0,)).toarray()))
c.item(
    "periodic 4^4: K - 0.5 C_onelink lifts all 16 corner modes to one gap (lowest 16, next)",
    (round(sv[0], 3), round(sv[15], 3), round(sv[16], 3)),
    np.all(np.abs(sv[:16] - 1) < 1e-10) and sv[16] > 1.5,
)

# ---- (6) box shape
nG4, el4 = stabiliser_elements((4, 4, 4, 4))
flips = [e for e in el4 if e["act"] < -0.5]
keeps = [e for e in el4 if e["act"] > 0.5]
c.item(
    "4^4 aaaa stabiliser: keep / flip C_asd(0,3); every flipping element moves the time axis",
    (len(keeps), len(flips)),
    len(keeps) == 16
    and len(flips) == 16
    and all(e["moves_time"] for e in flips)
    and all(abs(e["act_L"] - e["act"]) < 1e-9 for e in el4),
)
nG8, el8 = stabiliser_elements((4, 4, 4, 8))
c.item(
    "4^3 x 8 box group / stabiliser; all keep C_asd(0,3) (mean action on C_asd(0,3), P_L part)",
    (nG8, len(el8), round(np.mean([e["act"] for e in el8]), 3), round(np.mean([e["act_L"] for e in el8]), 3)),
    nG8 == 96 and len(el8) == 8 and all(e["act"] > 0.5 and e["act_L"] > 0.5 for e in el8),
)
free = load_free("free_baselines.json", "free_L4x8_aaaa.json", "free_L6x12_aaaa.json", "free_L8_aaaa.json")
f612 = free_lookup(free, 6, 12, "aaaa", 2.0)["phi_L"][0]
f48 = free_lookup(free, 4, 8, "aaaa", 2.0)["phi_L"][0]
live612, live48 = free_phi_L(6, 12, 2.0), free_phi_L(4, 8, 2.0)
c.item(
    "free phi_L per flavour, h = 2: 6^3 x 12, 4^3 x 8 (frozen)",
    (round(f612, 5), round(f48, 5)),
    abs(f612) > 1e-4 and abs(f48) > 1e-4,
)
c.item(
    "live recomputation of both (free_phi_L) = frozen baselines",
    max(abs(live612 - f612), abs(live48 - f48)),
    max(abs(live612 - f612), abs(live48 - f48)) < 1e-12,
    "{:.1e}",
)
f66p = free_lookup(free, 6, 6, "pppa", 2.0)["phi_L"][0]
c.item("6^4 pppa h = 2 (bc single out time)", f66p, abs(f66p) > 1e-4, "{:+.5f}")
hyp = [free_lookup(free, L, L, "aaaa", 2.0)["phi_L"][0] for L in (4, 6, 8)]
c.item("4^4 / 6^4 / 8^4 aaaa h = 2: zero (max)", max(abs(v) for v in hyp), max(abs(v) for v in hyp) < 1e-15, "{:.1e}")
fh = free_lookup(free, 6, 12, "aaaa", 2.0)["phi_f"][0]
c.item("6^3 x 12 h = 2: |phi_L| / phi_heavy (few per cent)", abs(f612) / fh, 0.01 < abs(f612) / fh < 0.1, "{:.3f}")
box = json.loads(derived("n1inst", "K52_box_shape.json").read_text())
series = {(r["L"], r["h"]): r["phi_L"] for r in box["aspect2"]}
c.record(
    "free phi_L at L_t = 2L, h = 1 / 2 (L = 4, 6, 8; frozen)",
    {L: (round(series[(L, 1.0)], 6), round(series[(L, 2.0)], 6)) for L in (4, 6, 8)},
)
d48 = max(abs(series[(4, 2.0)] - f48), abs(series[(6, 2.0)] - f612))
c.item("frozen box series = baselines at 4^3 x 8 and 6^3 x 12", d48, d48 < 1e-12, "{:.1e}")
fall = abs(series[(4, 2.0)]) / abs(series[(8, 2.0)])
c.item("L = 0 mod 4 class at L_t = 2L, h = 2: |phi_L| falls from L = 4 to 8 by", fall, fall > 5, "{:.1f}x")
c.done()
