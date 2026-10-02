"""K5.1: N1 instrument definitions -- the taste projectors P_L/P_R, the mass test of the stencilled pattern with the
phase-flipped control, taste chirality as a lattice pseudoscalar, the free baselines of every N1 observable at
L = 4/6/8 under both boundary conditions, and the light-doublet gate.

(1) TasteProjector on 4^4 (pppp, aaaa, pppa), 6^4 (aaaa, pppa), 4^3 x 8 (aaaa): Q real symmetric, Q^2 = Pi_nb, [Q, K] =
    [Q, plane operators] = [Q, C_L] = 0, P_L perp P_R, P_L + P_R = Pi_nb, FFT = dense, Q(0,1) = Q(0,2) = Q(0,3),
    Q(comp, -1) = -Q; boundary-mode counts.
(2) mass test on periodic 4^4 and 6^4: the merged CSR equals K - hC (16 / 32 neighbours for chiral / full); K - h C_chi
    has 8 zero singular values in P_L and 8 at h in P_R (h = 0.1, 0.5, 2); the phase-flipped pattern and C_0 fail it.
(3) taste chirality is a pseudoscalar: T Q T^T = +-Q under every signed-permutation symmetry of K and the shifts.
(4) free baselines (data/derived/free/free_baselines.json): the 4^4 and 4^3 x 8 rows and two 6^4 rows regenerated live
    (<= 1e-12); the gate -- GL_p0 = 1 on every all-antiperiodic box, GR_p0 = free-doublet formula < 1, light
    time-slice mass shift <= 0.02 for h <= 0.5 at L >= 6, heavy mass rising; 4^4 pppa readout void.
About 3 min.
"""

import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import scipy.sparse as sps

from masspairing.analysis.free import FREE_DIR, free_lookup, free_point, max_deviation
from masspairing.analysis.n1inst import flipped_pattern
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice, kinetic_matrix
from masspairing.operator import DoubletOperator
from masspairing.patterns import (
    PH,
    TasteProjector,
    chiral_mass,
    light_pattern,
    plane_op,
    singular_values,
    source_matrix,
    source_pattern,
)
from masspairing.symmetry import sign_field, site_map, translation

c = Check("K5.1")
LAB = lambda bc: "".join("a" if b == -1 else "p" for b in bc)  # noqa: E731

# ---- (1) projectors
for shape, bc in (
    ((4,) * 4, (1, 1, 1, 1)),
    ((4,) * 4, (-1,) * 4),
    ((4,) * 4, (1, 1, 1, -1)),
    ((6,) * 4, (-1,) * 4),
    ((6,) * 4, (1, 1, 1, -1)),
    ((4, 4, 4, 8), (-1,) * 4),
):
    lat = Lattice(shape, bc=bc)
    K = kinetic_matrix(lat).toarray()
    tp = TasteProjector(lat)
    Qm, Pnb, PL, PR = tp.dense(), tp.dense_Pnb(), tp.dense_PL(), tp.dense_PR()
    e = max(
        abs(Qm - Qm.T).max(),
        abs(Qm @ Qm - Pnb).max(),
        abs(Pnb @ Pnb - Pnb).max(),
        abs(Qm @ K - K @ Qm).max(),
        abs(PL @ PL - PL).max(),
        abs(PL @ PR).max(),
        abs(PL + PR - Pnb).max(),
    )
    e = max(e, max(abs(Qm @ plane_op(lat, *D).toarray() - plane_op(lat, *D).toarray() @ Qm).max() for D in PH))
    CL = light_pattern(lat, proj=tp)
    e = max(e, abs(Qm @ CL - CL @ Qm).max(), abs(CL + CL.T).max())
    e = max(e, max(abs(TasteProjector(lat, cc).dense() - Qm).max() for cc in ((0, 1), (0, 2))))
    e = max(e, abs(TasteProjector(lat, (0, 3), -1).dense() + Qm).max())
    v = np.random.default_rng(160).normal(size=(lat.V, 5))
    e = max(e, abs(tp.Q(v) - Qm @ v).max(), abs(tp.PL(v) - PL @ v).max())
    full = abs(Pnb - np.eye(lat.V)).max() < 1e-12
    c.item(
        f"{shape} {LAB(bc)}: projector algebra (boundary modes {tp.n_boundary}/{lat.V})",
        e,
        e < 1e-12 and full == (tp.n_boundary == 0),
        "{:.1e}",
    )

# ---- (2) mass test
for L in (4, 6):
    lat = Lattice((L,) * 4, bc=(1, 1, 1, 1))
    K = kinetic_matrix(lat).toarray()
    tp = TasteProjector(lat)
    PL, PR = tp.dense_PL(), tp.dense_PR()
    for name, nn in (("chiral", 16), ("full", 32)):
        t = DoubletOperator(lat, 0.0, h=0.7, pattern=name).merged_tables()
        Km = sps.csr_matrix((t["data0"] - 0.7 * t["ent_c"], t["indices"], t["indptr"]), shape=(lat.V,) * 2).toarray()
        e = abs(Km - (K - 0.7 * source_matrix(lat, name).toarray())).max()
        c.item(
            f"L={L} {name}: merged CSR = K - hC, {nn} neighbours",
            e,
            e < 1e-15 and np.all(np.diff(t["indptr"]) == nn),
            "{:.0e}",
        )
    C = source_matrix(lat, "chiral").toarray()
    for h in (0.1, 0.5, 2.0):
        w, v = np.linalg.eigh((K - h * C).T @ (K - h * C))
        s = np.sqrt(np.maximum(w, 0.0))
        z, hh = v[:, s < 1e-6], v[:, abs(s - h) < 1e-6]
        dz, dh = abs(PL @ z - z).max(), abs(PR @ hh - hh).max()
        c.item(
            f"L={L} h={h}: 8 zero modes in P_L, 8 modes at h in P_R (deviation)",
            max(dz, dh),
            z.shape[1] == 8 and hh.shape[1] == 8 and dz < 1e-9 and dh < 1e-9,
            "{:.1e}",
        )
    h = 0.5
    sf = singular_values(K - h * flipped_pattern(lat).toarray(), 16)
    s0 = singular_values(K - h * source_pattern(lat).toarray(), 20)
    sfull = singular_values(K - h * source_matrix(lat, "full").toarray(), 20)
    c.item(
        f"L={L} control: phase-flipped pattern fails (zero modes, lowest 16 at)",
        (int((sf < 1e-6).sum()), np.unique(np.round(sf, 4)).tolist()),
        int((sf < 1e-6).sum()) == 0 and np.allclose(sf, h / 2, atol=1e-6),
    )
    c.item(f"L={L} control: C_0 fails (zero modes stay)", int((s0 < 1e-6).sum()), int((s0 < 1e-6).sum()) == 16)
    c.record(
        f"L={L} full pattern: lowest 16 singular values (several gaps)", np.unique(np.round(sfull[:16], 4)).tolist()
    )
    c.item(f"L={L} full pattern lifts all 16 corner modes", int((sfull < 1e-6).sum()), int((sfull < 1e-6).sum()) == 0)

# ---- (3) pseudoscalar
witness = ((1, 0, 3, 2), (1, 1, 1, 1))
for bc, lab, n_exp in (
    ((-1,) * 4, "all antiperiodic", 384),
    ((1,) * 4, "all periodic", 384),
    ((1, 1, 1, -1), "production", 96),
):
    lat = Lattice((4,) * 4, bc=bc)
    K = kinetic_matrix(lat)
    Qm = TasteProjector(lat).dense()
    V = lat.V
    Csd = chiral_mass(lat).toarray()
    SD = np.array([chiral_mass(lat, cc, +1).toarray().reshape(-1) for cc in ((0, 1), (0, 2), (0, 3))])
    ASD = np.array([chiral_mass(lat, cc, -1).toarray().reshape(-1) for cc in ((0, 1), (0, 2), (0, 3))])

    def proj(B, v):
        return np.linalg.norm(B.T @ np.linalg.lstsq(B.T, v, rcond=None)[0] - v)

    n_sym = bad = n_fix = n_same = n_swap = 0
    wit_ok = None
    gens = [
        (site_map(lat, p, r), (p, r))
        for p in itertools.permutations(range(4))
        for r in itertools.product((1, -1), repeat=4)
    ]
    gens += [(translation(lat, mu), ("shift", mu)) for mu in range(4)]
    for g, name in gens:
        s = sign_field(K, g)
        if s is None:
            continue
        n_sym += 1
        T = sps.csr_matrix((s, (g, np.arange(V))), shape=(V, V))
        Qg = T @ Qm @ T.T
        sig = +1 if abs(Qg - Qm).max() < 1e-12 else (-1 if abs(Qg + Qm).max() < 1e-12 else 0)
        Cg = (T @ sps.csr_matrix(Csd) @ T.T).toarray().reshape(-1)
        in_sd, in_asd = proj(SD, Cg) < 1e-10, proj(ASD, Cg) < 1e-10
        if name == witness:
            wit_ok = sig == +1 and abs(Cg - Csd.reshape(-1)).max() < 1e-12
        if name[0] == "shift":
            bad += sig != -1
        elif in_sd:
            n_same += 1
            n_fix += abs(Cg - Csd.reshape(-1)).max() < 1e-12
            bad += sig != +1
        elif in_asd:
            n_swap += 1
            bad += sig != -1
        else:
            bad += 1
    c.item(
        f"4^4 {lab}: {n_sym - 4} hypercubic symmetries + 4 shifts; T Q T^T = +Q on {n_same} (keep the self-dual "
        f"triplet; {n_fix} fix C_chi), -Q on {n_swap} and on the shifts",
        bad,
        n_sym == n_exp + 4 and bad == 0 and (wit_ok or bc == (1, 1, 1, -1)),
    )

# ---- (4) free baselines and the gate
rows = json.loads((FREE_DIR / "free_baselines.json").read_text())
HS = [0.0, 0.2, 0.5, 1.0, 2.0]
live = [(4, 4, "aaaa", h, True) for h in HS] + [(4, 4, "pppa", h, True) for h in HS]
live += [(4, 8, "aaaa", h, True) for h in (0.0, 0.5)] + [(6, 6, "aaaa", 0.5, True), (6, 6, "pppa", 0.5, True)]
worst = 0.0
for L, Lt, bc, h, comp in live:
    worst = max(worst, max_deviation(free_point(L, Lt, bc, h, composite=comp), free_lookup(rows, L, Lt, bc, h)))
c.item(
    f"{len(live)} frozen baseline rows regenerated live (max abs deviation over every key)",
    worst,
    worst < 1e-12,
    "{:.1e}",
)


def get(L, bc, h, Lt=None):
    return free_lookup(rows, L, Lt or L, bc, h)


hs_here = {
    (4, "aaaa"): HS[1:],
    (6, "aaaa"): HS[1:],
    (8, "aaaa"): [0.5, 2.0],
    (4, "pppa"): HS[1:],
    (6, "pppa"): HS[1:],
    (8, "pppa"): [0.5],
}
for bc in ("aaaa", "pppa"):
    for L in (4, 6, 8):
        r0 = get(L, bc, 0.0)
        t = L // 2 - 1
        for h in hs_here[(L, bc)]:
            r = get(L, bc, h)
            p0 = np.asarray(r["p0"])
            s2 = 4.0 * np.sum(np.sin(p0) ** 2)
            cD, cd = np.cos(p0[0]) * np.cos(p0[3]), np.cos(p0[1]) * np.cos(p0[2])
            fR, fL = s2 / (s2 + (0.5 * h * (cD + cd)) ** 2), s2 / (s2 + (0.5 * h * (cD - cd)) ** 2)
            dL = r["mL_eff"][t] - r0["mL_eff"][t]
            dR = r["mR_eff"][1] - r0["mR_eff"][1] if L > 4 else r["mR_eff"][0] - r0["mR_eff"][0]
            tag = f"{bc} L={L} h={h}"
            if bc == "aaaa":
                c.item(
                    f"{tag}: GL_p0 = 1 exactly; GR_p0 = {r['GR_p0']:.4f} = free-doublet formula < 1",
                    abs(r["GL_p0"] - 1.0),
                    abs(r["GL_p0"] - 1.0) < 1e-10
                    and abs(r0["GL_p0"] - 1) < 1e-10
                    and abs(r["GR_p0"] - fR) < 1e-10
                    and r["GR_p0"] < 1,
                    "{:.1e}",
                )
            elif L == 4:
                c.item(
                    f"{tag}: momentum readout void (p0 on the zone boundary)",
                    (r["GL_p0"], r["GR_p0"]),
                    abs(r["GL_p0"]) < 1e-8 and abs(r["GR_p0"]) < 1e-8,
                )
            else:
                c.item(
                    f"{tag}: GL_p0 = {r['GL_p0']:.4f}, GR_p0 = {r['GR_p0']:.4f} = free-doublet formulas",
                    max(abs(r["GL_p0"] - fL), abs(r["GR_p0"] - fR)),
                    abs(r["GL_p0"] - fL) < 1e-10 and abs(r["GR_p0"] - fR) < 1e-10 and r["GR_p0"] < r["GL_p0"],
                    "{:.1e}",
                )
            if L > 4 or bc == "aaaa":
                c.item(f"{tag}: heavy m_R(t={1 if L > 4 else 0}) rises", dR, dR > 0, "{:+.3f}")
            if h <= 0.5 and L > 4:
                c.item(f"{tag}: light m_L shift at t* = L_t/2 - 1 (<= 0.02)", dL, abs(dL) <= 0.02, "{:+.4f}")
            elif L > 4:
                c.record(f"{tag}: light m_L shift at t* (not asserted: the O(h p^2) light term)", round(dL, 4))
r48, r480 = get(4, "aaaa", 0.5, 8), get(4, "aaaa", 0.0, 8)
p0 = np.asarray(r48["p0"])
s2 = 4 * np.sum(np.sin(p0) ** 2)
fL = s2 / (s2 + (0.25 * (np.cos(p0[0]) * np.cos(p0[3]) - np.cos(p0[1]) * np.cos(p0[2]))) ** 2)
d48 = r48["mL_eff"][3] - r480["mL_eff"][3]
c.item(
    f"4^3 x 8 aaaa h=0.5: GL_p0 = {r48['GL_p0']:.5f} (free-doublet formula), light m_L shift at t*",
    d48,
    abs(r48["GL_p0"] - fL) < 1e-10 and abs(d48) < 0.02,
    "{:+.4f}",
)
chiL = [
    (L, h, get(L, "aaaa", h)["chi_L_sum"] / get(L, "aaaa", 0.0)["chi_L_sum"] - 1)
    for L in (4, 6, 8)
    for h in hs_here[(L, "aaaa")]
    if h <= 1
]
c.item(
    "aaaa: chi_L_sum(h)/chi_L_sum(0) - 1 for h <= 1 (< 1 %), max",
    max(abs(v) for *_, v in chiL),
    all(abs(v) < 0.01 for *_, v in chiL),
    "{:.4f}",
)
c.record("8^4 aaaa: GR_p0 at h = 0, 0.5, 2", [round(get(8, "aaaa", h)["GR_p0"], 4) for h in (0.0, 0.5, 2.0)])
c.record("6^4 aaaa: GR_p0 at h = 0.2, 0.5, 1, 2", [round(get(6, "aaaa", h)["GR_p0"], 4) for h in HS[1:]])
c.record(
    "8^4 aaaa: chi_L_sum at h = 0, 0.5; chi_f_sum at h = 0, 0.5, 2",
    [round(get(8, "aaaa", h)["chi_L_sum"], 4) for h in (0.0, 0.5)]
    + [round(get(8, "aaaa", h)["chi_f_sum"], 3) for h in (0.0, 0.5, 2.0)],
)
c.done()
