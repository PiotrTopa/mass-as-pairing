"""The light block of the source per box, the restricted-mode control of the light pair susceptibility, and the
(4^4, 8^4) pair at P_c (K5.15).

(a) ``light_block``: the light-doublet block P_L C_chi P_L of the source, resolved on the V twisted plane waves of a
    box (matrix-free: P_L by FFT, C_chi sparse). ||P_L C_chi P_L||_F^2 = sum_p ||P_L C_chi P_L v_p||^2 over the
    orthonormal plane waves; on every momentum ||P_L C_chi P_L v_p|| = 2^{-3/2} | |cos p0 cos p3| - |cos p1 cos p2| |
    where the light/heavy split exists, 0 elsewhere (``light_block_formula`` evaluates this closed form). A momentum
    is source-free when the light block vanishes on it.
(b) ``control``: chi_L = V <phi_L^2> on stored 8^4 (y, h) = (3.0, 2) configurations (stage-1c replicas rA, rB; every
    second stored configuration with trajectory >= 100; 45 + 45) with P_L restricted to the source-free momenta
    (1536 of 4096) or to the IR class (the 256 momenta with every |cos p_mu| = cos(pi/8), the 8^4 analogue of the 256
    momenta of the 6^4 split), and the full P_L value stored with the chain, from
    data/derived/k5control/control_L8_y3_h2.npz (produced by scripts/run_k5_control.py from the raw archive). The
    exponent e = d ln chi / d ln V against the full chi_L of 4^4 (stage 0, trajectories >= 150) and 6^4 (stage 1b,
    trajectories >= 1000), each relative to the free exponent of the same observables. Errors as in
    ``complete_pair``: 10 blocks per member (20 for the pooled series), raised to sigma_naive sqrt(2 tau_int).
(c) ``pc_pair``: chi_L/free, e - e_free and g = G_L(p_min)/free on (4^4, 8^4) and (6^4, 8^4) at P_c = (2.41, -0.01),
    4^4 stage 0 (>= 150), 6^4 and 8^4 stage 1 (>= 100), with the same reader.

``chi_reduced`` is the reduced-basis estimator: for a restricted light pattern C_L' = W c W^T (W a real orthonormal
basis of a P_L-invariant momentum subset, c = W^T C_L W) the flavour-summed p = 0 pair susceptibility of
``N1Measure.pair_channel`` reduces to the r x r blocks B_ab = W^T G^{ab} W from one sparse LU of the doublet operator:
chi = [(sum_a phi_a)^2 + 1/2 sum_ab tr(B_ab^T c B_ab c)] / V, phi_a = -1/2 tr(c B_aa).
"""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse.linalg as spla

from ..corner import doublet_operator
from ..flavour import P as FP
from ..flavour import U as FU
from ..lattice import Lattice
from ..patterns import TasteProjector, chiral_mass
from .complete_pair import FREE_FILES, blocked, chain_chi, chain_path
from .n1stage import DERIVED, N1DIR, exponent, free_lookup, free_tables, open_chain, tau_int

CONTROL = DERIVED / "k5control"
CONTROL_COLUMNS = ("traj", "chi_ir", "chi_sf", "phi_ir", "phi_sf", "chi_full_stored")
V4, V6, V8 = 256, 1296, 4096
A_LIGHT = 2.0**-1.5


# ------------------------------------------------------------ (a) light block of the source
def _momenta(lat):
    ns = np.array(np.meshgrid(*[np.arange(L) for L in lat.shape], indexing="ij")).reshape(lat.d, -1).T
    th = np.array([0.5 if b == -1 else 0.0 for b in lat.bc])
    p = 2 * np.pi * (ns + th) / np.array(lat.shape)
    return ns, p


def plane_waves(lat, p):
    """(V, n) complex matrix of the twisted plane waves e^{i p.x}/sqrt(V) for momenta p (n, d)."""
    co = lat.coords.reshape(lat.d, lat.V)
    return np.exp(1j * (p @ co)).T / np.sqrt(lat.V)


def light_block(shape, bc=(-1, -1, -1, -1), chunk=256):
    """Momentum-resolved light block of C_chi on one box (matrix-free)."""
    lat = Lattice(shape, bc=bc)
    C = chiral_mass(lat).tocsr()
    tp = TasteProjector(lat)
    _, p = _momenta(lat)
    mv = np.empty(lat.V)
    nl = np.empty(lat.V)
    for i in range(0, lat.V, chunk):
        W = plane_waves(lat, p[i : i + chunk])
        PLW = tp.PL(W)
        nl[i : i + chunk] = np.linalg.norm(PLW, axis=0)
        mv[i : i + chunk] = np.linalg.norm(tp.PL(C @ PLW), axis=0)
    cosp = np.cos(p)
    split = nl > 1e-9
    formula = np.where(split, A_LIGHT * np.abs(np.abs(cosp[:, 0] * cosp[:, 3]) - np.abs(cosp[:, 1] * cosp[:, 2])), 0.0)
    normC = float(np.sqrt((C.data**2).sum()))
    return dict(
        V=lat.V,
        frac=float(np.sqrt((mv**2).sum()) / normC),
        n_split=int(split.sum()),
        n_source_free=int((split & (mv < 1e-9)).sum()),
        n_dressed=int((split & (mv >= 1e-9)).sum()),
        max_formula_dev=float(np.abs(mv - formula).max()),
        source_free_mask=(split & (mv < 1e-9)).reshape(shape),
        mv=mv,
        abs_cos=np.abs(cosp),
    )


def light_block_formula(shape, bc=(-1, -1, -1, -1)):
    """The closed form: ||P_L C P_L||_F / ||C||_F and the momentum counts, without any operator but sparse C."""
    lat = Lattice(shape, bc=bc)
    C = chiral_mass(lat).tocsr()
    _, p = _momenta(lat)
    c = np.cos(p)
    split = np.all(np.abs(c) > 1e-12, axis=1)
    m = np.where(split, A_LIGHT * np.abs(np.abs(c[:, 0] * c[:, 3]) - np.abs(c[:, 1] * c[:, 2])), 0.0)
    return dict(
        frac=float(np.sqrt((m**2).sum()) / np.sqrt((C.data**2).sum())),
        n_split=int(split.sum()),
        n_source_free=int((split & (m < 1e-9)).sum()),
        n_dressed=int((split & (m >= 1e-9)).sum()),
    )


def ir_class_mask(shape, bc=(-1, -1, -1, -1)):
    """Momenta with every |cos p_mu| equal to the largest value on the box (the momenta nearest the corners)."""
    lat = Lattice(shape, bc=bc)
    _, p = _momenta(lat)
    a = np.abs(np.cos(p))
    return np.all(np.abs(a - a.max()) < 1e-9, axis=1).reshape(shape)


# ------------------------------------------------------------ reduced-basis estimator
def momentum_mask_projector(lat, mask):
    """Dense real projector onto the plane waves with momentum index in mask (mask symmetric under p -> -p)."""
    _, p = _momenta(lat)
    Wp = plane_waves(lat, p[mask.reshape(-1)])
    P = Wp @ Wp.conj().T
    assert np.abs(P.imag).max() < 1e-10, "mask not symmetric under p -> -p"
    return P.real


def light_basis(lat, mask=None):
    """(W, c): a real orthonormal basis of P_L (restricted to mask) and the light pattern in it."""
    PL = TasteProjector(lat).dense_PL()
    P = PL if mask is None else PL @ momentum_mask_projector(lat, mask)
    assert np.abs(P - P.T).max() < 1e-9
    w, v = np.linalg.eigh(P)
    W = v[:, w > 0.5]
    CL = PL @ chiral_mass(lat, (0, 3), -1).toarray() @ PL
    CL = 0.5 * (CL - CL.T)
    CL[np.abs(CL) < 1e-14] = 0.0
    return W, W.T @ CL @ W


def chi_reduced(lat, y, sigma, h, W, c, lu=None):
    """(chi_L', phi_L') of one sigma configuration for the restricted light pattern W c W^T."""
    V, r = lat.V, W.shape[1]
    if lu is None:
        lu = spla.splu(doublet_operator(lat, y, sigma, h=h, pattern="chiral"))
    X = np.zeros((2, 2, r, r), complex)
    for q in range(2):
        R = np.zeros((2 * V, r), complex)
        R[q::2, :] = W
        Z = lu.solve(R)
        for pp in range(2):
            X[pp, q] = W.T @ Z[pp::2, :]
    B = np.zeros((4, 4, r, r))
    for a in range(4):
        for b in range(4):
            B[a, b] = (FU[a] * np.conj(FU[b]) * X[FP[a], FP[b]]).real
    phi = sum(-0.5 * np.trace(c @ B[a, a]) for a in range(4))
    conn = sum(0.5 * np.trace(B[a, b].T @ c @ B[a, b] @ c) for a in range(4) for b in range(4))
    return float((phi**2 + conn) / V), float(phi / V)


# ------------------------------------------------------------ (b) the control at (3.0, h = 2) on 8^4
def _stat(x, nb):
    m, s = blocked(x, nb)
    tau = tau_int(x)
    return m, max(s, float(np.std(x, ddof=1) / np.sqrt(len(x)) * np.sqrt(2 * max(tau, 0.5)))), float(tau)


def control_data():
    z = np.load(CONTROL / "control_L8_y3_h2.npz", allow_pickle=False)
    free = json.loads((CONTROL / "free_L8_h2_restricted.json").read_text())
    return {"rA": z["rA"], "rB": z["rB"]}, free


def control():
    """Rows per light pattern (full, ir, sf): per replica and pooled chi'/free, e - e_free on (4,8) and (6,8)."""
    data, fl = control_data()
    free = free_tables(FREE_FILES)
    fr4, fr6, fr8 = (free_lookup(free, L, L, "aaaa", 2.0)["chi_L_sum"] for L in (4, 6, 8))
    r4 = chain_chi(chain_path("S0", "L4", 4, "_h2"), tmin=150)
    r6 = chain_chi(chain_path("S1b", "L6", 6, "_h2"), tmin=1000)
    allc = np.concatenate([data["rA"], data["rB"]])
    out = dict(
        L4=dict(chi=r4["chi"], err=r4["err"], n=r4["n"], over_free=r4["chi"] / fr4),
        L6=dict(chi=r6["chi"], err=r6["err"], n=r6["n"], over_free=r6["chi"] / fr6),
        free=dict(full=fr8, ir=fl["chi_ir"], sf=fl["chi_sf"], L4=fr4, L6=fr6),
        n=dict(rA=len(data["rA"]), rB=len(data["rB"])),
        rows={},
    )
    for label, col, fref in (("full", 5, fr8), ("ir", 1, fl["chi_ir"]), ("sf", 2, fl["chi_sf"])):
        row = {}
        for rep, a in data.items():
            m, s, tau = _stat(a[:, col], 10)
            row[rep] = dict(chi=m, err=s, tau=tau, over_free=m / fref, over_free_err=s / fref)
        m, s, _ = _stat(allc[:, col], 20)
        e48, ee48 = exponent(r4["chi"], r4["err"], m, s, V4, V8)
        e68, ee68 = exponent(r6["chi"], r6["err"], m, s, V6, V8)
        ef48, ef68 = np.log(fref / fr4) / np.log(V8 / V4), np.log(fref / fr6) / np.log(V8 / V6)
        row["pooled"] = dict(chi=m, err=s, over_free=m / fref, over_free_err=s / fref)
        row["e48"] = dict(e=e48, err=ee48, e_free=ef48, minus_free=e48 - ef48)
        row["e68"] = dict(e=e68, err=ee68, e_free=ef68, minus_free=e68 - ef68)
        out["rows"][label] = row
    for label, col, fref in (("ir", 1, fl["chi_ir"]), ("sf", 2, fl["chi_sf"])):
        r = allc[:, col] / allc[:, 5]
        out["rows"][label]["share"] = dict(mean=float(r.mean()), median=float(np.median(r)), free=fref / fr8)
    return out


def control_vs_chains():
    """Max relative deviation of the stored full chi_L of the control rows from the derived stage-1c chains at the
    same trajectories, and the number of rows matched."""
    data, _ = control_data()
    dev, n = 0.0, 0
    for rep, a in data.items():
        d, _ = open_chain(chain_path("S1c", f"L8_h2_{rep}", 8, "_h2"))
        traj = np.asarray(d["ts_cfg_traj"])
        chi = np.asarray(d["ts_chi_L_sum"], float)
        for t, v in zip(a[:, 0], a[:, 5], strict=True):
            i = np.where(traj == int(t))[0]
            if len(i) == 1:
                n += 1
                dev = max(dev, abs(chi[i[0]] - v) / abs(chi[i[0]]))
    return dev, n


# ------------------------------------------------------------ (c) the pair at P_c
PC_FREE_FILES = FREE_FILES + ["free_L6_aaaa_h1.5_3.json"]


def _pc_path(L, y, tag):
    stage, sub = ("S0", "L4") if L == 4 else ("S1", f"L{L}")
    return N1DIR / stage / sub / f"L{L}_y{y}_k-0.01_g0_g60{tag}_chiral_bcaaaa.npz"


def _chain_obs(path, tmin, key, nb=10):
    d, _ = open_chain(path)
    keep = np.asarray(d["ts_cfg_traj"]) >= tmin
    x = np.asarray(d[key], float)[keep]
    m, s, _ = _stat(x, nb)
    return m, s


def pc_pair(y="2.41"):
    """Rows per h: chi_L/free and g on 4^4, 6^4, 8^4; e - e_free and g ratios on (4,8) and (6,8)."""
    free = free_tables(PC_FREE_FILES)
    V = {4: V4, 6: V6, 8: V8}
    rows = {}
    for h, tag in ((0.0, ""), (0.5, "_h0.5"), (1.0, "_h1"), (2.0, "_h2")):
        box = {}
        for L in (4, 6, 8):
            p = _pc_path(L, y, tag)
            tmin = 150 if L == 4 else 100
            fr = free_lookup(free, L, L, "aaaa", h)
            c = chain_chi(p, tmin)
            g, ge = _chain_obs(p, tmin, "ts_GL_p0")
            box[L] = dict(
                chi=c["chi"],
                err=c["err"],
                n=c["n"],
                fchi=fr["chi_L_sum"],
                over_free=c["chi"] / fr["chi_L_sum"],
                g=g / fr["GL_p0"],
                g_err=ge / fr["GL_p0"],
            )
        row = dict(box=box)
        for La, Lb in ((4, 8), (6, 8)):
            a, b = box[La], box[Lb]
            e, ee = exponent(a["chi"], a["err"], b["chi"], b["err"], V[La], V[Lb])
            ef = np.log(b["fchi"] / a["fchi"]) / np.log(V[Lb] / V[La])
            row[f"e{La}{Lb}"] = dict(e=e, err=ee, e_free=ef, minus_free=e - ef, g_ratio=b["g"] / a["g"])
        rows[h] = row
    return rows
