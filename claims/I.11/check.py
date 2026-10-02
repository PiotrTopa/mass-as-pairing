"""I.11: the light/heavy taste observables of the N1 measure are exact.

(1) free theory (y = 0) on all-antiperiodic 4^4 and 4^3 x 8 (no zone-boundary modes, P_L + P_R = 1), h in {0, 0.5}:
    the light/heavy split is exact (sum_t (CL_t + CR_t) = sum_t C_full(t), (GL_p0 + GR_p0)/2 = G_p0); at h = 0 light =
    heavy (CL_t = CR_t, GL_t = GR_t, GL_p0 = GR_p0 = 1); the momentum readouts equal the free-doublet formula
    4 sum sin^2 p0 / (4 sum sin^2 p0 + m(p0)^2), m_R = h (c_D + c_dual)/2, m_L = h (c_D - c_dual)/2; phi6L = 0; phi_R =
    the doublet-code phi_src of the channel measure with the chiral pattern; on 4^4 phi_L = 0.
(2) response identities on an interacting 4^4 aaaa configuration (y 2.41, kappa -0.01, h 0.3 chiral): chi_f_sum_conn =
    d<sum_a O^(a)>/dh and chi_L_sum_conn = d<O_L>/dh_L (light source h_L C_L added to the operator), central
    differences; the sabotage (sign of the second exchange term flipped) fails both.
(3) stochastic (n_noise 4 x 16 seeds) = dense on every taste column; the point-source taste_correlator (CG) = the dense
    P G P columns.
(4) the taste channels are added only with the chiral pattern: the C_0 measure has no taste keys and its pair channel
    is the C_0 one.
About 1 min.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

from masspairing.action import build_model
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice
from masspairing.measure import ChannelMeasure, N1Measure

c = Check("I.11")
AA = (-1, -1, -1, -1)


def n1(lat, y, kappa, h, **kw):
    return build_model(lat, y, kappa, 1.0, h=h, pattern="chiral", **kw)


# ---- (1) free theory
for shape in ((4, 4, 4, 4), (4, 4, 4, 8)):
    lat = Lattice(shape, bc=AA)
    b = {}
    for h in (0.0, 0.5):
        m = n1(lat, 0.0, -0.01, h)
        f0 = np.zeros(m.nfields)
        fm = N1Measure(m, exact=True, taste=True)
        b[h] = fm.bilinears(f0)
        Cfull = fm.correlator(f0)["Cf_f"].sum(0)
        e_split = max(
            abs((b[h]["CL_t"] + b[h]["CR_t"]).sum() - Cfull.sum()),
            abs(0.5 * (b[h]["GL_p0"] + b[h]["GR_p0"]) - b[h]["G_p0"]),
        )
        ps = ChannelMeasure(m, exact=True).bilinears(f0)["phi_src"] if h else 0.0
        e_amp = max(abs(b[h]["phi6L"]).max(), abs(b[h]["phi_f"].sum() - ps))
        if shape == (4, 4, 4, 4):
            e_amp = max(e_amp, abs(b[h]["phi_L"]).max())
        p0 = fm.p0
        s2 = 4.0 * np.sum(np.sin(p0) ** 2)
        cD, cd = np.cos(p0[0]) * np.cos(p0[3]), np.cos(p0[1]) * np.cos(p0[2])
        mR, mL = 0.5 * h * (cD + cd), 0.5 * h * (cD - cd)
        e_p0 = max(abs(b[h]["GR_p0"] - s2 / (s2 + mR**2)), abs(b[h]["GL_p0"] - s2 / (s2 + mL**2)))
        tag = f"{shape} h={h}"
        c.item(f"{tag}: light/heavy split exact", e_split, e_split < 1e-10, "{:.1e}")
        what = "phi6L = 0, phi_R = phi_src" + (", phi_L = 0" if shape == (4, 4, 4, 4) else "")
        c.item(f"{tag}: {what}", e_amp, e_amp < 1e-10, "{:.1e}")
        c.item(
            f"{tag}: GL_p0 {b[h]['GL_p0']:.5f}, GR_p0 {b[h]['GR_p0']:.5f} = free-doublet formula",
            e_p0,
            e_p0 < 1e-10,
            "{:.1e}",
        )
        c.record(f"{tag}: phi_L, phi_T_R, phi_T_L", [float(b[h]["phi_L"].sum()), b[h]["phi_T_R"], b[h]["phi_T_L"]])
    e_h0 = max(
        abs(b[0.0]["CL_t"] - b[0.0]["CR_t"]).max(),
        abs(b[0.0]["GL_t"] - b[0.0]["GR_t"]).max(),
        abs(b[0.0]["GL_p0"] - b[0.0]["GR_p0"]),
        abs(b[0.0]["GL_p0"] - 1.0),
    )
    c.item(f"{shape} h=0: light = heavy, GL_p0 = GR_p0 = 1", e_h0, e_h0 < 1e-10, "{:.1e}")
gl, gr = b[0.5]["GL_p0"], b[0.5]["GR_p0"]
c.item(
    "4^3 x 8 h=0.5: GL_p0 = 0.99978, GR_p0 = 0.98753",
    [round(gl, 5), round(gr, 5)],
    abs(gl - 0.99978) < 5e-6 and abs(gr - 0.98753) < 5e-6,
)

# ---- (2) response identities
lat = Lattice((4, 4, 4, 4), bc=AA)
V = lat.V
m = n1(lat, 2.41, -0.01, 0.3, cg_tol=1e-11)
F = np.asarray(m.start(np.random.default_rng(7), "hot"))
fm = N1Measure(m, exact=True, taste=True)
bx = fm.bilinears(F)
d = 1e-3


def phi_sum(h):
    return N1Measure(n1(lat, 2.41, -0.01, h), exact=True, taste=True, six_fermion=False).bilinears(F)["phi_f"].sum()


fdR = (phi_sum(0.3 + d) - phi_sum(0.3 - d)) / (2 * d)
CL = fm.CL
M4 = m.D.dense_real4(F)


def phiL_sum(hL):
    G = np.linalg.inv(M4 - hL * np.kron(CL, np.eye(4))).reshape(V, 4, V, 4)
    return fm.pair_channel(G=G, C=CL, prefix="L", summed=True)["phi_L"].sum()


fdL = (phiL_sum(d) - phiL_sum(-d)) / (2 * d)
e2 = max(abs(bx["chi_f_sum_conn"] - fdR) / abs(fdR), abs(bx["chi_L_sum_conn"] - fdL) / abs(fdL))
c.item(
    f"chi_f_sum_conn {bx['chi_f_sum_conn']:.5f} vs dphi_R/dh {fdR:.5f}; chi_L_sum_conn {bx['chi_L_sum_conn']:.5f} vs "
    f"dphi_L/dh_L {fdL:.5f} (relative)",
    e2,
    e2 < 1e-4,
    "{:.1e}",
)


def conn_sum(G, C, sgn3):
    """Flavour-summed connected susceptibility sum_ab sum_xz [A o CAC/4 - sgn3 CA o AC/4] / V (A = G^ab)."""
    Cd = C if isinstance(C, np.ndarray) else C.toarray()
    tot = 0.0
    for a in range(4):
        for bb in range(4):
            A = G[:, a, :, bb]
            CA, AC = Cd @ A, A @ Cd
            tot += (0.25 * A * (CA @ Cd) - sgn3 * 0.25 * CA * AC).sum()
    return tot / V


G = m.D.dense_G(F)
e_re = max(abs(conn_sum(G, fm.CR, 1.0) - bx["chi_f_sum_conn"]), abs(conn_sum(G, CL, 1.0) - bx["chi_L_sum_conn"]))
c.item("local recomputation of the two connected sums", e_re, e_re < 1e-12, "{:.1e}")
sab = min(abs(conn_sum(G, fm.CR, -1.0) - fdR) / abs(fdR), abs(conn_sum(G, CL, -1.0) - fdL) / abs(fdL))
c.item("sabotage (second exchange term sign flipped) off by at least", sab, sab > 0.05, "{:.0%}")

# ---- (3) stochastic vs dense
keys = [
    "phi_L",
    "chi_L",
    "chi_L_conn",
    "chi_L_pmin",
    "chi_L_corners",
    "chi_f_sum",
    "chi_f_sum_conn",
    "chi_f_sum_pmin",
    "chi_f_sum_corners",
    "chi_L_sum",
    "chi_L_sum_conn",
    "chi_L_sum_pmin",
    "chi_L_sum_corners",
    "phi6L",
    "phi6L_stag",
    "GL_p0",
    "GR_p0",
    "G_p0",
    "phi_f",
    "chi_f",
]
vals = {k: [] for k in keys}
nseed = 16
for sd in range(nseed):
    bs = N1Measure(m, n_noise=4, seed=sd, taste=True).bilinears(F)
    for k in keys:
        vals[k].append(np.asarray(bs[k], float).reshape(-1))
pulls = []
for k in keys:
    v = np.asarray(vals[k])
    mu, err = v.mean(0), v.std(0, ddof=1) / np.sqrt(nseed)
    ref = np.asarray(bx[k], float).reshape(-1)
    sel = err > 1e-12
    pulls += list((mu[sel] - ref[sel]) / err[sel])
pulls = np.asarray(pulls)
c.item(
    f"stochastic vs dense on {len(pulls)} quantities: largest pull", abs(pulls).max(), abs(pulls).max() <= 4.0, "{:.2f}"
)
rms = float(np.sqrt(np.mean(pulls**2)))
c.item("rms pull", rms, rms <= 1.3, "{:.2f}")
x0 = (2, 1, 3, 0)
i0 = int(np.ravel_multi_index(x0, lat.shape))
cs = N1Measure(m, exact=False, cg_tol=1e-11, taste=True).taste_correlator(F, x0=x0)
t_of = lat.coords.reshape(4, V)[-1]
eta4 = np.asarray(lat.eta[-1]).reshape(V)
e3c = 0.0
for which in ("L", "R"):
    GP = fm._project_G(G, which)
    Gf, Cf = np.zeros((4, 4)), np.zeros((4, 4))
    dt = (t_of - t_of[i0]) % 4
    wrap = np.where(t_of < t_of[i0], -1.0, 1.0)
    for a in range(4):
        np.add.at(Gf[a], dt, wrap * eta4[i0] * GP[:, a, i0, a])
        np.add.at(Cf[a], dt, (GP[:, :, i0, a] ** 2).sum(1))
    e3c = max(e3c, abs(cs[f"Gf_{which}"] - Gf).max(), abs(cs[f"Cf_{which}"] - Cf).max())
c.item("point-source taste_correlator (CG) vs dense P G P columns", e3c, e3c < 1e-8, "{:.1e}")
c.record(
    "dense: phi_R, chi_L_sum, GL_p0, GR_p0",
    [round(float(bx["phi_f"].sum()), 4), round(bx["chi_L_sum"], 4)]
    + [
        round(bx["GL_p0"], 3),
        round(bx["GR_p0"], 3),
    ],
)

# ---- (4) the C_0 measure
latp = Lattice((4, 4, 4, 4))
m0 = build_model(latp, 2.41, -0.01, 1.0, g10=0.05, g6=0.0)
F0 = np.asarray(m0.start(np.random.default_rng(3), "hot+hot"))
f0m = N1Measure(m0, exact=True)
b0 = f0m.bilinears(F0)
b0b = f0m.pair_channel(G=m0.D.dense_G(F0))
taste_keys = [k for k in b0 if k.startswith(("phi_L", "chi_L", "phi6L", "GL", "GR", "CL", "CR", "G_p0", "chi_f_sum"))]
same = all(np.array_equal(np.asarray(b0[k]), np.asarray(b0b[k])) for k in b0b)
c.item("C_0 measure: no taste keys, pair channel = the C_0 one", (len(taste_keys), same), not taste_keys and same)
c.done()
