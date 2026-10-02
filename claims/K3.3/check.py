"""K3.3: the response of the complete chi10 to the pure plaquette term is flat at P_c: R(8)/R(6) = 1.02(5), 0.97(2).

R(L) = [chi10(wedge, g) - chi10(completion, g)]/g at p = 0, g = 0.01, 0.02; the wedge action is the per-link
completion action minus g T_Q exactly (I.4), so R is the finite-g response to the pure (10,3,1) + two-site-6 term.
L = 4, 6: exact-propagator chains; L = 8: dense measurements every 10th trajectory of two continuation streams per
chain (o: original seed from the copy point NCOPY; r: re-seeded, first 50 trajectories after NCOPY dropped).

Criterion (fixed before the numbers were read), q = R(8)/R(6): relevant iff R(8) > 2 sigma and q - 1 > 2 sigma_q;
irrelevant iff q <= 1 + 2 sigma_q and q + 2 sigma_q < (V8/V6)^0.5 = 1.78; undecided otherwise.
Items: (1) equilibration, acceptance, 30 + 20 measurements per chain, o/r streams consistent; (2) verdict per g;
(3) power: injected growth R(8) -> 2 R(6) flagged relevant, R(8) -> 1.78 R(6) never called irrelevant;
(4) dense 8^4 chi10 = the stochastic (n_noise 8) estimate on the same chains before the copy point within 3 sigma,
and < 0.6 x free; (5) forced blocks of 2 and 5 measurements leave the verdicts unchanged.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from masspairing.analysis import k3
from masspairing.analysis.stats import jackknife
from masspairing.claimcheck import Check

GS = (0.01, 0.02)
NCOPY = {("compl", 0.01): 1268, ("compl", 0.02): 1071, ("wedge", 0.01): 1500, ("wedge", 0.02): 1500}
K0_STOCH = {("compl", 0.01): 300, ("compl", 0.02): 200, ("wedge", 0.01): 400, ("wedge", 0.02): 300}
K0_STOCH_REP = {0.01: 1100, 0.02: 900}  # re-seeded stochastic replicas of the completion chains
RCUT = 50
V6, V8 = 6**4, 8**4
FREE8 = k3.FREE_CHI10[8][0]

c = Check("K3.3")


def dense_streams(m, g):
    out = []
    for rep, k0 in (("", NCOPY[(m, g)]), ("_rA", NCOPY[(m, g)] + RCUT)):
        ((s, kk),), metas = k3.streams([(k3.p3_rel(m, g, 8, rep), k0)])
        meta = metas[0]
        assert meta["exact_ferm"] and meta["ferm_every"] == 10 and meta["g10"] == g
        assert (meta.get("quads") == "links") == (m == "compl")
        out.append(dict(rep=rep or "_o", k0=k0, nt=meta["n_stored"], s=k3.cut(s, kk), V=int(np.prod(meta["shape"]))))
    return out


D, ok1 = {}, True
for m in ("compl", "wedge"):
    for g in GS:
        st = dense_streams(m, g)
        chis = [np.asarray(d["s"]["ts_chi10"], float) for d in st]
        ctrl = [d["V"] * np.asarray(d["s"]["ts_phi_stag_sq"], float) for d in st]
        for d, x in zip(st, chis, strict=True):
            acc = float(np.mean(d["s"]["ts_accepted"]))
            fz = float(np.mean(d["s"]["ts_s2_0"][:50])) / ((12 if m == "compl" else 2) * g)
            mu, e, t, _ = k3.stat(x)
            ok1 &= fz >= 0.8 and acc >= 0.4 and len(x) in (30, 20)
            c.record(
                f"{m} g={g:g} {d['rep']}: n, chi10, tau_int, acceptance, freeze",
                f"{len(x)} meas (traj {d['k0']}..{d['nt']}) {mu:.4f}({e:.4f}) {t:.2f} {acc:.2f} {fz:.2f}",
            )
        mu, e, t, per, chi2 = k3.combine_streams(chis)
        pull = (per[0][0] - per[1][0]) / np.hypot(per[0][1], per[1][1])
        ok1 &= abs(pull) < 3
        cm, ce, *_ = k3.combine_streams(ctrl)
        D[(m, g)] = dict(chi10=(mu, e), ctrl=(cm, ce), parts=chis, pull=pull)
        c.record(f"{m} g={g:g} combined: chi10, stream pull", f"{mu:.4f}({e:.4f}), {pull:+.2f} sigma")
c.item("(1) freeze >= 0.8, acceptance >= 0.4, 30 + 20 dense measurements per chain, o/r pulls < 3 sigma", ok1, ok1)

R = {}
for L in (4, 6):
    for g in GS:
        w = k3.ensemble_of([(k3.p3_rel("wedge", g, L), 450 if (L, g) == (6, 0.01) else 0)])
        cc = k3.ensemble_of([(k3.p3_rel("compl", g, L), 0)])
        R[(L, g)] = (
            (w["chi10"]["mean"] - cc["chi10"]["mean"]) / g,
            np.hypot(w["chi10"]["err"], cc["chi10"]["err"]) / g,
        )
        a, b = w["V_phi_stag_sq"], cc["V_phi_stag_sq"]
        R[(L, g, "ctrl")] = ((a["mean"] - b["mean"]) / g, np.hypot(a["err"], b["err"]) / g)
for g in GS:
    (a, ea), (b, eb) = D[("wedge", g)]["chi10"], D[("compl", g)]["chi10"]
    R[(8, g)] = ((a - b) / g, np.hypot(ea, eb) / g)
    (a, ea), (b, eb) = D[("wedge", g)]["ctrl"], D[("compl", g)]["ctrl"]
    R[(8, g, "ctrl")] = ((a - b) / g, np.hypot(ea, eb) / g)

lnV = np.log(V8 / V6)
for g in GS:
    q, sq = k3.ratio(R[(8, g)], R[(6, g)])
    q46, sq46 = k3.ratio(R[(6, g)], R[(4, g)])
    v = k3.response_verdict(R[(8, g)], R[(6, g)])
    c.item(
        f"(2) g = {g:g}: R(4), R(6), R(8), R(8)/R(6), d ln R/d ln V (6,8), [R(6)/R(4)] -> verdict",
        f"{R[(4, g)][0]:.2f}({R[(4, g)][1]:.2f}) {R[(6, g)][0]:.2f}({R[(6, g)][1]:.2f}) {R[(8, g)][0]:.2f}"
        f"({R[(8, g)][1]:.2f}); {q:.3f}({sq:.3f}); {np.log(q) / lnV:+.3f}({sq / q / lnV:.3f}); [{q46:.2f}({sq46:.2f})]"
        f" -> {v}",
        v == "irrelevant",
    )
    qc, sqc = k3.ratio(R[(8, g, "ctrl")], R[(6, g, "ctrl")])
    qc46, sqc46 = k3.ratio(R[(6, g, "ctrl")], R[(4, g, "ctrl")])
    c.record(
        f"    eps-channel control V<|phi_stag|^2> response g = {g:g}: R(4), R(6), R(8), R(8)/R(6), R(6)/R(4)",
        f"{R[(4, g, 'ctrl')][0]:.0f}({R[(4, g, 'ctrl')][1]:.0f}) {R[(6, g, 'ctrl')][0]:.0f}"
        f"({R[(6, g, 'ctrl')][1]:.0f}) {R[(8, g, 'ctrl')][0]:.0f}({R[(8, g, 'ctrl')][1]:.0f}); {qc:.2f}({sqc:.2f}),"
        f" {qc46:.2f}({sqc46:.2f})",
    )
growth = [
    (abs(R[(6, g, "ctrl")][0]) - abs(R[(4, g, "ctrl")][0])) / np.hypot(R[(6, g, "ctrl")][1], R[(4, g, "ctrl")][1])
    for g in GS
]
c.item(
    "control: |R_eps(6)| - |R_eps(4)| in units of its error at g = 0.01, 0.02 (> 2)",
    growth,
    min(growth) > 2,
    fmt="{0[0]:.1f}, {0[1]:.1f}",
)

# (3) power: inject a growth into the dense wedge chi10 at 8^4
ok3 = True
for factor, need in ((2.0, ("relevant",)), ((V8 / V6) ** 0.5, ("relevant", "undecided"))):
    for g in GS:
        (a, ea), (b, eb) = D[("wedge", g)]["chi10"], D[("compl", g)]["chi10"]
        shift = factor * R[(6, g)][0] * g - (a - b)
        R8 = ((a + shift - b) / g, np.hypot(ea, eb) / g)
        v = k3.response_verdict(R8, R[(6, g)])
        ok3 &= v in need
        c.record(
            f"    injected R(8) -> {factor:.2f} R(6) at g = {g:g} (chi10 shift {shift:+.4f})",
            f"R(8)/R(6) = {k3.ratio(R8, R[(6, g)])[0]:.2f} -> {v}",
        )
c.item("(3) a 2x growth is flagged relevant and an exponent-0.5 growth is never called irrelevant", ok3, ok3)

# (4) dense vs stochastic before the copy point (o stream from its thermalisation cut; replicas of the completion)
ok4 = True
for (m, g), d in D.items():
    o_rel = k3.p3_rel(m, g, 8)
    spec = [(o_rel, K0_STOCH[(m, g)])]
    if m == "compl":
        spec += [(o_rel.replace("F8_P3x_compl", f"F8_P3_compl_{r}"), K0_STOCH_REP[g]) for r in ("rA", "rB")]
    st, metas = k3.streams(spec)
    parts = [k3.window(st[0][0], st[0][1], NCOPY[(m, g)] - metas[0]["k0_stored"])] + [k3.cut(s, k) for s, k in st[1:]]
    stoch_mean = float(np.mean(parts[0]["ts_chi10"]))  # the original stream
    stoch_err = k3.analyse(k3.concat(parts), metas[0])["chi10"]["err"]  # all stochastic streams of the chain
    mu, e = d["chi10"]
    pull = (mu - stoch_mean) / np.hypot(e, stoch_err)
    ok4 &= abs(pull) < 3 and mu + 2 * e < 0.6 * FREE8
    c.record(
        f"    {m} g={g:g}: dense chi10, stochastic (n_noise 8), pull, error ratio, dense/free",
        f"{mu:.4f}({e:.4f}) {stoch_mean:.4f}({stoch_err:.4f}) {pull:+.2f} {stoch_err / e:.1f} {mu / FREE8:.3f}",
    )
c.item("(4) dense = stochastic within 3 sigma on every chain; chi10(P_c, 8^4) + 2 sigma < 0.6 x free", ok4, ok4)

# (5) forced block lengths of 2 and 5 dense measurements
ok5 = True
for bl in (2, 5):
    for g in GS:
        e = {}
        for m in ("compl", "wedge"):
            ev = [jackknife(p, np.mean, blen=bl)[1] for p in D[(m, g)]["parts"]]
            e[m] = float(np.sum(1 / np.asarray(ev) ** 2) ** -0.5)
        R8 = (R[(8, g)][0], np.hypot(e["wedge"], e["compl"]) / g)
        v = k3.response_verdict(R8, R[(6, g)])
        q, sq = k3.ratio(R8, R[(6, g)])
        ok5 &= v == "irrelevant"
        c.record(
            f"    block {bl} measurements, g = {g:g}", f"R(8) {R8[0]:.2f}({R8[1]:.2f}), q = {q:.3f}({sq:.3f}) -> {v}"
        )
c.item("(5) verdicts unchanged with blocks of 2 and 5 measurements", ok5, ok5)
c.done()
