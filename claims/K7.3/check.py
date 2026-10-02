"""K7.3: 8^4 three-start hysteresis at P_c with 2000 trajectories per start, the inner pair y = 2.368 / 2.452 at L = 8
and the pooled P_c: the pre-registered rule returns "CONTINUOUS (consistent with, L <= 8), sharpened".

masspairing.analysis.t3a.analyse_sharpened on the 39 pilot chains + the five L = 8 chains of the kappa_c line
(data/derived/k7/t3a, set "sharpened": afm / cold / hot at y = 2.41, seeds 9711 / 9712 / 9713, 2000 trajectories each;
y = 2.368 and 2.452, 1000 trajectories each). Asserts:
  (1) the three 8^4 starts are scored (tau_int window converged, tau_int < N/50, half-split < 3 sigma on m^2, sigma^2);
  (2) H1(8): all 15 afm / cold / hot pulls < 3 sigma; combined H1 (6^4 and 8^4) negative;
  (3) the four 8^4 P_c chains (the pilot chain, computed on the GPU path without the fused kernels, and the three
      starts, fused path) are consistent (< 3 sigma): no code-path discrepancy; pooled P_c at L = 6 and 8;
  (4) H2 negative on the five new chains, energy cumulant ratio + 2 sigma < 0.5; H4 continuous-consistent pooled and
      unpooled; H5 within 2 sigma; the crossing persists (|Delta(xi_2/L)| at 2.41 <= 3 sigma, falls at 2.537);
  (5) inner pair: theta_inner exists, its first-order mock is classified continuous (no power, reported only); the
      SYM-side (6,8) drift theta_outer - theta_inner is within 2 sigma of zero;
  (6) controls: injected jump flagged; mixture / AR(1) at the 8^4 tau_int; the first-order mock is flagged first order
      by H4 with the pooled errors, the smooth mock is clean;
  (7) verdict by the rule: sharpened continuous-consistent, every P_c chain scored; three sabotaged copies through the
      unchanged pipeline (ordered-start memory, two-state signal, late drift) never return it; phase labels.
"""

import os
import pathlib
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from masspairing.analysis import t3a
from masspairing.analysis.stats import jackknife
from masspairing.claimcheck import Check

c = Check("K7.3")
pilot_recs, new_recs = t3a.load_records("pilot"), t3a.load_records("sharpened")
P = t3a.analyse_pilot(pilot_recs)
S = t3a.analyse_sharpened(pilot_recs, new_recs, P)
v, ctl = S["verdict"], S["verdict"]["controls"]
f = lambda x, e, d=3: f"{x:.{d}f}({e:.{d}f})"
AFM, HOT = "F12_hyst_L8_afm/L8_y2.41_k-0.01.npz", "F12_hyst_L8_hot/L8_y2.41_k-0.01.npz"

lengths = {r["tag"]: (len(r["accepted"]), len(r["O4"])) for r in new_recs}
c.item(
    "chains: trajectories, fermion measurements (every 2nd trajectory)",
    lengths,
    all(lengths[f"F12_hyst_L8_{s}/L8_y2.41_k-0.01.npz"] == (2000, 1000) for s in ("afm", "cold", "hot"))
    and all(lengths[f"F12_L8_k-0.01_inner/L8_y{y}_k-0.01.npz"] == (1000, 500) for y in ("2.368", "2.452")),
)

# (1) scoring of the three starts
h8 = v["hyst8"]
for st in ("afm", "cold", "hot"):
    s = h8[st]
    c.item(
        f"(1) {st} (seed {s['seed']}): n after the cut, acceptance, tau_int(m^2) (< N/50),"
        " half-split m^2 / sigma^2 (sigma)",
        f"{s['n']}, {s['acc']:.2f}, {s['tau']:.1f} (< {s['n'] / 50:.0f}),"
        f" {s['half_pulls']['m2']:.1f} / {s['half_pulls']['sig2']:.1f}",
        s["scored"] and s["tau_ok"] and not s["few_samples"] and s["stationary"],
    )
afm = next(r for r in new_recs if r["tag"] == AFM)
m2 = afm["Sigma_stag_abs"] ** 2
bl = h8["afm"]["blen"]
b1 = jackknife(m2[t3a.CUT : 1000], np.mean, blen=bl)
b2 = jackknife(m2[1000:], np.mean, blen=bl)
win = [float(m2[i : i + 200].mean()) for i in range(t3a.CUT, 2000, 200)]
c.record(
    "(1) afm chain slow mode: m^2 x 1e3 over trajectories 100-1000 | 1000-2000 (sigma); 200-trajectory windows",
    f"{f(b1[0] * 1e3, b1[1] * 1e3, 2)} | {f(b2[0] * 1e3, b2[1] * 1e3, 2)}"
    f" ({abs(b1[0] - b2[0]) / np.hypot(b1[1], b2[1]):.1f});"
    f" {', '.join(f'{w * 1e3:.1f}' for w in win)}",
)

# (2) hysteresis at 8^4
pulls = v["H1_8_pulls"]
c.record("(2) pulls afm/cold/hot", ", ".join(f"{k} {x:.1f}" for k, x in pulls.items()))
for st in ("afm", "cold", "hot"):
    s = h8[st]
    c.record(
        f"(2) {st}: |Sigma_stag|, sigma^2, O4, S(pi), xi_2/L",
        f"{f(s['mabs'], s['e_mabs'], 4)}, {f(s['sig2'], s['e_sig2'], 4)}, {f(s['O4'], s['e_O4'], 4)},"
        f" {f(s['Spi'], s['e_Spi'], 1)},"
        f" {f(s['xiL'], s['e_xiL'])}",
    )
c.item(
    "(2) H1(8): largest of 15 pulls [observable]; m^2 pulls; energy-like (sigma^2, O4) largest",
    f"{v['H1_8_max']:.2f} [{max(pulls, key=pulls.get)}]; "
    + "/".join(f"{pulls[k]:.1f}" for k in ("afm-cold:m2", "afm-hot:m2", "cold-hot:m2"))
    + f"; {v['H1_8_energy_max']:.2f}",
    v["H1_8"] == "negative" and v["H1_8_max"] < 3,
)
c.item(
    "(2) combined H1 (6^4 largest pull | 8^4)",
    f"{v['H1']} ({v['H1_6_max']:.1f} | {v['H1_8_max']:.1f})",
    v["H1"] == "negative",
)

# (3) four replicas at 8^4, pooled P_c
c.item(
    "(3) four 8^4 P_c chains, 30 pulls: largest; pilot chain vs the starts largest; among the starts largest",
    f"{v['replica4_max']:.2f}; {v['old_vs_new_max']:.2f}; {v['new_vs_new_max']:.2f}",
    v["replica4_max"] < 3 and not v["old_chain_outlier"],
)
old_rec = next(r for r in pilot_recs if r["tag"] == "F_L8_k-0.01/L8_y2.41_k-0.01.npz")
c.record(
    "(3) fused GPU path: pilot chain | afm, cold, hot",
    f"{old_rec['fast']} | {[r['fast'] for r in new_recs if 'hyst' in r['tag']]}",
)
p6, p8 = v["pooled"]["L6"], v["pooled"]["L8"]
for L, p in ((6, p6), (8, p8)):
    c.item(
        f"(3) pooled P_c, L = {L}: replicas, n, largest pairwise m^2 pull; xi_2/L, S(pi), R4, |Sigma_stag|",
        f"{p['replicas']}, {p['n']}, {p['replica_pull']:.1f}; {f(p['xiL'], p['e_xiL'])}, {f(p['Spi'], p['e_Spi'], 2)},"
        f" {f(p['R4'], p['e_R4'])}, {f(p['mabs'], p['e_mabs'], 4)}",
        p["scored"] and p["replica_pull"] < 3,
    )

# (4) signals
r, er = v["H2_Ve_ratio"]
c.item(
    "(4) H2: bimodal new chains; energy-cumulant ratio (2/3 - V_e)_8/(2/3 - V_e)_6",
    f"{v['H2_bimodal_new'] or 'none'}; {f(r, er, 2)}",
    v["H2"] == "negative" and r + 2 * er < 0.5,
)
hp, hu = v["H4_pooled"], v["H4_unpooled"]
c.item(
    "(4) H4 gamma/nu_eff pooled (eta_eff; distance from 4 in sigma) | unpooled | grid maxima",
    f"{f(hp['g'], hp['eg'], 2)} ({2 - hp['g']:+.2f}; {(4 - hp['g']) / hp['eg']:.0f}) |"
    f" {f(hu['g'], hu['eg'], 2)} | {f(hp['gm'], hp['egm'], 2)}",
    v["H4"] == "continuous-consistent" and hp["g"] + 2 * hp["eg"] < 4 and hu["g"] + 2 * hu["eg"] < 4,
)
c.item(
    "(4) H5 R4(8) - R4(6) at 2.41: pooled | unpooled (within 2 sigma)",
    f"{v['H5_pooled'][0]:+.3f}({v['H5_pooled'][1]:.3f}) | {v['H5_unpooled'][0]:+.3f}({v['H5_unpooled'][1]:.3f})",
    v["H5"] == "continuous-consistent",
)
cp, cu = v["crossing"]["pooled"], v["crossing"]["unpooled"]
growth = p8["xiL"] / p6["xiL"]
e_growth = growth * np.hypot(p8["e_xiL"] / p8["xiL"], p6["e_xiL"] / p6["xiL"])
c.item(
    "(4) crossing: Delta(xi_2/L) at 2.41 pooled | unpooled (sigma, |.| <= 3); Delta at 2.537; xi_2/L growth 6 -> 8"
    " (first-order peak growth 8/6)",
    f"{cp[0]:+.1f} | {cu[0]:+.1f}; {cp[1]:+.3f}; {f(growth, e_growth, 2)} ({8 / 6:.2f})",
    v["crossing_persists"] and abs(cp[0]) <= 3 and abs(cu[0]) <= 3 and cp[1] < 0,
)

# (5) inner pair
fss = {(x["L"], round(x["y"], 3)): x for x in S["fss"]}
for y in (2.368, 2.452):
    s = fss[(8, y)]
    d = s["xiL"] - fss[(6, y)]["xiL"]
    ed = np.hypot(s["e_xiL"], fss[(6, y)]["e_xiL"])
    g = t3a.ratio_exponent(fss[(6, y)]["Spi"], fss[(6, y)]["e_Spi"], s["Spi"], s["e_Spi"])
    c.record(
        f"(5) L = 8, y = {y}: xi_2/L, S(pi), tau_int, acceptance, scored; Delta(xi_2/L) (sigma); gamma/nu_eff",
        f"{f(s['xiL'], s['e_xiL'])}, {f(s['Spi'], s['e_Spi'], 1)}, {s['tau']:.1f}, {s['acc']:.2f}, {s['scored']};"
        f" {d:+.3f}({ed:.3f}) ({d / ed:+.1f}); {f(*g, 2)}",
    )
c.item(
    "(5) inner pair at L = 8: y = 2.368 scored, y = 2.452 not scored (tau_int > N/50); xi_2/L maximum at y (L = 6, 8)",
    f"{fss[(8, 2.368)]['scored']}, {fss[(8, 2.452)]['scored']} {fss[(8, 2.452)]['flags']}; {v['xiL_max_y']}",
    fss[(8, 2.368)]["scored"] and not fss[(8, 2.452)]["scored"] and v["xiL_max_y"] == {6: 2.41, 8: 2.41},
)
th = v["H3_theta"]
fo, co = ctl["H3H4_firstorder_mock"], ctl["H3H4_continuous_mock"]
c.item(
    "(5) theta_inner SYM | SMG (unscored input); first-order mock theta at Delta = 0.042 and its class",
    f"{f(th['SYM-0.042']['theta'], th['SYM-0.042']['e_theta'], 2)} |"
    f" {f(th['SMG-0.042']['theta'], th['SMG-0.042']['e_theta'], 1)};"
    f" {fo['theta']['SYM-0.042'][:2]} {fo['inner_class']}",
    np.isfinite(th["SYM-0.042"]["theta"]) and np.isfinite(th["SMG-0.042"]["theta"]) and not v["H3_power"],
)
d, ed = v["H3_drift"]["SYM"]
c.item(
    "(5) (6,8) drift theta_outer - theta_inner: SYM (within 2 sigma of 0) | SMG (unscored input);"
    " smooth mock theta_inner",
    f"{d:+.2f}({ed:.2f}) | {v['H3_drift']['SMG'][0]:+.2f}({v['H3_drift']['SMG'][1]:.1f});"
    f" {co['theta']['SYM-0.042'][:2]}",
    abs(d) < 2 * ed,
)

# (6) controls
c.item(
    "(6) injected jump at 8^4 (afm m^2 + 6 x combined error away from cold): pull (> 4)",
    ctl["H1_injected_pull"],
    ctl["H1_injected_flagged"],
    "{:.1f}",
)
c.item(
    "(6) mixtures 5 sigma | 4 sigma detected; AR(1) false positives; at tau_int",
    f"{ctl['H2_mixture5_detected']} | {ctl['H2_mixture4_detected']}; {ctl['H2_ar1_false_positive']};"
    f" {ctl['H2_tau_used']:.1f}",
    ctl["H2_pass"],
)
c.item(
    "(6) first-order mock gamma/nu (class) | smooth mock gamma/nu (class); smooth mock theta never first order",
    f"{fo['gamma']} ({fo['gamma_class']}) | {co['gamma']} ({co['gamma_class']})",
    ctl["H4_firstorder_mock_flagged"]
    and ctl["H4_continuous_mock_clean"]
    and ctl["H3_continuous_mock_clean"]
    and ctl["exclusion_pass"],
)

# (7) verdict, sabotage, labels
c.item(
    "(7) verdict by the rule; every P_c chain scored",
    f"{v['verdict']}; {v['pc_scored']}",
    v["verdict"] == t3a.SHARPENED and v["pc_scored"],
)


def sabotaged(mutate):
    recs = [dict(r) for r in new_recs]
    mutate({r["tag"]: r for r in recs})
    return t3a.analyse_sharpened(pilot_recs, recs, P)["verdict"]


delta_a = 8 * float(np.hypot(h8["afm"]["e_m2"], h8["cold"]["e_m2"]))


def memory(R):  # ordered-start memory: afm m^2 shifted by 8 x the afm-cold combined error
    R[AFM]["Sigma_stag_abs"] = np.sqrt(R[AFM]["Sigma_stag_abs"] ** 2 + delta_a)


def telegraph(R):  # two-state signal (levels 5 sd apart, dwell ~ 60 trajectories) in the hot chain's m and sigma^2
    rng = np.random.default_rng(192)
    n = len(R[HOT]["accepted"])
    state = np.zeros(n, bool)
    s = False
    for i in range(n):
        if rng.random() < 1 / 60:
            s = not s
        state[i] = s
    for key in ("Sigma_stag_abs", "sigma2"):
        x = R[HOT][key]
        R[HOT][key] = x + 5 * x[t3a.CUT :].std() * state


def drift(R):  # late drift: the afm chain's second half shifted by its half-split + 4 x the combined half error
    x = R[AFM]["Sigma_stag_abs"]
    m2 = x[t3a.CUT :] ** 2
    h = len(m2) // 2
    a, ea = jackknife(m2[:h], np.mean, blen=bl)
    b, eb = jackknife(m2[h:], np.mean, blen=bl)
    m2n = x**2
    m2n[t3a.CUT + h :] += (a - b) + 4 * np.hypot(ea, eb)
    R[AFM]["Sigma_stag_abs"] = np.sqrt(m2n)


va = sabotaged(memory)
c.item(
    f"(7a) sabotage, ordered-start memory (afm m^2 + {delta_a:.2e}): H1(8) (largest pull) -> verdict",
    f"{va['H1_8']} ({va['H1_8_max']:.1f}) -> {va['verdict']}",
    va["H1_8"] == "positive" and va["verdict"] != t3a.SHARPENED,
)
vb = sabotaged(telegraph)
bim = [n for t, n in vb["H2_bimodal_new"] if t == HOT]
c.item(
    "(7b) sabotage, two-state signal in the hot chain: bimodal series, H1(8) (largest pull), H2 -> verdict",
    f"{bim}, {vb['H1_8']} ({vb['H1_8_max']:.1f}), {vb['H2']} -> {vb['verdict']}",
    (bim or not vb["hyst8"]["hot"]["scored"]) and vb["verdict"] != t3a.SHARPENED,
)
vc = sabotaged(drift)
hc = vc["hyst8"]["afm"]
c.item(
    "(7c) sabotage, late drift in the afm chain: half-split (sigma), flags, P_c scored -> verdict",
    f"{hc['half_pulls']['m2']:.1f}, {hc['flags']}, {vc['pc_scored']} -> {vc['verdict']}",
    not hc["scored"] and "non-stationary" in hc["flags"] and not vc["pc_scored"] and vc["verdict"] == "UNDECIDED",
)
labels = S["labels"]
c.item(
    "(7) phase labels of the five chains (y, start: phase, trajectories)",
    [(x["y"], x["start"], x["phase"], x["ntraj"]) for x in labels],
    len(labels) == 5 and all(x["ntraj"] == (2000 if "hyst" in x["file"] else 1000) for x in labels),
)
c.done()
