"""K7.1: no first-order signature of the transition at P_c = (2.41, -0.01) at L <= 8 (pilot set, 39 chains).

Analysis of masspairing.analysis.t3a.analyse_pilot on the 39 chains of data/derived/k7/t3a (set "pilot"): the kappa =
-0.01 line at seven y (second seeds at 2.368, 2.41, 2.452 on 6^4), the kappa = -0.04 and +0.02 lines, the afm / cold /
hot start set at P_c on 6^4. First 100 trajectories of every chain cut; blocked jackknife (block = 2 tau_int(m^2)).
  (1) H1: every afm / cold / hot pull on {m^2, sigma^2, O4, S(pi), S(pi + p_min)} < 3 sigma, all three chains scored;
  (2) H2: no chain bimodal in m, sigma^2 or O4 (Scott and 0.75 x Scott); (2/3 - V_e)_8 / (2/3 - V_e)_6 + 2 sigma < 0.5;
  (3) H4: gamma/nu_eff from S(pi) at y = 2.41 and from the grid maxima, + 2 sigma < 4; H5: R4(8) - R4(6) within 2 sigma;
  (4) the crossing: |Delta(xi_2/L)| at 2.41 <= 3 sigma, xi_2/L falls with L at every other y of the kappa_c line;
      the slope exponent theta is reported (its first-order control is classified continuous: no power on this grid);
  (5) controls: injected jump flagged; 5-sigma mixture detected >= 38/40 with <= 2/40 AR(1) false positives; the
      first-order mock is never called continuous by H4; the smooth mock is continuous-consistent, never first order;
  (6) verdict "CONTINUOUS (consistent with, L <= 8)" (the same by the letter of the rule with H3 kept);
  (7) the kappa = +0.02 and -0.04 lines; the unscored chains; a phase label for every chain.
"""

import os
import pathlib
import sys
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from masspairing.analysis import t3a
from masspairing.claimcheck import Check

c = Check("K7.1")
recs = t3a.load_records("pilot")
P = t3a.analyse_pilot(recs)
v, comb, ctl = P["verdict"], P["comb"], P["verdict"]["controls"]
c.item("chains in the pilot set", len(recs), len(recs) == 39)
f = lambda x, e, d=3: f"{x:.{d}f}({e:.{d}f})"
for k in sorted(comb):
    s = comb[k]
    c.record(
        f"kappa = {k[0]:+.2f}, L = {k[1]}, y = {k[2]}: n, tau, |Sigma_stag|, R4, xi_2/L, S(pi), sigma^2, O4, flags",
        f"{s['n']}, {s['tau']:.1f}, {f(s['mabs'], s['e_mabs'], 4)}, {f(s['R4'], s['e_R4'])}, {f(s['xiL'], s['e_xiL'])},"
        f" {f(s['Spi'], s['e_Spi'], 2)}, {f(s['sig2'], s['e_sig2'], 4)}, {f(s['O4'], s['e_O4'], 4)},"
        f" {'' if s['scored'] else 'not scored '}{','.join(s['flags'])}"
        + (f" ({s['replicas']} replicas, pull {s['replica_pull']:.1f})" if s["replicas"] > 1 else ""),
    )

# (1) hysteresis at P_c on 6^4
h = v["hyst"]
pulls = v["H1_pulls"]
c.record("(1) pulls afm/cold/hot", ", ".join(f"{k} {x:.1f}" for k, x in pulls.items()))
energy = max(x for k, x in pulls.items() if k.endswith(":sig2") or k.endswith(":O4"))
c.item(
    "(1) H1: largest of 15 pulls (< 3) [observable]; energy-like (sigma^2, O4) largest",
    f"{v['H1_max']:.2f} [{max(pulls, key=pulls.get)}]; {energy:.2f}",
    v["H1"] == "negative" and v["H1_max"] < 3,
)
c.item(
    "(1) tau_int(m^2) of the afm / cold / hot chains; all scored (stationary, tau_int < N/50)",
    ", ".join(f"{h[s]['tau']:.1f}" for s in ("afm", "cold", "hot")),
    all(h[s]["scored"] and h[s]["stationary"] for s in ("afm", "cold", "hot")),
)

# (2) double peaks, energy cumulant
c.item(
    "(2) H2: chains bimodal in m, sigma^2 or O4 (117 series)",
    v["H2_bimodal"] or "none",
    v["H2"] == "negative" and not v["H2_bimodal"],
)
r, er = v["H2_Ve_ratio"]
ve = v["lines"][-0.01]["Ve_min"]
c.item(
    "(2) max_y (2/3 - V_e(sigma^2)): L = 6 [y] | L = 8 [y] | ratio (+2 sigma < 0.5; V^-1 law 0.32)",
    f"{ve[6][0]:.3e}({ve[6][1]:.1e}) [{ve[6][2]}] | {ve[8][0]:.3e}({ve[8][1]:.1e}) [{ve[8][2]}] | {r:.2f}({er:.2f})",
    r + 2 * er < 0.5,
)

# (3) susceptibility exponent, Binder ratio
P6 = {k[2]: x for k, x in comb.items() if k[0] == -0.01 and k[1] == 6}
P8 = {k[2]: x for k, x in comb.items() if k[0] == -0.01 and k[1] == 8}
g, eg = v["H4_gamma_nu"]
gm, egm = v["H4_gridmax"]
c.item(
    "(3) S(pi) at y = 2.41: L = 6 -> L = 8; gamma/nu_eff; eta_eff; distance from 4 (sigma)",
    f"{f(P6[2.41]['Spi'], P6[2.41]['e_Spi'], 1)} -> {f(P8[2.41]['Spi'], P8[2.41]['e_Spi'], 1)}; {f(g, eg, 2)};"
    f" {2 - g:+.2f}({eg:.2f}); {(4 - g) / eg:.1f}",
    v["H4"] == "continuous-consistent" and g + 2 * eg < 4,
)
c.item("(3) gamma/nu_eff from the grid maxima of S(pi) (+2 sigma < 4)", f(gm, egm, 2), gm + 2 * egm < 4)
d, ed = v["H5_dR4"]
c.item(
    "(3) R4 at y = 2.41: L = 6 -> L = 8; difference within 2 sigma",
    f"{f(P6[2.41]['R4'], P6[2.41]['e_R4'])} -> {f(P8[2.41]['R4'], P8[2.41]['e_R4'])}; {d:+.3f}({ed:.3f})",
    abs(d) < 2 * ed,
)

# (4) the crossing and the slope exponent
diff = v["lines"][-0.01]["diff"]
c.item(
    "(4) xi_2/L at y = 2.41: L = 6 | L = 8 | difference (sigma, |.| <= 3)",
    f"{f(P6[2.41]['xiL'], P6[2.41]['e_xiL'])} | {f(P8[2.41]['xiL'], P8[2.41]['e_xiL'])} | {diff[2.41]['sig']:+.1f}",
    abs(diff[2.41]["sig"]) <= 3,
)
others = {y: diff[y]["sig"] for y in diff if y != 2.41}
c.item(
    "(4) Delta(xi_2/L) = (L = 8) - (L = 6) at the other y of the kappa_c line (sigma; all negative)",
    ", ".join(f"{y}: {x:+.1f}" for y, x in others.items()),
    all(x < 0 for x in others.values()) and v["crossing_persists"],
)
th = v["H3_theta"]
c.record(
    "(4) slope exponent theta between 2.41 and 2.41 -/+ 0.127: SYM side | SMG side (1/nu_eff for a cusp, 2/nu_eff for a"
    " smooth top)",
    f"{f(th['SYM-0.127']['theta'], th['SYM-0.127']['e_theta'], 2)} |"
    f" {f(th['SMG-0.127']['theta'], th['SMG-0.127']['e_theta'], 2)}",
)
c.item(
    "(4) no L = 8 chain at y = 2.368 / 2.452 on the kappa_c line in the pilot set (inner-pair theta not formed)",
    [k for k, t in th.items() if t.get("missing")],
    th["SYM-0.042"].get("missing") and th["SMG-0.042"].get("missing"),
)
fo, co = ctl["H3H4_firstorder_mock"], ctl["H3H4_continuous_mock"]
c.item(
    "(4) first-order mock: theta (SYM, SMG) and its class -> theta has no power, H3 reported only",
    f"{fo['theta']} {fo['theta_class']}",
    not ctl["H3_has_power"],
)

# (5) controls
c.item(
    "(5) injected jump (afm m^2 + 6 x combined error away from cold): pull (> 4)",
    ctl["H1_injected_pull"],
    ctl["H1_injected_flagged"],
    "{:.1f}",
)
c.item(
    "(5) mixtures 5 sigma | 4 sigma detected; AR(1) false positives (tau of the 6^4 P_c point)",
    f"{ctl['H2_mixture5_detected']} | {ctl['H2_mixture4_detected']}; {ctl['H2_ar1_false_positive']}",
    ctl["H2_pass"],
)
c.item(
    "(5) first-order mock gamma/nu (class, never continuous) | smooth mock gamma/nu (class) and theta classes",
    f"{fo['gamma']} ({fo['gamma_class']}) | {co['gamma']} ({co['gamma_class']}), {co['theta_class']}",
    ctl["H4_firstorder_mock_not_continuous"] and ctl["H4_continuous_mock_clean"] and ctl["H3_continuous_mock_clean"],
)

# (6) verdict
c.item(
    "(6) verdict by the rule [by the letter with H3 kept]",
    f"{v['verdict']} [{v['verdict_prereg_letter']}]",
    v["verdict"] == t3a.CONTINUOUS and v["verdict_prereg_letter"] == "CONTINUOUS" and v["pc_scored"],
)

# (7) the other kappa lines, unscored chains, phase labels
dp = v["lines"][0.02]["diff"]
Pp6 = {k[2]: x for k, x in comb.items() if k[0] == 0.02 and k[1] == 6}
Pp8 = {k[2]: x for k, x in comb.items() if k[0] == 0.02 and k[1] == 8}
c.item(
    "(7) kappa = +0.02, y = 2.452: |Sigma_stag| L = 6 | L = 8 (ratio ~ 1: ordered); gamma/nu_eff; R4 at L = 8",
    f"{f(Pp6[2.452]['mabs'], Pp6[2.452]['e_mabs'])} | {f(Pp8[2.452]['mabs'], Pp8[2.452]['e_mabs'])}"
    f" ({dp[2.452]['ratio_mabs']:.2f});"
    f" {f(dp[2.452]['gamma_nu'], dp[2.452]['e_gamma_nu'], 2)}; {Pp8[2.452]['R4']:.2f}",
    abs(dp[2.452]["ratio_mabs"] - 1) < 0.15 and Pp8[2.452]["mabs"] >= 0.15,
)
c.item(
    "(7) kappa = +0.02, y = 2.368, 2.41: |Sigma_stag|_6/|Sigma_stag|_8 and Delta(xi_2/L) (sigma; rising with L)",
    ", ".join(f"{y}: {dp[y]['ratio_mabs']:.2f}, {dp[y]['sig']:+.1f}" for y in (2.368, 2.41)),
    all(1.25 < dp[y]["ratio_mabs"] < 1.5 and dp[y]["dxiL"] > 0 for y in (2.368, 2.41)),
)
dm = v["lines"][-0.04]["diff"]
Pm6 = {k[2]: x for k, x in comb.items() if k[0] == -0.04 and k[1] == 6}
c.item(
    "(7) kappa = -0.04: Delta(xi_2/L) (sigma, all negative), |Sigma_stag|_6/|Sigma_stag|_8, gamma/nu_eff per y",
    "; ".join(
        f"{y}: {x['sig']:+.1f}, {x['ratio_mabs']:.2f}, {x['gamma_nu']:+.2f}({x['e_gamma_nu']:.2f})"
        for y, x in dm.items()
    ),
    all(
        x["dxiL"] < 0 and 1.7 < x["ratio_mabs"] < 1.8 and abs(x["gamma_nu"]) < 2 * x["e_gamma_nu"] + 0.1
        for x in dm.values()
    ),
)
pc6, pc8 = comb[(-0.01, 6, 2.41)], comb[(-0.01, 8, 2.41)]
c.record(
    "(7) kappa = -0.01, y = 2.41 (P_c): |Sigma_stag|_6/|Sigma_stag|_8 (ordered ~ 1; fluctuation sqrt(V_8/V_6) = 1.78)",
    f"{pc6['mabs'] / pc8['mabs']:.2f}",
)
c.item(
    "(7) kappa = -0.04, L = 6: O4 at y = 2.028 -> 2.452 (rises into the SMG range)",
    f"{Pm6[2.028]['O4']:.4f} -> {Pm6[2.452]['O4']:.4f}",
    Pm6[2.028]["O4"] < 0.02 and Pm6[2.452]["O4"] > 0.08,
)
unscored = sorted((s["kappa"], s["L"], s["y"], ",".join(s["flags"])) for s in P["stats"] if not s["scored"])
c.item(
    "(7) unscored chains (kappa, L, y, flags)",
    unscored,
    [(k, L, y) for k, L, y, _ in unscored]
    == [(-0.04, 6, 2.283), (-0.04, 6, 2.368), (-0.04, 6, 2.41), (0.02, 6, 2.452)],
)
labels = P["labels"]
count = Counter((x["kappa"], x["L"], x["phase"]) for x in labels)
c.item(
    "(7) phase labels (kappa, L, phase): count",
    dict(sorted(count.items())),
    len(labels) == 39 and all(x["phase"] in ("SYM", "AFM", "SMG", "critical") for x in labels),
)
c.record("(7) chains labelled critical", sum(x["phase"] == "critical" for x in labels))
assert np.isfinite(g)
c.done()
