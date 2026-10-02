#!/usr/bin/env python3
"""Freeze notebook-side numbers for the K3 / I.4-I.7 equivalence tests (needs a notebook checkout and the archive).

    MASSPAIRING_NOTEBOOK=/path/to/notebook MASSPAIRING_ARCHIVE=/path/to/archive \\
        $MASSPAIRING_NOTEBOOK/.venv/bin/python tests/equivalence/make_k3_reference.py

Writes tests/reference/k3/:
* analysis.json -- the notebook's ensemble analysis (thermalisation cuts, replica concatenation, blocked jackknife) of
  every ensemble of tests/equivalence/k3_cases.py, read from the raw archive chains;
* exact.json -- the notebook's exact 2^3 Hubbard-Stratonovich numbers and its stored all-orders bag references;
* chains.npz -- short prefixes of the notebook's exact-determinant Metropolis and RHMC runs (same seeds as the claims),
  measured with the notebook's complete-channel measure;
* channels.json -- the notebook's dense real-flavour reference (corner structure factors, local energies; boundary
  signs off) on a stored 4^4 configuration.
"""

import json
import os
import pathlib
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = pathlib.Path(__file__).resolve().parent
NB = pathlib.Path(os.environ["MASSPAIRING_NOTEBOOK"])
ARCH = pathlib.Path(os.environ["MASSPAIRING_ARCHIVE"])
OUT = HERE.parent / "reference" / "k3"
sys.path.insert(0, str(NB / "src"))
sys.path.insert(0, str(NB / "scripts"))
sys.path.insert(0, str(NB / "claims" / "C070_wedge_hmc_exact"))
sys.path.insert(0, str(next((NB / "claims").glob("C103_*"))))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from k3_cases import KEYS, ensembles, stochastic8  # noqa: E402


def analysis():
    from laneP_ensemble import ensemble, stream_means

    out = {}
    cases = dict(ensembles())
    cases.update({k: v["notebook"] for k, v in stochastic8().items()})
    for name, spec in cases.items():
        r = ensemble([(ARCH / rel, k0) for rel, k0 in spec], name.replace(" ", "").replace(",", "_"))
        row = dict(acceptance=r["acceptance"], ntraj=r["ntraj"], blen=r["blen"])
        for k in KEYS:
            if k in r and isinstance(r[k], dict):
                row[k] = [r[k]["mean"], r[k]["err"]] + ([r[k]["corner"]] if "corner" in r[k] else [])
        if name.startswith("stoch8"):
            row["o_stream_mean_chi10"] = stream_means([(ARCH / spec[0][0], spec[0][1])], "chi10")[0]
        out[name] = row
    return out


def exact():
    from exact_hs import gaussian_second_moment, hs_exact, pf_polynomial, poly_mul
    from unhiggsed.hmc.wedge import WedgeGeometry, WedgeLattice, build_kinetic_d

    lat3 = WedgeLattice((2, 2, 2), bc=(-1, -1, -1))
    K03 = build_kinetic_d(lat3).toarray()
    out = {"bags": json.loads((NB / "claims" / "C070_wedge_hmc_exact" / "reference_2x2x2.json").read_text())}
    for name, (g10, g6) in (("interior", (0.4, 0.1)), ("edge+", (0.5, 0.5))):
        r = hs_exact(WedgeGeometry(lat3, "all", g10, g6), K03)
        out[name] = dict(Z_over_Z0=r["Z_over_Z0"], sum_s2=r["sum_s2"], sum_term=r["sum_term"], s1=float(r["s1"].sum()))
    g = WedgeGeometry(lat3, "links", 0.3, 0.0)
    P2 = poly_mul(*(2 * [pf_polynomial(g, K03)]))
    out["links"] = dict(Z_over_Z0=gaussian_second_moment(P2, np.sqrt(1.0 / (2.0 * g.inv4g))))
    return out


def chains():
    from unhiggsed.hmc.observables import scalar_observables
    from unhiggsed.hmc.wedge import WedgeHMC, WedgeLattice, WedgeMeasure, WedgeModel, link_field_observables

    def collect(series, lat, mm, fm, F):
        so = scalar_observables(lat, mm.split(F)[0])
        fo = fm.bilinears(F)
        lo = link_field_observables(mm, F)
        row = dict(sigma2=so["sigma2"], Sigma_stag_abs=so["Sigma_stag_abs"], s2=lo["s2"], E_bond=fo["E_bond"].mean())
        for k in ("O4", "phi_stag_sq", "dimer_sq_par", "chi10", "chi6", "E10", "E6", "phi_src", "phi_src_dh"):
            row[k] = fo[k]
        for k, v in row.items():
            series.setdefault(k, []).append(float(v))

    def metropolis(lat, mk, nsweeps, seed, delta_s, measure_from):
        """The notebook claims' Metropolis loop (C070, C111, C112), measured with the complete-channel measure."""
        Y2 = 2.0
        rng = np.random.default_rng(seed)
        mm = mk()
        g = mm.geom
        V = lat.V
        fm = WedgeMeasure(mm, exact=True, chi10="full")
        F = mm.start(rng, "hot")
        D = mm.D.dense(F)
        ld = np.linalg.slogdet(D)[1]
        sb = mm.s_boson(F)
        series = {}
        I = g.I.tocsr()
        touch = []
        for f in range(g.n_fields):
            ent = []
            for link, c_ in zip(
                I.indices[I.indptr[f] : I.indptr[f + 1]], I.data[I.indptr[f] : I.indptr[f + 1]], strict=True
            ):
                u, v = int(g.lu[link]), int(g.lv[link])
                for c in (0, 1):
                    ent.append((2 * u + c, 2 * v + c, -c_))
                    ent.append((2 * v + c, 2 * u + c, c_))
            touch.append(ent)
        yblock = lambda s3: 1j * Y2 * np.array([[s3[2], s3[0] - 1j * s3[1]], [s3[0] + 1j * s3[1], -s3[2]]])
        for sweep in range(nsweeps):
            for i in rng.permutation(V + g.n_fields):
                Fp = F.copy()
                Dp = D.copy()
                if i < V:
                    x = int(i)
                    Fp[[x, V + x, 2 * V + x]] += 0.6 * rng.normal(size=3)
                    Dp[2 * x : 2 * x + 2, 2 * x : 2 * x + 2] = yblock(Fp[[x, V + x, 2 * V + x]])
                else:
                    f = int(i - V)
                    ds = delta_s * rng.normal()
                    Fp[mm.nsig + f] += ds
                    for a, b, c in touch[f]:
                        Dp[a, b] += c * ds
                ldp = np.linalg.slogdet(Dp)[1]
                sbp = mm.s_boson(Fp)
                if np.log(rng.random()) < (ldp - ld) - (sbp - sb):
                    F, D, ld, sb = Fp, Dp, ldp, sbp
            if sweep >= measure_from and sweep % 5 == 0:
                collect(series, lat, mm, fm, F)
        return series

    def rhmc(lat, mk, ntraj, seed, nsteps, jitter):
        mm = mk()
        hh = WedgeHMC(mm, tau=1.0, nsteps=nsteps, seed=seed)
        jr = np.random.default_rng(seed + 424242)
        fm = WedgeMeasure(mm, exact=True, chi10="full")
        F = mm.start(hh.rng, "hot")
        series = {}
        for it in range(200 + ntraj):
            if jitter:
                hh.tau = 1.0 * (1.0 + jitter * (2.0 * jr.random() - 1.0))
            F, info = hh.trajectory(F)
            if it < 200:
                continue
            series.setdefault("accepted", []).append(float(info["accepted"]))
            collect(series, lat, mm, fm, F)
        return series

    l2 = WedgeLattice((2, 2, 2, 2), bc=(-1,) * 4)
    l24 = WedgeLattice((2, 2, 2, 4), bc=(-1,) * 4)
    wedge = lambda **kw: WedgeModel(l2, 2.0, 0.0, 1.0, g10=0.4, g6=0.1, **kw)
    links = lambda **kw: WedgeModel(l2, 2.0, 0.0, 1.0, g10=0.15, quads="links", **kw)
    src = lambda **kw: WedgeModel(l24, 2.0, 0.0, 1.0, g10=0.4, g6=0.1, h10=0.1, **kw)
    runs = {
        "wedge_2x4_metropolis": metropolis(l2, wedge, 100, 76, 0.7, 0),
        "wedge_2x4_rhmc": rhmc(l2, lambda: wedge(cg_tol=1e-10), 10, 77, 6, 0.0),
        "links_2x4_metropolis": metropolis(l2, links, 100, 112, 0.9, 0),
        "links_2x4_rhmc": rhmc(l2, lambda: links(cg_tol=1e-10), 10, 113, 8, 0.3),
        "source_metropolis_h0.1": metropolis(l24, src, 260, 1122, 0.7, 200),
        "source_rhmc_h0.1": rhmc(l24, lambda: src(cg_tol=1e-10), 10, 1124, 8, 0.3),
    }
    # HMC on 2^3 (C070 item 3)
    lat3 = WedgeLattice((2, 2, 2), bc=(-1, -1, -1))
    m3 = WedgeModel(lat3, 0.0, 0.0, 1.0, g10=0.4, g6=0.1, quads="all", cg_tol=1e-10)
    h3 = WedgeHMC(m3, tau=1.0, nsteps=5, seed=73)
    fm3 = WedgeMeasure(m3, exact=True)
    g = m3.geom
    gf = np.asarray([g.g[j] for j in g.f_pair])
    F = m3.start(h3.rng, "hot")
    ser = {k: [] for k in ("s2", "term", "s1", "bond")}
    for it in range(300 + 20):
        F, info = h3.trajectory(F)
        if it < 300:
            continue
        s = m3.split(F)[1]
        ser["s2"].append(float((s**2).sum()))
        ser["term"].append(float(((s**2 - 2 * gf) / (4 * gf)).sum()))
        ser["s1"].append(float(s.sum()))
        S = fm3._dense_S(F)
        Kl = g.lsign * 2.0 * np.einsum("nii->n", S[g.lu, :, g.lv, :]).real
        ser["bond"].append(
            float(
                sum(
                    2 * g.g[g.f_pair[f]] * Kl[g.qlinks[g.f_quad[f], 2 * g.f_pair[f] : 2 * g.f_pair[f] + 2]].sum()
                    for f in range(g.n_fields)
                )
            )
        )
    runs["wedge_2x3_hmc"] = ser
    return {f"{n}/{k}": np.asarray(v) for n, s in runs.items() for k, v in s.items()}


def channels():
    from chi10_full import flavour_propagator, local_energies, structure_factors
    from unhiggsed.hmc.wedge import WedgeLattice, WedgeMeasure, WedgeModel

    d = np.load(ARCH / "results/laneW2/c072/L4_y2.41_k-0.01_g0.1_g60.npz", allow_pickle=True)
    m = json.loads(str(d["meta"]))
    lat = WedgeLattice(tuple(m["shape"]), bc=tuple(m["bc"]))
    model = WedgeModel(lat, m["y"], m["kappa"], m["lam"], g10=m["g10"], g6=m["g6"], quads=m["quads"])
    F = d["fields_final"]
    S = np.linalg.inv(model.D.dense(F)).reshape(lat.V, 2, lat.V, 2)
    Gf = flavour_propagator(S)
    meas = WedgeMeasure(model, exact=True, chi10="full", bc_signs=False)
    sf, loc = structure_factors(meas, Gf), local_energies(meas, Gf)
    return dict(chi10_corners=list(sf["phi"]), chi6_corners=list(sf["lam"]), E10=loc["phi"], E6=loc["lam"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:]) or {"analysis", "exact", "chains", "channels"}
    if "analysis" in only:
        (OUT / "analysis.json").write_text(json.dumps(analysis(), indent=1) + "\n")
    if "exact" in only:
        (OUT / "exact.json").write_text(json.dumps(exact(), indent=1) + "\n")
    if "channels" in only:
        (OUT / "channels.json").write_text(json.dumps(channels(), indent=1) + "\n")
    if "chains" in only:
        np.savez_compressed(OUT / "chains.npz", **chains())
    print("written", sorted(only))


if __name__ == "__main__":
    main()
