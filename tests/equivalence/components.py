"""Component-level computations, written twice: once against the notebook package (``unhiggsed``) and once against
``masspairing``. Each function returns a dict name -> array; the reference maker freezes the notebook side.
"""

from __future__ import annotations

import numpy as np

SEED = 20261002


def _rng(k):
    return np.random.default_rng(SEED + k)


# --------------------------------------------------------------------------------------------------------------
# notebook side
# --------------------------------------------------------------------------------------------------------------
def notebook_side():
    from unhiggsed import winding as nw
    from unhiggsed.hmc import chiral_mass as ncm
    from unhiggsed.hmc import lattice_symmetry as nls
    from unhiggsed.hmc import rational as nrat
    from unhiggsed.hmc.action import Model as NModel
    from unhiggsed.hmc.lattice import Lattice as NLattice
    from unhiggsed.hmc.majorana import (
        MajoranaMeasure,
        MajoranaWedgeHasenbuschModel,
        MajoranaWedgeModel,
        pf_sign_config,
        pfaffian_householder,
        pfaffian_logpr,
    )
    from unhiggsed.hmc.observables import FermionMeasure as NFM
    from unhiggsed.hmc.observables import scalar_observables as nso
    from unhiggsed.hmc.wedge import WedgeLattice, WedgeMeasure, WedgeModel, build_kinetic_d, source_pattern

    out = {}
    # lattices and patterns
    for tag, shape, bc in (
        ("pppa4", (4,) * 4, (1, 1, 1, -1)),
        ("aaaa4", (4,) * 4, (-1,) * 4),
        ("aaaa4x8", (4, 4, 4, 8), (-1,) * 4),
    ):
        lat = WedgeLattice(shape, bc=bc)
        out[f"K_{tag}"] = build_kinetic_d(lat).toarray()
        out[f"C0_{tag}"] = source_pattern(lat).toarray()
        for comp in ((0, 3), (0, 1), (0, 2)):
            for ds in (1, -1):
                out[f"Cchi_{tag}_{comp[0]}{comp[1]}_{ds}"] = ncm.chiral_mass(lat, comp, ds).toarray()
        out[f"Cfull_{tag}"] = ncm.all_planes_mass(lat).toarray()
        out[f"plane_par_{tag}"] = ncm.plane_op(lat, 1, 2, parity=1).toarray()
        out[f"plane_nophase_{tag}"] = ncm.plane_op(lat, 0, 3, phase=()).toarray()
        tp = ncm.TasteProjector(lat)
        out[f"Q_{tag}"] = tp.dense()
        out[f"PL_{tag}"] = tp.dense_PL()
        out[f"CL_{tag}"] = ncm.light_pattern(lat)
    # rational approximations
    for i, (lo, hi) in enumerate(((1e-3, 50.0), (2e-6, 80.0), (1e-9, 70.0))):
        pf = nrat.invsqrt_partial_fractions(12, lo, hi)
        out[f"pf_inv_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])
        pf = nrat.ratio_sqrt_partial_fractions(12, 0.05, 0.5, lo, hi)
        out[f"pf_ratio_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])
        pf = nrat.shifted_invsqrt_partial_fractions(12, 0.5, lo, hi)
        out[f"pf_shift_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])
        pf = nrat.invsqrt_partial_fractions_robust(30, lo, hi)
        out[f"pf_robust_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])

    # models on 4^4: plain eps, wedge, chiral source, selective source, Hasenbusch
    def model_items(tag, model, fields, lat, hb=False):
        rng = _rng(1)
        eta = lat.random_psi(rng)
        Y = model.D.prepare(fields)
        out[f"{tag}_apply"] = model.D.apply(eta, Y=Y)
        out[f"{tag}_apply_dag"] = model.D.apply_dag(eta, Y=Y)
        out[f"{tag}_sb"] = np.array(model.s_boson(fields))
        out[f"{tag}_fb"] = np.asarray(model.f_boson(fields)).reshape(-1)
        if hb:
            phis = []
            for k in range(model.nterms):
                v, r = model.heatbath_term(k, fields, lat.random_psi(rng))
                phis.append(v)
                out[f"{tag}_hb{k}"] = v
            model.set_window(r["ritz_min"], r["ritz_max"])
            for k in range(model.nterms):
                out[f"{tag}_s{k}"] = np.array(model.s_term(k, fields, phis[k]))
                out[f"{tag}_f{k}"] = np.asarray(model.f_term(k, fields, phis[k])[0]).reshape(-1)
        else:
            phi, r = model.heatbath_phi(fields, eta)
            out[f"{tag}_phi"] = phi
            model.set_window(r["ritz_min"], r["ritz_max"])
            out[f"{tag}_spf"] = np.array(model.s_pf(fields, phi))
            out[f"{tag}_fpf"] = np.asarray(model.f_pf(fields, phi)[0]).reshape(-1)
            out[f"{tag}_dinv"] = model.solve_Dinv(fields, eta)

    lat = NLattice((4,) * 4)
    nm = NModel(lat, 2.41, -0.01, 1.0)
    sigma = lat.random_sigma(_rng(2), 0.7)
    model_items("plain", nm, sigma, lat)
    out["plain_dense"] = nm.D.dense(sigma)
    out["scalar_Ct"] = nso(lat, sigma)["Ct"]
    out["scalar_S"] = np.array([nso(lat, sigma)[k] for k in ("S_0", "S_pi", "S_pmin", "S_pi_pmin", "sigma2", "sigma4")])
    fm = NFM(nm, n_noise=3, seed=5)
    b = fm.bilinears(sigma)
    out["fm_noise"] = np.array([b[k] for k in ("phi_sq", "phi_stag_sq", "O4", "O4_biased")])
    fm = NFM(nm, exact=True, seed=5)
    b = fm.bilinears(sigma)
    out["fm_exact"] = np.array([b[k] for k in ("phi_sq", "phi_stag_sq", "O4")])
    c = fm.correlator(sigma, x0=(1, 2, 3, 1))
    out["fm_Gf"], out["fm_Cf"] = c["Gf"], c["Cf"]

    wl = WedgeLattice((4,) * 4)
    wm = WedgeModel(wl, 2.41, -0.01, 1.0, g10=0.1, g6=0.05)
    f = wm.start(_rng(3), "hot+hot")
    model_items("wedge", wm, f, wl)
    out["wedge_dense"] = wm.D.dense(f)
    meas = WedgeMeasure(wm, exact=True, chi10="full")
    b = meas.bilinears(f)
    for k in ("chi10", "chi6", "E10", "E6", "phi_src", "phi_src_dh", "phi_sq", "O4", "dimer_sq_par"):
        out[f"wm_exact_{k}"] = np.array(b[k])
    out["wm_exact_corners"] = b["chi10_corners"]
    out["wm_exact_pmin"] = b["chi6_pmin"]
    meas = WedgeMeasure(wm, n_noise=3, seed=11, chi10="full")
    b = meas.bilinears(f)
    for k in ("chi10", "chi6", "E10", "E6", "phi_src", "phi_src_dh", "phi_sq", "O4", "dimer_sq_par"):
        out[f"wm_noise_{k}"] = np.array(b[k])
    out["wm_noise_corners"] = b["chi10_corners"]

    al = WedgeLattice((4,) * 4, bc=(-1,) * 4)
    cm = MajoranaWedgeModel(al, 2.41, -0.01, 1.0, h_sel=0.0, flavours=[1.0] * 4, h10=0.7, h10_pattern="chiral")
    f = cm.start(_rng(4), "hot+zero")
    model_items("chiral", cm, f, al)
    for tag, exact in (("n1_exact", True), ("n1_noise", False)):
        meas = MajoranaMeasure(cm, n_noise=3, exact=exact, seed=13, taste=True)
        b = meas.bilinears(f)
        for k in (
            "phi_f",
            "chi_f",
            "chi_f_sum",
            "phi_L",
            "chi_L_sum",
            "chi_L_pmin",
            "GL_p0",
            "GR_p0",
            "phi6L_sq",
            "O4",
        ):
            out[f"{tag}_{k}"] = np.asarray(b[k])
        if exact:
            for k in ("CL_t", "GL_t", "CR_t", "phi_T_R", "C_T_L"):
                out[f"{tag}_{k}"] = np.asarray(b[k])
            c = meas.correlator(f)
        else:
            c = meas.taste_correlator(f, x0=(0, 1, 2, 3))
            out[f"{tag}_CfL"] = c["Cf_L"]
            c = meas.correlator(f, x0=(0, 1, 2, 3))
        out[f"{tag}_Cf_f"] = c["Cf_f"]

    sm = MajoranaWedgeModel(wl, 2.41, -0.01, 1.0, h_sel=0.3, flavours=[0.0, 0.0, 0.0, 1.0])
    f = sm.start(_rng(5), "hot+zero")
    model_items("selective", sm, f, wl)
    out["selective_M"] = sm.D.dense_real4(f)
    meas = MajoranaMeasure(sm, exact=True, seed=17)
    b = meas.bilinears(f)
    for k in ("phi_f", "chi_f", "phi_T"):
        out[f"sel_exact_{k}"] = np.asarray(b[k])
    r = pf_sign_config(sm, f, [0.1, 0.3, 0.6], flips=True)
    out["pf_sign"], out["pf_log"], out["pf_flips"] = r["sign"], r["logratio"], r["flips"]
    M = sm.D.dense_real4(f)
    out["pf_hh"] = np.array(pfaffian_householder(M))
    out["pf_pr"] = np.array(pfaffian_logpr(M))

    hm = MajoranaWedgeHasenbuschModel(
        al, 2.41, -0.01, 1.0, [0.05, 0.5], h_sel=0.0, flavours=[1.0] * 4, h10=0.7, h10_pattern="chiral"
    )
    f = hm.start(_rng(6), "hot+zero")
    model_items("hasenbusch", hm, f, al, hb=True)

    # symmetry zeros and corner blocks
    pats = {
        "asd03": ncm.chiral_mass(al, (0, 3), -1),
        "asd01": ncm.chiral_mass(al, (0, 1), -1),
        "sd03": ncm.chiral_mass(al),
    }
    res, nG, nS = nls.classify(pats, al, ncm.chiral_mass(al), translations=False)
    out["sym_ratios"] = np.array([res[k] for k in sorted(res)] + [nG, nS])
    L4 = NLattice((4,) * 4, bc=(-1,) * 4)
    sig = L4.random_sigma(_rng(7), 0.8)
    D = nw.doublet_operator(L4, 2.41, sig, h=0.5)
    cb = nw.CornerBlocks(D, L4.shape, n_int=2)
    rec = nw.analyse(cb, L4.bc)["all"]
    out["corner"] = np.array([rec[k] for k in ("f_M", "N_odd", "alpha", "normO1", "normO3", "normE1")])
    return out


# --------------------------------------------------------------------------------------------------------------
# clean side
# --------------------------------------------------------------------------------------------------------------
def clean_side():
    from masspairing import corner as cw
    from masspairing import patterns as cp
    from masspairing import rational as crat
    from masspairing import symmetry as cs
    from masspairing.action import build_model
    from masspairing.lattice import Lattice, kinetic_matrix
    from masspairing.measure import ChannelMeasure, FermionMeasure, N1Measure, scalar_observables
    from masspairing.pfaffian import pf_sign_config, pfaffian_householder, pfaffian_logpr

    out = {}
    for tag, shape, bc in (
        ("pppa4", (4,) * 4, (1, 1, 1, -1)),
        ("aaaa4", (4,) * 4, (-1,) * 4),
        ("aaaa4x8", (4, 4, 4, 8), (-1,) * 4),
    ):
        lat = Lattice(shape, bc=bc)
        out[f"K_{tag}"] = kinetic_matrix(lat).toarray()
        out[f"C0_{tag}"] = cp.source_pattern(lat).toarray()
        for comp in ((0, 3), (0, 1), (0, 2)):
            for ds in (1, -1):
                out[f"Cchi_{tag}_{comp[0]}{comp[1]}_{ds}"] = cp.chiral_mass(lat, comp, ds).toarray()
        out[f"Cfull_{tag}"] = cp.all_planes_mass(lat).toarray()
        out[f"plane_par_{tag}"] = cp.plane_op(lat, 1, 2, parity=1).toarray()
        out[f"plane_nophase_{tag}"] = cp.plane_op(lat, 0, 3, phase=()).toarray()
        tp = cp.TasteProjector(lat)
        out[f"Q_{tag}"] = tp.dense()
        out[f"PL_{tag}"] = tp.dense_PL()
        out[f"CL_{tag}"] = cp.light_pattern(lat)
    for i, (lo, hi) in enumerate(((1e-3, 50.0), (2e-6, 80.0), (1e-9, 70.0))):
        pf = crat.invsqrt_partial_fractions(12, lo, hi)
        out[f"pf_inv_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])
        pf = crat.ratio_sqrt_partial_fractions(12, 0.05, 0.5, lo, hi)
        out[f"pf_ratio_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])
        pf = crat.shifted_invsqrt_partial_fractions(12, 0.5, lo, hi)
        out[f"pf_shift_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])
        pf = crat.invsqrt_partial_fractions_robust(30, lo, hi)
        out[f"pf_robust_{i}"] = np.concatenate([[pf["c0"]], pf["rho"], pf["poles"]])

    def model_items(tag, model, fields, lat, hb=False):
        rng = _rng(1)
        eta = lat.random_psi(rng)
        Y = model.D.prepare(fields)
        out[f"{tag}_apply"] = model.D.apply(eta, Y=Y)
        out[f"{tag}_apply_dag"] = model.D.apply_dag(eta, Y=Y)
        out[f"{tag}_sb"] = np.array(model.s_boson(fields))
        out[f"{tag}_fb"] = np.asarray(model.f_boson(fields)).reshape(-1)
        if hb:
            phis = []
            for k in range(model.nterms):
                v, r = model.heatbath_term(k, fields, lat.random_psi(rng))
                phis.append(v)
                out[f"{tag}_hb{k}"] = v
            model.set_window(r["ritz_min"], r["ritz_max"])
            for k in range(model.nterms):
                out[f"{tag}_s{k}"] = np.array(model.s_term(k, fields, phis[k]))
                out[f"{tag}_f{k}"] = np.asarray(model.f_term(k, fields, phis[k])[0]).reshape(-1)
        else:
            phi, r = model.heatbath_phi(fields, eta)
            out[f"{tag}_phi"] = phi
            model.set_window(r["ritz_min"], r["ritz_max"])
            out[f"{tag}_spf"] = np.array(model.s_pf(fields, phi))
            out[f"{tag}_fpf"] = np.asarray(model.f_pf(fields, phi)[0]).reshape(-1)
            out[f"{tag}_dinv"] = model.solve_Dinv(fields, eta)

    lat = Lattice((4,) * 4)
    nm = build_model(lat, 2.41, -0.01, 1.0)
    sigma = lat.random_sigma(_rng(2), 0.7)
    f = nm.pack(sigma, np.zeros(0))
    model_items("plain", nm, f, lat)
    out["plain_dense"] = nm.D.dense(f)
    so = scalar_observables(lat, sigma)
    out["scalar_Ct"] = so["Ct"]
    out["scalar_S"] = np.array([so[k] for k in ("S_0", "S_pi", "S_pmin", "S_pi_pmin", "sigma2", "sigma4")])
    fm = FermionMeasure(nm, n_noise=3, seed=5)
    b = fm.bilinears(f)
    out["fm_noise"] = np.array([b[k] for k in ("phi_sq", "phi_stag_sq", "O4", "O4_biased")])
    fm = FermionMeasure(nm, exact=True, seed=5)
    b = fm.bilinears(f)
    out["fm_exact"] = np.array([b[k] for k in ("phi_sq", "phi_stag_sq", "O4")])
    c = fm.correlator(f, x0=(1, 2, 3, 1))
    out["fm_Gf"], out["fm_Cf"] = c["Gf"], c["Cf"]

    wm = build_model(lat, 2.41, -0.01, 1.0, g10=0.1, g6=0.05)
    f = wm.start(_rng(3), "hot+hot")
    model_items("wedge", wm, f, lat)
    out["wedge_dense"] = wm.D.dense(f)
    for tag, kw in (("wm_exact", dict(exact=True)), ("wm_noise", dict(n_noise=3, seed=11))):
        b = ChannelMeasure(wm, **kw).bilinears(f)
        for k in ("chi10", "chi6", "E10", "E6", "phi_src", "phi_src_dh", "phi_sq", "O4", "dimer_sq_par"):
            out[f"{tag}_{k}"] = np.array(b[k])
        out[f"{tag}_corners"] = b["chi10_corners"]
        if tag == "wm_exact":
            out[f"{tag}_pmin"] = b["chi6_pmin"]

    al = Lattice((4,) * 4, bc=(-1,) * 4)
    cm = build_model(al, 2.41, -0.01, 1.0, h=0.7, pattern="chiral")
    f = cm.start(_rng(4), "hot+zero")
    model_items("chiral", cm, f, al)
    for tag, exact in (("n1_exact", True), ("n1_noise", False)):
        meas = N1Measure(cm, n_noise=3, exact=exact, seed=13, taste=True)
        b = meas.bilinears(f)
        for k in (
            "phi_f",
            "chi_f",
            "chi_f_sum",
            "phi_L",
            "chi_L_sum",
            "chi_L_pmin",
            "GL_p0",
            "GR_p0",
            "phi6L_sq",
            "O4",
        ):
            out[f"{tag}_{k}"] = np.asarray(b[k])
        if exact:
            for k in ("CL_t", "GL_t", "CR_t", "phi_T_R", "C_T_L"):
                out[f"{tag}_{k}"] = np.asarray(b[k])
            c = meas.correlator(f)
        else:
            c = meas.taste_correlator(f, x0=(0, 1, 2, 3))
            out[f"{tag}_CfL"] = c["Cf_L"]
            c = meas.correlator(f, x0=(0, 1, 2, 3))
        out[f"{tag}_Cf_f"] = c["Cf_f"]

    sm = build_model(lat, 2.41, -0.01, 1.0, h_sel=0.3, mask=(0.0, 0.0, 0.0, 1.0))
    f = sm.start(_rng(5), "hot+zero")
    model_items("selective", sm, f, lat)
    out["selective_M"] = sm.D.dense_real4(f)
    b = N1Measure(sm, exact=True, seed=17).bilinears(f)
    for k in ("phi_f", "chi_f", "phi_T"):
        out[f"sel_exact_{k}"] = np.asarray(b[k])
    r = pf_sign_config(sm, f, [0.1, 0.3, 0.6], flips=True)
    out["pf_sign"], out["pf_log"], out["pf_flips"] = r["sign"], r["logratio"], r["flips"]
    M = sm.D.dense_real4(f)
    out["pf_hh"] = np.array(pfaffian_householder(M))
    out["pf_pr"] = np.array(pfaffian_logpr(M))

    hm = build_model(al, 2.41, -0.01, 1.0, h=0.7, pattern="chiral", hasenbusch=[0.05, 0.5])
    f = hm.start(_rng(6), "hot+zero")
    model_items("hasenbusch", hm, f, al, hb=True)

    pats = {
        "asd03": cp.chiral_mass(al, (0, 3), -1),
        "asd01": cp.chiral_mass(al, (0, 1), -1),
        "sd03": cp.chiral_mass(al),
    }
    res, nG, nS = cs.classify(pats, al, cp.chiral_mass(al), translations=False)
    out["sym_ratios"] = np.array([res[k] for k in sorted(res)] + [nG, nS])
    sig = al.random_sigma(_rng(7), 0.8)
    D = cw.doublet_operator(al, 2.41, sig, h=0.5)
    cb = cw.CornerBlocks(D, al.shape, n_int=2)
    rec = cw.analyse(cb, al.bc)["all"]
    out["corner"] = np.array([rec[k] for k in ("f_M", "N_odd", "alpha", "normO1", "normO3", "normE1")])
    return out
