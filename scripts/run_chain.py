#!/usr/bin/env python3
"""Run one Markov chain of the model and store its time series.

    python scripts/run_chain.py --L 4 --y 2.41 --kappa -0.01 --ntraj 200 --out runs/eps          # epsilon model
    python scripts/run_chain.py --L 6 --y 2.41 --g10 0.1 --g6 0.1 --measure channels --tau-jitter 0.3 --out runs/wedge
    python scripts/run_chain.py --L 6 --bc aaaa --y 3.0 --h 2 --pattern chiral --measure n1 --exact --out runs/n1
    python scripts/run_chain.py --L 4 --y 2.41 --h-sel 0.2 --flavours 4 --measure n1 --exact --pf-sign --out runs/sel

Model: the epsilon-vertex sigma field at (y, kappa, lambda); optional link fields of the completed 10-channel term
(g10, g6, --quads); optional flavour-blind pattern h C (--h, --pattern) or flavour-selective source (--h-sel,
--flavours; RHMC on |Pf M|, sign not in the weight). Integrator: Omelyan, or Hasenbusch + nested time scales
(--hasenbusch, --mts). Measurements: sigma observables every --meas-every trajectories after --ntherm; fermion
observables every --ferm-every measurements with the measure --measure (epsilon | channels | n1), dense (--exact)
or with --n-noise Z2 noise vectors.

Output: <out>/L{L}[x{Lt}]_y{y}_k{kappa}_g{g10}_g6{g6}[suffixes].npz with the series ts_<key>, the final configuration
(fields_final, flat [sigma, s]), the random-number state and a meta record. --resume continues a stored chain
(--ntraj is the total). Seeds: the chain uses seed + int(1000 y) + int(100000 g10); the fermion noise
that + 7 + 100003 * (trajectories already stored).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from masspairing.action import build_model  # noqa: E402
from masspairing.flavour import flavour_mask  # noqa: E402
from masspairing.hmc import HMC, MTSHMC  # noqa: E402
from masspairing.lattice import Lattice, get_xp, parse_bc, to_numpy  # noqa: E402
from masspairing.measure import ChannelMeasure, FermionMeasure, N1Measure, scalar_observables  # noqa: E402
from masspairing.wedge import link_field_observables  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--L", type=int, required=True)
    a("--Lt", type=int, default=None, help="time extent (default L)")
    a("--bc", default="pppa", help="fermion boundary conditions per axis, p/a (default pppa)")
    a("--y", type=float, required=True, help="Yukawa coupling (P_c = (2.41, -0.01) in these units)")
    a("--kappa", type=float, default=-0.01)
    a("--lam", type=float, default=1.0)
    a("--g10", type=float, default=0.0, help="10-channel coupling of the completed term (0 = no link fields)")
    a("--g6", type=float, default=0.0, help="two-site 6-channel partner coupling, |g6| <= g10")
    a("--quads", default="all", choices=["all", "checkerboard", "links"])
    a("--h", type=float, default=0.0, help="flavour-blind source strength (K -> K - h C)")
    a("--pattern", default="c0", choices=["c0", "chiral", "full"], help="source pattern")
    a("--h-sel", type=float, default=0.0, help="flavour-selective source strength (real basis, |Pf| weight)")
    a(
        "--flavours",
        nargs="+",
        default=["4"],
        help="flavours of the selective source (1-based) or four weights",
    )
    a("--measure", default="epsilon", choices=["epsilon", "channels", "n1"])
    a("--exact", action="store_true", help="dense fermion propagator (small lattices)")
    a("--n-noise", type=int, default=4)
    a("--no-corr", action="store_true", help="skip the point-source correlators")
    a("--no-six-fermion", action="store_true", help="skip the composite channels of the n1 measure")
    a(
        "--pf-sign",
        action="store_true",
        help="Pfaffian sign of the selective source at every fermion measurement",
    )
    a("--pf-sign-h", type=float, nargs="+", default=None, help="extra h values of the sign profile")
    a("--store-cfgs", action="store_true", help="store the configuration at every fermion measurement")
    a("--ntraj", type=int, default=200, help="trajectories after thermalisation (total, also under --resume)")
    a("--ntherm", type=int, default=50)
    a("--tau", type=float, default=1.0)
    a("--tau-jitter", type=float, default=0.0, help="tau = tau0 (1 + j u), u uniform in (-1, 1)")
    a("--nsteps", type=int, default=10)
    a("--nsteps-therm", type=int, default=None, help="steps in the first half of thermalisation (2 nsteps)")
    a("--hasenbusch", type=float, nargs="+", default=None, metavar="SHIFT")
    a("--mts", type=int, nargs="+", default=None, metavar="N")
    a("--mts-assign", type=int, nargs="+", default=None, metavar="LEVEL")
    a("--n-rational", type=int, default=12)
    a("--cg-tol", type=float, default=1e-8)
    a("--lanczos-check-grow", type=int, default=256)
    a("--lanczos-m-max", type=int, default=None)
    a("--start", default="hot+zero", help="sigma hot|cold|afm plus +hot|+zero|+dimer for the link fields")
    a("--hot-scale", type=float, default=0.5)
    a("--seed", type=int, default=0)
    a("--meas-every", type=int, default=1)
    a("--ferm-every", type=int, default=2)
    a("--checkpoint-every", type=int, default=50)
    a("--device", default="cpu", choices=["cpu", "cuda"])
    a("--resume", action="store_true")
    a("--out", required=True)
    return p.parse_args(argv)


def chain_path(args):
    L, Lt = args.L, args.Lt or args.L
    tag = f"L{L}" if Lt == L else f"L{L}x{Lt}"
    suf = ("_links" if args.quads == "links" else "") + (f"_h{args.h:g}" if args.h else "")
    if args.h_sel:
        m = flavour_mask(args.flavours)
        suf += f"_hsel{args.h_sel:g}_f" + "".join(str(i + 1) for i in range(4) if m[i])
    if args.pattern != "c0":
        suf += f"_{args.pattern}"
    if args.bc != "pppa":
        suf += f"_bc{args.bc}"
    name = f"{tag}_y{args.y:g}_k{args.kappa:g}_g{args.g10:g}_g6{args.g6:g}{suf}.npz"
    return pathlib.Path(args.out) / name


def run(args):
    xp = get_xp(args.device)
    L, Lt = args.L, args.Lt or args.L
    lat = Lattice((L, L, L, Lt), bc=parse_bc(args.bc), xp=xp)
    selective = bool(args.h_sel)
    model = build_model(
        lat,
        args.y,
        args.kappa,
        args.lam,
        g10=args.g10,
        g6=args.g6,
        quads=args.quads,
        h=args.h,
        pattern=args.pattern,
        h_sel=args.h_sel,
        mask=flavour_mask(args.flavours) if selective else (0.0, 0.0, 0.0, 1.0),
        hasenbusch=args.hasenbusch,
        n_rational=args.n_rational,
        cg_tol=args.cg_tol,
        lanczos_check_grow=args.lanczos_check_grow,
        lanczos_m_max=args.lanczos_m_max,
    )
    seed = args.seed + int(1000 * args.y) + int(100000 * args.g10)
    if args.hasenbusch is None:
        hmc = HMC(model, tau=args.tau, nsteps=args.nsteps, seed=seed, tau_jitter=args.tau_jitter)
    else:
        hmc = MTSHMC(
            model,
            tau=args.tau,
            levels=args.mts or [args.nsteps],
            assign=args.mts_assign,
            seed=seed,
            tau_jitter=args.tau_jitter,
        )
    path = chain_path(args)
    series = {}
    ntherm, ndone = args.ntherm, 0
    if args.resume and path.exists():
        old = np.load(path, allow_pickle=True)
        fields = xp.asarray(old["fields_final"])
        series = {k: list(old[k]) for k in old.files if k.startswith("ts_")}
        hmc.rng.bit_generator.state = json.loads(str(old["rng_state"]))
        ntherm = 0
        ndone = len(series.get("ts_dH", []))
    else:
        fields = model.start(hmc.rng, args.start, args.hot_scale)
    fseed = seed + 7 + 100003 * ndone
    if args.measure == "epsilon":
        fm = FermionMeasure(model, n_noise=args.n_noise, exact=args.exact, seed=fseed)
    elif args.measure == "channels":
        fm = ChannelMeasure(model, n_noise=args.n_noise, exact=args.exact, seed=fseed)
    else:
        fm = N1Measure(
            model,
            n_noise=args.n_noise,
            exact=args.exact,
            seed=fseed,
            six_fermion=not args.no_six_fermion,
            taste=args.pattern == "chiral" and not selective,
        )
    pf_hs = [args.h_sel] + list(args.pf_sign_h or [])
    if ndone >= args.ntraj:
        print(f"{path}: complete ({ndone} >= {args.ntraj})", flush=True)
        return path

    def push(k, v):
        series.setdefault("ts_" + k, []).append(v)

    t0 = time.time()
    nmeas = 0
    total = ntherm + args.ntraj - ndone
    nst_therm = args.nsteps_therm or 2 * args.nsteps
    for it in range(total):
        fields, info = hmc.trajectory(fields, nsteps=nst_therm if it < ntherm // 2 else None)
        if it < ntherm:
            continue
        for k in ("dH", "accepted", "cg_iters", "ritz_min", "ritz_max"):
            push(k, info[k])
        if args.hasenbusch is not None:
            push("cg_iters_level", info["cg_iters_level"])
            push("lanczos_m", [sum(info["lanczos_m"][0]), sum(info["lanczos_m"][1]), sum(info["lanczos_m"][2])])
            push("n0", info["n0"])
        if (it - ntherm) % args.meas_every == 0:
            sigma, s = model.split(fields)
            for k, v in scalar_observables(lat, sigma).items():
                push(k, v)
            if model.geom is not None:
                for k, v in link_field_observables(model.geom, s).items():
                    push(k, v)
            elif args.measure != "epsilon":
                push("s2", 0.0)
            do_ferm = nmeas % args.ferm_every == 0
            push("ferm_flag", do_ferm)
            if do_ferm:
                Y = model.D.prepare(fields)
                for k, v in fm.bilinears(fields, Y=Y).items():
                    push(k, v)
                if not args.no_corr:
                    co = fm.correlator(fields, Y=Y)
                    if args.measure == "n1":
                        push("Gf_f", co["Gf_f"])
                        push("Cf_f", co["Cf_f"])
                        if fm.taste and not args.exact:
                            x0 = tuple(int(c) for c in co["x0"])
                            for k, v in fm.taste_correlator(fields, Y=Y, x0=x0).items():
                                push(k, v)
                    else:
                        push("Gf", co["Gf"].real)
                        push("Cf", co["Cf"])
                if args.store_cfgs:
                    push("cfg", to_numpy(fields).copy())
                    push("cfg_traj", it - ntherm + ndone)
                if selective and args.pf_sign:
                    from masspairing.pfaffian import pf_sign_config

                    r = pf_sign_config(model, to_numpy(fields), pf_hs)
                    push("pf_sign", float(r["sign"][0]))
                    push("pf_logratio", float(r["logratio"][0]))
                    push("pf_logdet0", r["logdet0"])
                    if len(pf_hs) > 1:
                        push("pf_sign_prof", r["sign"][1:])
                        push("pf_logratio_prof", r["logratio"][1:])
            nmeas += 1
        n = it - ntherm + 1
        if n % args.checkpoint_every == 0 or it == total - 1:
            save(path, args, lat, fields, hmc, series, t0, n + ndone)
            acc = np.mean(series["ts_accepted"][-args.checkpoint_every :])
            print(
                f"{path.name}: traj {n + ndone}/{args.ntraj} acc {acc:.2f} " f"{(time.time() - t0) / n:.2f} s/traj",
                flush=True,
            )
    return path


def save(path, args, lat, fields, hmc, series, t0, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(vars(args))
    meta.update(shape=lat.shape, bc_signs=lat.bc, wall_s=time.time() - t0, ntraj_done=n, hmc_stats=hmc.model.stats)
    data = {k: np.asarray(v) for k, v in series.items()}
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(
        tmp,
        fields_final=to_numpy(fields),
        rng_state=json.dumps(hmc.rng.bit_generator.state),
        meta=json.dumps(meta, default=str),
        **data,
    )
    os.replace(tmp, path)


if __name__ == "__main__":
    run(parse())
