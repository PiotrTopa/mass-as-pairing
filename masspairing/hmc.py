"""Rational hybrid Monte Carlo: Gaussian momenta, Lanczos heat bath, Omelyan integration, Metropolis step.

``HMC`` integrates with the second-order Omelyan (PQPQP) scheme, lambda = 0.1931833275037836. ``MTSHMC`` nests it
over several time scales for a ``HasenbuschModel``. Both are deterministic and palindromic, hence reversible and
area preserving; the accept/reject uses the exact (Lanczos) actions, so the measure does not depend on the
integrator, the step size or the force fraction.

Trajectory-length jitter (``tau_jitter = j``): tau = tau_0 (1 + j u), u uniform in (-1, 1), drawn from a stream
independent of the configuration. It is exact and breaks the freezing of nearly free link-field modes at
omega tau = n pi.
"""

from __future__ import annotations

import math

import numpy as np

from .lattice import xp_of

OMELYAN_LAMBDA = 0.1931833275037836
JITTER_SEED_OFFSET = 424242


class HMC:
    def __init__(self, model, tau=1.0, nsteps=10, seed=0, tau_jitter=0.0, lanczos_window=True):
        self.model, self.tau0, self.nsteps = model, float(tau), int(nsteps)
        self.tau = self.tau0
        self.rng = np.random.default_rng(seed)
        self.tau_jitter = float(tau_jitter)
        self.jrng = np.random.default_rng(seed + JITTER_SEED_OFFSET) if self.tau_jitter else None
        self.lat = model.lat
        self.lanczos_window = lanczos_window
        self.spec = None

    def _momenta(self):
        return self.lat.xp.asarray(self.rng.normal(size=self.model.nfields))

    def _draw_tau(self):
        if self.jrng is not None:
            self.tau = self.tau0 * (1.0 + self.tau_jitter * (2.0 * self.jrng.random() - 1.0))

    def hamiltonian(self, fields, pi, phi):
        m = self.model
        spf, info = m.s_pf(fields, phi, want_ritz=True)
        return float(0.5 * float(xp_of(pi).sum(pi**2)) + m.s_boson(fields) + spf), info

    def force(self, fields, phi):
        fb = self.model.f_boson(fields)
        ff, it = self.model.f_pf(fields, phi)
        return fb + ff, it

    def integrate(self, fields, pi, phi, nsteps=None, direction=+1):
        nsteps = nsteps or self.nsteps
        dt = (self.tau / nsteps) * direction
        iters = 0
        lam = OMELYAN_LAMBDA
        for _ in range(nsteps):
            fields = fields + lam * dt * pi
            f, it = self.force(fields, phi)
            iters += it
            pi = pi + 0.5 * dt * f
            fields = fields + (1.0 - 2.0 * lam) * dt * pi
            f, it = self.force(fields, phi)
            iters += it
            pi = pi + 0.5 * dt * f
            fields = fields + lam * dt * pi
        return fields, pi, iters

    def trajectory(self, fields, nsteps=None, force_accept=False):
        """One trajectory; returns (new fields, info dict)."""
        self._draw_tau()
        m = self.model
        pi = self._momenta()
        eta = self.lat.random_psi(self.rng)
        phi, info = m.heatbath_phi(fields, eta)
        lo, hi = info["ritz_min"], info["ritz_max"]
        if self.spec is not None and self.lanczos_window:
            lo, hi = min(lo, self.spec[0]), max(hi, self.spec[1])
        m.set_window(lo, hi)
        H0, info0 = self.hamiltonian(fields, pi, phi)
        fields1, pi1, iters = self.integrate(fields, pi, phi, nsteps)
        H1, info1 = self.hamiltonian(fields1, pi1, phi)
        self.spec = (
            min(info0["ritz_min"], info1["ritz_min"]),
            max(info0["ritz_max"], info1["ritz_max"]),
        )
        dH = H1 - H0
        acc = force_accept or (dH <= 0) or (self.rng.random() < math.exp(-dH))
        out = fields1 if acc else fields
        return out, dict(
            dH=dH,
            accepted=bool(acc),
            H0=H0,
            cg_iters=iters,
            ritz_min=self.spec[0],
            ritz_max=self.spec[1],
            rational_err=m.pf["maxerr"],
            lanczos_m=(info["m"], info0["m"], info1["m"]),
        )

    def reversibility(self, fields, nsteps=None):
        """Integrate forward, flip the momenta, integrate back: relative |d fields|, |d pi| and dH."""
        m = self.model
        pi = self._momenta()
        phi, info = m.heatbath_phi(fields, self.lat.random_psi(self.rng))
        m.set_window(info["ritz_min"], info["ritz_max"])
        H0, _ = self.hamiltonian(fields, pi, phi)
        s1, p1, _ = self.integrate(fields, pi, phi, nsteps)
        H1, _ = self.hamiltonian(s1, p1, phi)
        s2, p2, _ = self.integrate(s1, -p1, phi, nsteps)
        xp = xp_of(fields)
        ds = float(xp.linalg.norm(s2 - fields) / xp.linalg.norm(fields))
        dp = float(xp.linalg.norm(p2 + pi) / xp.linalg.norm(pi))
        return dict(dsigma=ds, dpi=dp, dH=H1 - H0)


class MTSHMC(HMC):
    """Nested Omelyan integrator over time scales for a HasenbuschModel.

    levels = [n_0, m_1, ...]: level 0 takes n_0 steps per trajectory, level l takes m_l sub-steps per half step of
    level l - 1. ``assign[k]`` is the level of pseudofermion term k (k < nterms) and of the boson (k = nterms);
    default: term k -> level min(k, L - 1), boson -> the finest level. Consecutive kicks of one level with no
    position update in between share one force evaluation.
    """

    def __init__(self, model, tau=1.0, levels=(10,), assign=None, seed=0, tau_jitter=0.0, lanczos_window=True):
        super().__init__(
            model,
            tau=tau,
            nsteps=int(levels[0]),
            seed=seed,
            tau_jitter=tau_jitter,
            lanczos_window=lanczos_window,
        )
        self.levels = [int(x) for x in levels]
        nl, nt = len(self.levels), model.nterms
        if assign is None:
            assign = [min(k, nl - 1) for k in range(nt)] + [nl - 1]
        self.assign = [int(a) for a in assign]
        assert len(self.assign) == nt + 1 and all(0 <= a < nl for a in self.assign), (self.assign, nl)
        self.groups = [[k for k in range(nt + 1) if self.assign[k] == lv] for lv in range(nl)]
        assert all(self.groups), "every level needs at least one force"
        self.last_stats = None

    def hamiltonian(self, fields, pi, phis):
        m = self.model
        vals, infos = m.s_pf_terms(fields, phis, want_ritz=True)
        kin = 0.5 * float(xp_of(pi).sum(pi**2))
        return float(kin + m.s_boson(fields) + sum(vals)), dict(
            terms=vals,
            ritz_min=min(r["ritz_min"] for r in infos),
            ritz_max=max(r["ritz_max"] for r in infos),
            m=tuple(r["m"] for r in infos),
        )

    def level_force(self, lv, fields, phis, stats=None):
        m = self.model
        f, it = None, 0
        Y = None
        for k in self.groups[lv]:
            if k == m.nterms:
                fk = m.f_boson(fields)
            else:
                if Y is None:
                    Y = m.D.prepare(fields)
                fk, itk = m.f_term(k, fields, phis[k], Y=Y)
                it += itk
            f = fk if f is None else f + fk
        if stats is not None:
            stats["iters"] += it
            stats["iters_level"][lv] += it
            stats["nforce"][lv] += 1
        return f

    def integrate(self, fields, pi, phis, nsteps=None, direction=+1):
        n0 = int(nsteps or self.levels[0])
        counts = [n0] + self.levels[1:]
        nl = len(counts)
        lam = OMELYAN_LAMBDA
        st = dict(sigma=fields, pi=pi, iters=0, iters_level=[0] * nl, nforce=[0] * nl)
        cache = [(None, None)] * nl

        def kick(lv, h):
            if cache[lv][0] is not st["sigma"]:
                cache[lv] = (st["sigma"], self.level_force(lv, st["sigma"], phis, st))
            st["pi"] = st["pi"] + h * cache[lv][1]

        def inner(lv, T):
            if lv + 1 == nl:
                st["sigma"] = st["sigma"] + T * st["pi"]
            else:
                run(lv + 1, T / counts[lv + 1], counts[lv + 1])

        def run(lv, h, n):
            kick(lv, lam * h)
            for i in range(n):
                inner(lv, 0.5 * h)
                kick(lv, (1.0 - 2.0 * lam) * h)
                inner(lv, 0.5 * h)
                kick(lv, (2.0 * lam if i < n - 1 else lam) * h)

        run(0, direction * (self.tau / n0), n0)
        self.last_stats = st
        return st["sigma"], st["pi"], st["iters"]

    def trajectory(self, fields, nsteps=None, force_accept=False):
        self._draw_tau()
        m = self.model
        pi = self._momenta()
        phis, infos = [], []
        for k in range(m.nterms):
            eta = self.lat.random_psi(self.rng)
            phi, info = m.heatbath_term(k, fields, eta)
            phis.append(phi)
            infos.append(info)
        lo, hi = min(r["ritz_min"] for r in infos), max(r["ritz_max"] for r in infos)
        if self.spec is not None and self.lanczos_window:
            lo, hi = min(lo, self.spec[0]), max(hi, self.spec[1])
        m.set_window(lo, hi)
        H0, info0 = self.hamiltonian(fields, pi, phis)
        fields1, pi1, iters = self.integrate(fields, pi, phis, nsteps)
        H1, info1 = self.hamiltonian(fields1, pi1, phis)
        self.spec = (
            min(info0["ritz_min"], info1["ritz_min"]),
            max(info0["ritz_max"], info1["ritz_max"]),
        )
        dH = H1 - H0
        acc = force_accept or (dH <= 0) or (self.rng.random() < math.exp(-dH))
        out = fields1 if acc else fields
        st = self.last_stats
        return out, dict(
            dH=dH,
            accepted=bool(acc),
            H0=H0,
            cg_iters=iters,
            ritz_min=self.spec[0],
            ritz_max=self.spec[1],
            rational_err=m.pf["maxerr"],
            lanczos_m=(tuple(r["m"] for r in infos), info0["m"], info1["m"]),
            cg_iters_level=list(st["iters_level"]),
            nforce_level=list(st["nforce"]),
            dS_terms=[float(a - b) for a, b in zip(info1["terms"], info0["terms"], strict=False)],
            n0=int(nsteps or self.levels[0]),
        )

    def reversibility(self, fields, nsteps=None):
        m = self.model
        pi = self._momenta()
        phis, infos = [], []
        for k in range(m.nterms):
            phi, info = m.heatbath_term(k, fields, self.lat.random_psi(self.rng))
            phis.append(phi)
            infos.append(info)
        m.set_window(min(r["ritz_min"] for r in infos), max(r["ritz_max"] for r in infos))
        H0, _ = self.hamiltonian(fields, pi, phis)
        s1, p1, _ = self.integrate(fields, pi, phis, nsteps)
        H1, _ = self.hamiltonian(s1, p1, phis)
        s2, p2, _ = self.integrate(s1, -p1, phis, nsteps)
        xp = xp_of(fields)
        ds = float(xp.linalg.norm(s2 - fields) / xp.linalg.norm(fields))
        dp = float(xp.linalg.norm(p2 + pi) / xp.linalg.norm(pi))
        return dict(dsigma=ds, dpi=dp, dH=H1 - H0)
