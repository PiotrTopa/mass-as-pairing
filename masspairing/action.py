"""Action, forces and heat bath of the model.

    S_B  = sum_x [1/2 |sigma|^2 + (lambda/4) |sigma|^4] - (kappa/2) sum_x sigma . box sigma  +  sum_f s_f^2 / (4 g_f)
    S_pf = phi^dagger (D^dagger D)^(-1/2) phi           (one doublet: det D_c = det(D^dagger D)^(1/2) > 0)

box sigma = sum_mu [sigma(x + 2mu) + sigma(x - 2mu) - 2 sigma(x)]. With the flavour-selective source the same
S_pf with M^T M gives the weight |Pf M| = det(M^T M)^(1/4).

The pseudofermion action and the heat bath use Lanczos matrix functions at tolerance ``lanczos_tol``, so the
accept/reject step targets the exact measure. The molecular-dynamics force uses Zolotarev partial fractions of
t^(-1/2) (multishift CG); their error only affects the acceptance rate, never expectation values.

Hasenbusch split: det(A)^(1/2), A = D^dagger D, as prod_k det[(A + s_{k-1})/(A + s_k)]^(1/2) det(A + s_n)^(1/2),
0 = s_0 < s_1 < ... < s_n (D^dagger D + mu^2 = (D + mu)^dagger (D + mu) since D^dagger = -D), one pseudofermion per
term.
"""

from __future__ import annotations

import numpy as np

from .lattice import Lattice, xp_of
from .operator import DoubletOperator
from .rational import force_pf
from .solvers import cg, lanczos, multishift_cg

ACTION_CAP = 20000  # Lanczos iteration caps (raised by ``lanczos_m_max``)
HEATBATH_CAP = 4000


class Model:
    """Bosonic action and the pseudofermion machinery for one DoubletOperator.

    Parameters
    ----------
    lat, op : the lattice and the fermion operator (``op.geom`` carries the link-field prior, if any)
    kappa, lam : hopping and quartic couplings of sigma
    n_rational : degree of the Zolotarev force fraction
    cg_tol, lanczos_tol : solver tolerances (the Lanczos one fixes the accept/reject precision)
    spectral_safety : the force fraction is fitted on [safety0 lambda_min, safety1 lambda_max] of the running Ritz
    window
    lanczos_check_grow, lanczos_m_max : Lanczos check spacing parameter and iteration-cap floor
    """

    def __init__(
        self,
        lat: Lattice,
        op: DoubletOperator,
        kappa: float,
        lam: float,
        n_rational: int = 12,
        cg_tol: float = 1e-8,
        lanczos_tol: float = 1e-10,
        spectral_safety=(0.2, 1.3),
        lanczos_check_grow: int = 256,
        lanczos_m_max=None,
    ):
        self.lat, self.D = lat, op
        self.y = op.y
        self.kappa, self.lam = float(kappa), float(lam)
        self.geom = op.geom
        self.n_rational = int(n_rational)
        self.cg_tol, self.lanczos_tol = cg_tol, lanczos_tol
        self.safety = spectral_safety
        self.lanczos_check_grow = int(lanczos_check_grow)
        self.lanczos_m_max = None if lanczos_m_max is None else int(lanczos_m_max)
        self.nsig = 3 * lat.V
        self.nfields = self.nsig + op.n_fields
        self.pf = None
        self.window = None
        self.stats = dict(cg_iters=0, lanczos_iters=0, solves=0)

    # ---- configurations -----------------------------------------------------------------------
    def split(self, fields):
        return self.D.split(fields)

    def pack(self, sigma, s):
        xp = xp_of(sigma)
        return xp.concatenate([sigma.reshape(-1), xp.asarray(s).reshape(-1)])

    def start(self, rng, how="hot", hot_scale=0.5):
        """Initial configuration "<sigma>[+<links>]": sigma hot | cold | afm; links hot | zero | dimer (default hot)."""
        lat, xp = self.lat, self.lat.xp
        sig_how, s_how = (how.split("+") + ["hot"])[:2] if "+" in how else (how, "hot")
        if sig_how == "hot":
            sigma = lat.random_sigma(rng, hot_scale)
        else:
            sigma = xp.zeros((3,) + lat.shape)
            sigma[2] = 1.0 if sig_how == "cold" else lat.eps
        if self.geom is None:
            s = np.zeros(0)
        else:
            s = self.geom.start(rng, s_how)
        return self.pack(sigma, xp.asarray(s))

    # ---- boson ----------------------------------------------------------------------------------
    def s_boson(self, fields):
        sigma, s = self.split(fields)
        xp = xp_of(fields)
        s2 = xp.sum(sigma**2, axis=0)
        val = xp.sum(0.5 * s2 + 0.25 * self.lam * s2**2)
        if self.kappa:
            val = val - 0.5 * self.kappa * xp.sum(sigma * self.lat.box(sigma))
        val = float(val)
        if self.D.n_fields:
            val += float(xp.sum(self.geom.dev(xp)["inv4g"] * s * s))
        return val

    def f_boson(self, fields):
        """-dS_B/d(sigma, s)."""
        sigma, s = self.split(fields)
        xp = xp_of(fields)
        s2 = xp.sum(sigma**2, axis=0)
        f = -sigma * (1.0 + self.lam * s2)
        if self.kappa:
            f = f + self.kappa * self.lat.box(sigma)
        if self.D.n_fields:
            fs = -2.0 * self.geom.dev(xp)["inv4g"] * s
        else:
            fs = xp.zeros(0)
        return xp.concatenate([f.reshape(-1), fs])

    # ---- pseudofermion ------------------------------------------------------------------------
    def _A(self, fields):
        Y = self.D.prepare(fields)
        return lambda v: self.D.AA(v, Y=Y)

    def _cap(self, m_max):
        return m_max if self.lanczos_m_max is None else max(m_max, self.lanczos_m_max)

    def _lanczos(self, fields, v, f, want_vector, m_max):
        return lanczos(
            self._A(fields),
            v,
            f,
            tol=self.lanczos_tol,
            want_vector=want_vector,
            m_max=self._cap(m_max),
            check_grow=self.lanczos_check_grow,
        )

    def set_window(self, lam_min, lam_max):
        """Fix the force fraction for spec(D^dagger D) in [lam_min, lam_max] (called once per trajectory)."""
        lo, hi = lam_min * self.safety[0], lam_max * self.safety[1]
        self.window = (lo, hi)
        self.pf = force_pf("invsqrt", self.n_rational, lo, hi)
        return self.pf

    def s_pf(self, fields, phi, want_ritz=False):
        r = self._lanczos(fields, phi, lambda t: t**-0.5, False, ACTION_CAP)
        self.stats["lanczos_iters"] += r["m"]
        return (r["value"].real, r) if want_ritz else r["value"].real

    def heatbath_phi(self, fields, eta):
        """phi = (D^dagger D)^(1/4) eta for complex Gaussian eta. Returns (phi, Lanczos info)."""
        r = self._lanczos(fields, eta, lambda t: t**0.25, True, HEATBATH_CAP)
        self.stats["lanczos_iters"] += r["m"]
        return r["vec"], r

    def _ms_solve(self, Y, b, poles, tol=None):
        tol = self.cg_tol if tol is None else tol
        return multishift_cg(lambda v: self.D.AA(v, Y=Y), b, poles, tol=tol, maxit=100000)

    def pf_force(self, fields, phi, pf, Y=None):
        """-d/d(sigma, s) of phi^dagger [c0 + sum_l rho_l (A + p_l)^-1] phi. Returns (flat force, CG iterations).

        With chi_l = (A + p_l)^-1 phi: dS/dsigma_a(x) = 2 y sum_l rho_l Im[(D chi_l)(x)^dagger tau_a chi_l(x)] and
        -dS/da_l = -2 sum rho Re[(D chi)(u)^dagger chi(v) - (D chi)(v)^dagger chi(u)] for link l = {u, v}.
        """
        xp = xp_of(fields)
        if Y is None:
            Y = self.D.prepare(fields)
        chis, it = self._ms_solve(Y, phi, pf["poles"])
        self.stats["cg_iters"] += it
        self.stats["solves"] += 1
        C = xp.ascontiguousarray(chis.transpose(1, 0, 2))
        DC = self.D.apply(C, Y=Y)
        rho = xp.asarray(pf["rho"])
        cu, cd = DC[..., 0].conj(), DC[..., 1].conj()
        u, d = C[..., 0], C[..., 1]
        t1 = cu * d + cd * u
        t2 = -1j * cu * d + 1j * cd * u
        t3 = cu * u - cd * d
        fsig = (-2.0 * self.y) * xp.stack([(t1.imag * rho).sum(1), (t2.imag * rho).sum(1), (t3.imag * rho).sum(1)])
        if not self.D.n_fields:
            return xp.concatenate([fsig.reshape(-1), xp.zeros(0)]), it
        t = self.geom.dev(xp)
        lu, lv = t["lu"], t["lv"]
        e = (DC[lu].conj() * C[lv] - DC[lv].conj() * C[lu]).real.sum(-1)
        fa = -2.0 * (e * rho).sum(1)
        fs = t["I"] @ fa
        return xp.concatenate([fsig.reshape(-1), fs]), it

    def f_pf(self, fields, phi):
        if self.pf is None:
            raise RuntimeError("call set_window() first")
        return self.pf_force(fields, phi, self.pf)

    # ---- propagator for measurements -------------------------------------------------------------
    def solve_Dinv(self, fields, eta, tol=None, Y=None):
        """D^-1 eta = -(D^dagger D)^-1 D eta (uses D^dagger = -D)."""
        assert self.D.mass == 0.0
        if Y is None:
            Y = self.D.prepare(fields)
        rhs = -self.D.apply(eta, Y=Y)
        z, it = cg(lambda v: self.D.AA(v, Y=Y), rhs, tol=tol or self.cg_tol)
        self.stats["cg_iters"] += it
        self.stats["solves"] += 1
        return z


class HasenbuschModel(Model):
    """Model with det(D^dagger D)^(1/2) split into Hasenbusch ratio terms and a heavy term.

    Term k = 0 ... n-1 is the ratio with (a, b) = (s_k, s_{k+1}):
        S = phi^dagger [(A + b)/(A + a)]^(1/2) phi,  heat bath phi = [(A + a)/(A + b)]^(1/4) eta;
    term n is the heavy term S = phi^dagger (A + s_n)^(-1/2) phi, heat bath (A + s_n)^(1/4) eta.
    """

    def __init__(self, lat, op, kappa, lam, shifts, **kw):
        super().__init__(lat, op, kappa, lam, **kw)
        s = [0.0] + sorted(float(x) for x in shifts)
        assert all(s[i] < s[i + 1] for i in range(len(s) - 1)) and len(s) >= 2, s
        self.shifts = s[1:]
        self.terms = [("ratio", s[k], s[k + 1]) for k in range(len(s) - 1)] + [("heavy", s[-1], None)]
        self.pfs = None
        self.stats.update(cg_iters_term=[0] * len(self.terms), lanczos_iters_term=[0] * len(self.terms))

    @property
    def nterms(self):
        return len(self.terms)

    def term_function(self, k, what):
        kind, a, b = self.terms[k]
        if kind == "heavy":
            return (lambda t: (t + a) ** -0.5) if what == "action" else (lambda t: (t + a) ** 0.25)
        if what == "action":
            return lambda t: ((t + b) / (t + a)) ** 0.5
        return lambda t: ((t + a) / (t + b)) ** 0.25

    def set_window(self, lam_min, lam_max):
        lo, hi = lam_min * self.safety[0], lam_max * self.safety[1]
        self.window = (lo, hi)
        self.pfs = []
        for kind, a, b in self.terms:
            if kind == "heavy":
                self.pfs.append(force_pf("shifted", self.n_rational, lo, hi, a=a))
            else:
                self.pfs.append(force_pf("ratio", self.n_rational, lo, hi, a=a, b=b))
        self.pf = dict(maxerr=max(p["maxerr"] for p in self.pfs))
        return self.pfs

    def s_term(self, k, fields, phi, want_ritz=False):
        r = self._lanczos(fields, phi, self.term_function(k, "action"), False, ACTION_CAP)
        self.stats["lanczos_iters"] += r["m"]
        self.stats["lanczos_iters_term"][k] += r["m"]
        return (r["value"].real, r) if want_ritz else r["value"].real

    def heatbath_term(self, k, fields, eta):
        r = self._lanczos(fields, eta, self.term_function(k, "heatbath"), True, HEATBATH_CAP)
        self.stats["lanczos_iters"] += r["m"]
        self.stats["lanczos_iters_term"][k] += r["m"]
        return r["vec"], r

    def f_term(self, k, fields, phi, Y=None):
        if self.pfs is None:
            raise RuntimeError("call set_window() first")
        f, it = self.pf_force(fields, phi, self.pfs[k], Y=Y)
        self.stats["cg_iters_term"][k] += it
        return f, it

    def s_pf_terms(self, fields, phis, want_ritz=False):
        vals, infos = [], []
        for k, phi in enumerate(phis):
            v, r = self.s_term(k, fields, phi, want_ritz=True)
            vals.append(v)
            infos.append(r)
        return (vals, infos) if want_ritz else vals


def build_model(
    lat,
    y,
    kappa,
    lam,
    g10=0.0,
    g6=0.0,
    quads="all",
    h=0.0,
    pattern="c0",
    h_sel=0.0,
    mask=(0.0, 0.0, 0.0, 1.0),
    mass=0.0,
    hasenbusch=None,
    geom=None,
    **kw,
):
    """Convenience constructor: geometry (only with link fields), operator and (Hasenbusch) model."""
    from .wedge import WedgeGeometry

    if geom is None and (g10 or g6 or quads == "links"):
        geom = WedgeGeometry(lat, quads, g10, g6)
    op = DoubletOperator(lat, y, geom=geom, h=h, pattern=pattern, mass=mass, h_sel=h_sel, mask=mask)
    if hasenbusch:
        return HasenbuschModel(lat, op, kappa, lam, hasenbusch, **kw)
    return Model(lat, op, kappa, lam, **kw)


__all__ = ["Model", "HasenbuschModel", "build_model", "np"]
