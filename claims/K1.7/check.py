"""K1.7: the anomaly-free Z_4 squares to fermion parity; the two-state partner algebra (zero vs pole vs seesaw); the
composite Dirac partner psibar psibar psi of one Weyl 16 and its Majorana channels; the lattice eps-vertex partner."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import itertools

import numpy as np

from masspairing.algebra.characters import (
    decompose,
    dims,
    invariants,
    ms_ext_powers,
    ms_sym_powers,
    ms_tensor,
    multiplicity,
    pretty,
    spinor4,
    spinor16,
    weyl_lorentz,
)
from masspairing.claimcheck import Check
from masspairing.lattice import Lattice

np.set_printoptions(legacy="1.25")  # plain scalar reprs in the printed numbers
c = Check("K1.7")

# ---- (1) Z4^2 = (-1)^F
g16 = 1j * np.eye(16)  # e^{i pi Q_psi / 2}, Q_psi = 1 on the 16
c.item(
    "continuum: (e^{i pi Q_psi/2})^2 = -1 on the 16",
    np.allclose(g16 @ g16, -np.eye(16)),
    np.allclose(g16 @ g16, -np.eye(16)),
)
lat = Lattice((4, 4, 4, 4))
eps = lat.eps_np.reshape(lat.V)
glat = np.diag(1j**eps)  # e^{i pi Q_eps / 2}: i on even, -i on odd sites
c.item(
    "lattice: (e^{i pi Q_eps/2})^2 = -1 on every site of 4^4",
    np.allclose(glat @ glat, -np.eye(lat.V)),
    np.allclose(glat @ glat, -np.eye(lat.V)),
)
bad = [(n, m) for n in range(7) for m in range(7) if (n + m) % 2 == 1 and ((n - m) % 2 == 0 or (2 * (n - m)) % 4 != 2)]
c.item(
    "every fermionic psi^n psibar^m (n + m odd, n, m <= 6): odd charge, Majorana square of charge 2 mod 4; "
    "violations",
    len(bad),
    not bad,
)
bad = [s for k in (1, 3, 5) for s in itertools.product((1, -1), repeat=k) if sum(s) % 2 == 0 or (2 * sum(s)) % 4 != 2]
c.item(
    "lattice: every odd product of chi's (any sublattices, 1, 3, 5 fields) has odd U(1)_eps charge; violations",
    len(bad),
    not bad,
)

# ---- (2) two-state partner algebra G^-1 = [[ip, D], [D, ip + Mp]]
rng = np.random.default_rng(149)
worst_zero = worst_pole = worst_seesaw = 0.0
for _ in range(200):
    D, Mp = rng.uniform(0.2, 2.0), rng.uniform(0.2, 2.0)
    p = 1e-6
    Minv = np.linalg.inv(np.array([[1j * p, D], [D, 1j * p + Mp]]))
    worst_pole = max(worst_pole, abs(Minv[0, 0] - (-Mp / D**2)) / (Mp / D**2))
    for p in (1e-3, 1e-2, 1e-1):
        Minv0 = np.linalg.inv(np.array([[1j * p, D], [D, 1j * p]]))
        worst_zero = max(worst_zero, abs(abs(Minv0[0, 0]) - p / (p**2 + D**2)) / (p / (p**2 + D**2)))
    M = 1000.0 * Mp
    ev = np.linalg.eigvalsh(np.array([[0.0, D], [D, M]]))
    worst_seesaw = max(worst_seesaw, abs(min(abs(ev)) - D**2 / M) / (D**2 / M))
c.item(
    "G_chichi(p -> 0) = -M_p/Delta^2 (rel. dev. at p = 1e-6, 200 random (Delta, M_p))",
    worst_pole,
    worst_pole < 1e-4,
    fmt="{:.1e}",
)
c.item("M_p = 0: |G_chichi(p)| = |p|/(p^2 + Delta^2) (rel. dev.)", worst_zero, worst_zero < 1e-10, fmt="{:.1e}")
c.item(
    "M_p >> Delta: light eigenvalue Delta^2/M_p (rel. dev. at M_p/Delta >= 100)",
    worst_seesaw,
    worst_seesaw < 2e-4,
    fmt="{:.1e}",
)

# ---- (3) Spin(10) x Lorentz characters (D5 x A1 x A1)
s16b = spinor16(+1)
V, Vb = weyl_lorentz(-1), weyl_lorentz(-1, dotted=True)
lam_partner = max(s16b) + (1, 0)  # 16bar (x) (1/2, 0)
m1 = multiplicity(ms_tensor(ms_tensor(Vb, Vb), V), lam_partner, k=2)
m3 = multiplicity(ms_tensor(ms_tensor(V, V), V), lam_partner, k=2)
c.item(
    "multiplicity of 16bar (x) (1/2,0) in psibar psibar psi (charge -1) and in psi psi psi (charge 3)",
    (m1, m3),
    (m1, m3) == (3, 4),
)
n_dirac = invariants(ms_tensor(V, ms_tensor(ms_tensor(Vb, Vb), V)), k=2)
n_grass = invariants(ms_tensor(ms_ext_powers(V, 2)[2], ms_ext_powers(Vb, 2)[2]), k=2)
c.item(
    "invariants of psi.(psibar psibar psi): character level, as Grassmann polynomials (= |Phi10|^2, |Phi126|^2)",
    (n_dirac, n_grass),
    (n_dirac, n_grass) == (3, 2),
)
d = decompose(ms_sym_powers(s16b, 2)[2])
c.item("Sym^2(16bar): the composite Majorana mass channels", pretty(d), dims(d) == [10, 126])

# ---- (4) lattice SU(4) = Spin(6) (D3 characters)
s4, s4b = spinor4(+1), spinor4(-1)
c.item(
    "Lambda^3(4) = 4bar: Psi^a = eps_abcd chi^b chi^c chi^d",
    ms_ext_powers(s4, 3)[3] == s4b,
    ms_ext_powers(s4, 3)[3] == s4b,
)
n = invariants(ms_tensor(s4, s4b), n=3)
c.item("singlets in 4 x 4bar (chi^a Psi^a is the on-site vertex, charge 4 = 0 mod 4)", n, n == 1)
d10b, d6 = decompose(ms_sym_powers(s4b, 2)[2], n=3), decompose(ms_ext_powers(s4b, 2)[2], n=3)
c.item(
    "composite pair Psi_x Psi_z: Sym^2(4bar), Lambda^2(4bar) (charge 6 = 2 mod 4)",
    (pretty(d10b, n=3), pretty(d6, n=3)),
    pretty(d10b, n=3) == "10(1,1,-1)" and pretty(d6, n=3) == "6(1,0,0)",
)
e10, e6 = decompose(ms_sym_powers(s4, 2)[2], n=3), decompose(ms_ext_powers(s4, 2)[2], n=3)
c.item(
    "elementary pair chi_x chi_z: Sym^2(4), Lambda^2(4)",
    (pretty(e10, n=3), pretty(e6, n=3)),
    pretty(e10, n=3) == "10(1,1,1)" and pretty(e6, n=3) == "6(1,0,0)",
)
c.done()
