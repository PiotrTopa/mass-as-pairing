"""The sign-free (tau_2 K) class of the doublet operator as linear algebra.

The doublet psi in C^{2V} is R^{4V} (four real flavours per site, ``flavour.pack``). The sign-free class is the set of
real-linear operators that are complex-linear (commute with J_i = multiplication by i) and commute with the
antiunitary tau_2 K (J_tau: psi -> tau_2 conj(psi)). J_i^2 = J_tau^2 = -1 and {J_i, J_tau} = 0: the two generate a
quaternion algebra H on the four real flavours of every site, whose commutant in M_4(R) is span{1, Gamma_a}
(``flavour.GAMMA``). The class is therefore span{1, Gamma_a} (x) R^{V x V}; every element commutes with the other
triplet Gamma' (``flavour.GAMMA2``), which acts on flavour space as right quaternion multiplication.

For such an operator the spectrum of D_c is paired as +-i mu and det D_c = det(D_c^+ D_c)^{1/2} > 0 on every
configuration; the on-site propagator blocks S(x, x) are quaternions i n(x).tau with n real.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from ..flavour import GAMMA

TAU2 = np.array([[0, -1j], [1j, 0]])


def realmap(f):
    """4 x 4 real matrix of a real-linear map f on C^2 in the basis (Re psi_0, Im psi_0, Re psi_1, Im psi_1)."""
    M = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = 1.0
        z = np.array([e[0] + 1j * e[1], e[2] + 1j * e[3]])
        w = f(z)
        M[:, j] = [w[0].real, w[0].imag, w[1].real, w[1].imag]
    return M


def quaternion_units():
    """(J_i, J_tau) as real 4 x 4 matrices."""
    return realmap(lambda z: 1j * z), realmap(lambda z: TAU2 @ np.conj(z))


def commutant(gens, n):
    """Orthonormal basis (columns, column-major vec) of {A in M_n(R): [A, g] = 0 for all g}."""
    rows = [np.kron(np.eye(n), g) - np.kron(g.T, np.eye(n)) for g in gens]
    return sla.null_space(np.vstack(rows))


def class_distance(M, V):
    """Relative distance of a 4V x 4V real matrix from span{1, Gamma_a} (x) R^{V x V} (orthogonal projection)."""
    B = [np.eye(4)] + list(GAMMA)
    Mb = M.reshape(V, 4, V, 4)
    P = np.zeros_like(Mb)
    for b in B:
        P += np.einsum("xayb,ab->xy", Mb, b)[:, None, :, None] * b[None, :, None, :] / 4.0
    return np.linalg.norm(P - Mb) / np.linalg.norm(Mb)


def flavour_restriction(M, V, u):
    """(u^T (x) 1) M (u (x) 1) for a unit flavour vector u (site-major 4V basis)."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    Pu = np.kron(np.eye(V), u[:, None])
    return Pu.T @ M @ Pu


def onsite_structure(Dc, V):
    """Structure of a dense doublet operator D_c (2V x 2V) on one configuration.

    Returns max|Re eigenvalue|, the +-i mu pairing defect of the spectrum, Im and Re-defect of log det D_c against
    1/2 log det D_c^+ D_c, |Tr D_c^-1|, max|tr S(x, x)|, the quaternion defect max|tau_2 S* tau_2 - S| of the on-site
    blocks, the residual of S(x, x) = i n.tau and of det S(x, x) = |n|^2, and n(x) itself."""
    ev = np.linalg.eigvals(Dc)
    im = np.sort(ev.imag)
    logdet = np.sum(np.log(ev))
    Sinv = np.linalg.inv(Dc)
    blocks = Sinv.reshape(V, 2, V, 2)[np.arange(V), :, np.arange(V), :]
    n = np.empty((V, 3))
    n[:, 0] = (-0.5j * (blocks[:, 0, 1] + blocks[:, 1, 0])).real
    n[:, 1] = (-0.5j * (-1j * blocks[:, 1, 0] + 1j * blocks[:, 0, 1])).real
    n[:, 2] = (-0.5j * (blocks[:, 0, 0] - blocks[:, 1, 1])).real
    taus = (np.array([[0, 1], [1, 0]]), TAU2, np.diag([1.0, -1.0]))
    recon = blocks - 1j * sum(n[:, a, None, None] * taus[a] for a in range(3))
    return dict(
        re_ev=float(np.abs(ev.real).max()),
        pairing=float(np.abs(im + im[::-1]).max()),
        im_logdet=float(logdet.imag),
        re_logdet_defect=float(logdet.real - 0.5 * np.sum(np.log(np.abs(ev) ** 2))),
        trace_inv=float(abs(np.trace(Sinv))),
        tr_onsite=float(np.abs(np.trace(blocks, axis1=1, axis2=2)).max()),
        quaternion=float(np.abs(np.einsum("ab,xbc,cd->xad", TAU2, blocks.conj(), TAU2) - blocks).max()),
        n_residual=float(np.abs(recon).max()),
        det_residual=float(np.abs(np.linalg.det(blocks) - (n**2).sum(1)).max()),
        n=n,
    )
