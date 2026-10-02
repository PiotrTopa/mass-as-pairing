"""masspairing: the lattice code behind "Mass as pairing".

Four reduced staggered fermion flavours (the lattice proxy of one Spin(10) generation) with the epsilon-vertex
sigma field, optional sign-free link fields of the completed 10-channel term, and flavour-blind or flavour-selective
same-parity sources; rational hybrid Monte Carlo; the observables and estimators of the paper's claims.
"""

from .action import HasenbuschModel, Model, build_model
from .hmc import HMC, MTSHMC
from .lattice import Lattice, kinetic_matrix, parse_bc
from .operator import DoubletOperator
from .wedge import WedgeGeometry

__version__ = "1.0.0"

__all__ = [
    "DoubletOperator",
    "HMC",
    "HasenbuschModel",
    "Lattice",
    "MTSHMC",
    "Model",
    "WedgeGeometry",
    "build_model",
    "kinetic_matrix",
    "parse_bc",
]
