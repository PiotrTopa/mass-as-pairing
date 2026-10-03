"""Pin BLAS/OpenMP to one thread before numpy loads.

The equivalence tests compare against frozen references bit for bit; multi-threaded BLAS changes the summation
order and therefore the last bits of some reductions.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
