from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


#
# Flux value stencils
# ---------------------
def compute_zonal_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[-1, 0, 0] - a_in)


def compute_meridional_flux(flux: FloatField, a_in: FloatField, del_term: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        flux = del_term * (a_in[0, -1, 0] - a_in)


#
# Q update stencil
# ------------------
def update_q(
    q: FloatField, rarea: FloatFieldIJ, fx: FloatField, fy: FloatField, cd: float
):
    with computation(PARALLEL), interval(...):
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])
def del2cubed1(
        q: FloatField, rarea: FloatFieldIJ, del_u: FloatFieldIJ, del_v: FloatFieldIJ,cd: float, ntimes: int):
    with computation(PARALLEL), interval(...):
        fx = del_v * (q[-1, 0, 0] - q)
        fy = del_u * (q[0, -1, 0] - q)
        # race condition
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])


def del2cubed3(
        q: FloatField, rarea: FloatFieldIJ, del_u: FloatFieldIJ, del_v: FloatFieldIJ,cd: float, ntimes: int):
    with computation(PARALLEL), interval(...):
        fx = del_v * (q[-1, 0, 0] - q)
        fy = del_u * (q[0, -1, 0] - q)
        # race condition
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])
        #if ntimes > 1:
        fx = del_v * (q[-1, 0, 0] - q)
        fy = del_u * (q[0, -1, 0] - q)
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])
        #if ntimes > 2:
        fx = del_v * (q[-1, 0, 0] - q)
        fy = del_u * (q[0, -1, 0] - q)
        q = q + cd * rarea * (fx - fx[1, 0, 0] + fy - fy[0, 1, 0])
    
class HyperdiffusionDamping:
    """
    Fortran name is del2_cubed
    """

    def __init__(self, grid, nmax: int):
        """
        Args:
            grid: fv3core grid object
        """
        self.grid = spec.grid
        self._ntimes = min(3, nmax)
        origin =  self.grid.compute_origin()
        domain= self.grid.domain_shape_compute()
        if self._ntimes == 1:
            self._del2cubed = FrozenStencil(
                del2cubed1,
                origin=origin,
                domain=domain
            )
        elif self._ntimes == 3:
            self._del2cubed = FrozenStencil(
                del2cubed3,
                origin=origin,
                domain=domain
        )

    def __call__(self, qdel: FloatField, cd: float):
        """
        Perform hyperdiffusion damping/filtering

        Args:
            qdel (inout): Variable to be filterd
            nmax: Number of times to apply filtering
            cd: Damping coeffcient
        """


        self._del2cubed(
            qdel,
            self.grid.rarea,
            self.grid.del6_u,  self.grid.del6_v,
            cd,self._ntimes,
        )
