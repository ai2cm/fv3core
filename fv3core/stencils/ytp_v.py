from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    compile_assert,
    computation,
    horizontal,
    interval,
    region,
)

from fv3core.decorators import FrozenStencil
from fv3core.stencils import yppm
from fv3core.utils.grid import GridData, GridIndexing, axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def advect_v(
    v: FloatField,
    vb_contra: FloatField,
    rdy: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
    dt: float,
):
    """
    Advect covariant C-grid y-wind using contravariant y-wind on cell corners.

    Inputs:
        v: covariant y-wind on D grid
        vb_contra: contravariant y-wind on cell corners
        rdy: 1.0 / dy
        bl: ???
        br: ???
        dt: timestep in seconds

    Returns:
        updated_v: v having been advected by v_on_cell_corners
    """
    from __externals__ import jord

    b0 = bl + br
    cfl = vb_contra * dt * rdy[0, -1] if vb_contra > 0 else vb_contra * dt * rdy[0, 0]
    fx0 = yppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(jord < 8):
        advection_mask = yppm.get_advection_mask(bl, b0, br)
    else:
        advection_mask = 1.0
    return yppm.final_flux(vb_contra, v, fx0, advection_mask)


def advect_v_along_y_stencil_defn(
    courant: FloatField,
    v: FloatField,
    flux: FloatField,
    dy: FloatFieldIJ,
    dya: FloatFieldIJ,
    rdy: FloatFieldIJ,
):

    with computation(PARALLEL), interval(...):
        flux = advect_v_along_y(courant, v, dy, dya, rdy, 1.0)


@gtscript.function
def advect_v_along_y(
    vb_contra: FloatField,
    v: FloatField,
    dy: FloatFieldIJ,
    dya: FloatFieldIJ,
    rdy: FloatFieldIJ,
    dt: float,
):
    """
    Advect covariant y-wind on D-grid using contravariant y-wind on cell corners.

    Named xtp_u in the original Fortran code.

    Args:
        vb_contra: contravariant y-wind on cell corners
        u: covariant x-wind on D-grid
        dy: gridcell spacing in y-direction
        dya: a-grid gridcell spacing in y-direction
        rdy: 1 / dy
        dt: timestep in seconds
    """
    from __externals__ import i_end, i_start, j_end, j_start, jord

    if __INLINED(jord < 8):
        al = yppm.compute_al(v, dy)

        bl = al[0, 0, 0] - v[0, 0, 0]
        br = al[0, 1, 0] - v[0, 0, 0]

    else:
        dm = yppm.dm_jord8plus(v)
        al = yppm.al_jord8plus(v, dm)

        compile_assert(jord == 8)

        bl, br = yppm.blbr_jord8(v, al, dm)
        bl, br = yppm.bl_br_edges(bl, br, v, dya, al, dm)

        with horizontal(region[:, j_start + 1], region[:, j_end - 1]):
            bl, br = yppm.pert_ppm_standard_constraint_fcn(v, bl, br)

    # Zero corners
    with horizontal(
        region[i_start, j_start - 1 : j_start + 1],
        region[i_start, j_end : j_end + 2],
        region[i_end + 1, j_start - 1 : j_start + 1],
        region[i_end + 1, j_end : j_end + 2],
    ):
        bl = 0.0
        br = 0.0

    return advect_v(v, vb_contra, rdy, bl, br, dt)


class YTP_V:
    def __init__(
        self,
        grid_indexing: GridIndexing,
        grid_data: GridData,
        grid_type: int,
        jord: int,
    ):
        if jord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently ytp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert grid_type < 3

        origin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(1, 1, 0))
        self._dy = grid_data.dy
        self._dya = grid_data.dya
        self._rdy = grid_data.rdy
        ax_offsets = axis_offsets(grid_indexing, origin, domain)

        self.stencil = FrozenStencil(
            advect_v_along_y_stencil_defn,
            externals={
                "jord": jord,
                "mord": jord,
                "xt_minmax": False,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, c: FloatField, v: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in y-dir.

        Args:
        c (in): product of y-dir wind on cell corners and timestep
        v (in): y-dir wind on Arakawa D-grid
        flux (out): Flux of kinetic energy
        """

        self.stencil(c, v, flux, self._dy, self._dya, self._rdy)
