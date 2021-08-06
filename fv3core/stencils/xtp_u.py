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
from fv3core.stencils import xppm, yppm
from fv3core.utils.grid import GridData, GridIndexing, axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def advect_u(
    u: FloatField,
    u_on_cell_corners: FloatField,
    rdx: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
    dt: float,
):
    """
    Advect covariant C-grid x-wind using contravariant x-wind on cell corners.

    Inputs:
        u: covariant x-wind on C grid
        u_on_cell_corners: contravariant x-wind on cell corners
        rdx: 1.0 / dx
        bl: ???
        br: ???
        dt: timestep in seconds

    Returns:
        updated_u: u having been advected by u_on_cell_corners
    """
    # Could try merging this with xppm version.

    from __externals__ import iord

    b0 = bl + br
    cfl = (
        u_on_cell_corners * dt * rdx[-1, 0]
        if u_on_cell_corners > 0
        else u_on_cell_corners * dt * rdx
    )
    fx0 = xppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(iord < 8):
        advection_mask = xppm.get_advection_mask(bl, b0, br)
    else:
        advection_mask = 1.0
    return xppm.apply_flux(u_on_cell_corners, u, fx0, advection_mask)


def xtp_u_stencil_defn(
    courant: FloatField,
    u: FloatField,
    flux: FloatField,
    dx: FloatFieldIJ,
    dxa: FloatFieldIJ,
    rdx: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        flux = advect_u_along_x(courant, u, dx, dxa, rdx, 1.0)


@gtscript.function
def get_bl_br(u, dx, dxa):
    """
    Args:
        u: covariant x-wind on D-grid
        dx: gridcell spacing in x-direction
        dxa: gridcell spacing in x-direction on A-grid

    Returns:
        bl: ???
        br: ???
    """
    from __externals__ import i_end, i_start, iord, j_end, j_start

    if __INLINED(iord < 8):
        u_on_cell_corners = xppm.compute_al(u, dx)

        bl = u_on_cell_corners[0, 0, 0] - u[0, 0, 0]
        br = u_on_cell_corners[1, 0, 0] - u[0, 0, 0]

    else:
        dm = xppm.dm_iord8plus(u)
        u_on_cell_corners = xppm.al_iord8plus(u, dm)

        compile_assert(iord == 8)

        bl, br = xppm.blbr_iord8(u, u_on_cell_corners, dm)
        bl, br = xppm.bl_br_edges(bl, br, u, dxa, u_on_cell_corners, dm)

        with horizontal(region[i_start + 1, :], region[i_end - 1, :]):
            bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

    # Zero corners
    with horizontal(
        region[i_start - 1 : i_start + 1, j_start],
        region[i_start - 1 : i_start + 1, j_end + 1],
        region[i_end : i_end + 2, j_start],
        region[i_end : i_end + 2, j_end + 1],
    ):
        bl = 0.0
        br = 0.0
    return bl, br


@gtscript.function
def advect_u_along_x(
    ub_contra: FloatField,
    u: FloatField,
    dx: FloatFieldIJ,
    dxa: FloatFieldIJ,
    rdx: FloatFieldIJ,
    dt: float,
):
    """
    Advect covariant x-wind on D-grid using contravariant x-wind on cell corners.

    Named xtp_u in the original Fortran code.

    Args:
        u_on_cell_corners: contravariant x-wind on cell corners
        u: covariant x-wind on D-grid
        dx: gridcell spacing in x-direction
        dxa: a-grid gridcell spacing in x-direction
        rdx: 1 / dx
        dt: timestep in seconds
    """
    # in the Fortran, dt is folded into ub_contra and called "courant"
    bl, br = get_bl_br(u, dx, dxa)
    # TODO: merge this function with advect_u by calling get_bl_br inside advect_u
    updated_u = advect_u(u, ub_contra, rdx, bl, br, dt)
    return updated_u


class XTP_U:
    def __init__(
        self,
        grid_indexing: GridIndexing,
        grid_data: GridData,
        grid_type: int,
        iord: int,
    ):
        if iord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        assert grid_type < 3

        origin = grid_indexing.origin_compute()
        domain = grid_indexing.domain_compute(add=(1, 1, 0))
        self._dx = grid_data.dx
        self._dxa = grid_data.dxa
        self._rdx = grid_data.rdx
        ax_offsets = axis_offsets(grid_indexing, origin, domain)
        self.stencil = FrozenStencil(
            xtp_u_stencil_defn,
            externals={
                "iord": iord,
                "mord": iord,
                "xt_minmax": False,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, c: FloatField, u: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in x-dir.

        Args:
            c (in): product of x-dir wind on cell corners and timestep
            u (in): x-dir wind on D-grid
            flux (out): Flux of kinetic energy
        """
        self.stencil(
            c,
            u,
            flux,
            self._dx,
            self._dxa,
            self._rdx,
        )
