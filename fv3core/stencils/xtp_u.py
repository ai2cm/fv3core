from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.global_config as global_config
from fv3core.stencils import xppm, yppm
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def _get_flux(
    u: FloatField,
    courant: FloatField,
    rdx: FloatFieldIJ,
    bl: FloatField,
    br: FloatField,
):
    """
    Compute the x-dir flux of kinetic energy(?).

    Inputs:
        u: x-dir wind
        courant: Courant number in flux form
        rdx: 1.0 / dx
        bl: ???
        br: ???

    Returns:
        flux: Kinetic energy flux
    """
    # Could try merging this with xppm version.

    from __externals__ import iord

    b0 = bl + br
    cfl = courant * rdx[-1, 0] if courant > 0 else courant * rdx[0, 0]
    fx0 = xppm.fx1_fn(cfl, br, b0, bl)

    if __INLINED(iord < 8):
        tmp = xppm.get_tmp(bl, b0, br)
    else:
        tmp = 1.0
    return xppm.final_flux(courant, u, fx0, tmp)


def _compute_stencil(
    courant: FloatField,
    u: FloatField,
    flux: FloatField,
    dx: FloatFieldIJ,
    dxa: FloatFieldIJ,
    rdx: FloatFieldIJ,
):
    from __externals__ import i_end, i_start, iord, j_end, j_start

    with computation(PARALLEL), interval(...):

        if __INLINED(iord < 8):
            al = xppm.compute_al(u, dx)

            bl = al[0, 0, 0] - u[0, 0, 0]
            br = al[1, 0, 0] - u[0, 0, 0]

            # Zero corners
            with horizontal(
                region[i_start - 1 : i_start + 1, j_start],
                region[i_start - 1 : i_start + 1, j_end + 1],
                region[i_end : i_end + 2, j_start],
                region[i_end : i_end + 2, j_end + 1],
            ):
                bl = 0.0
                br = 0.0

        else:
            dm = xppm.dm_iord8plus(u)
            al = xppm.al_iord8plus(u, dm)

            assert __INLINED(iord == 8)
            # {
            bl, br = xppm.blbr_iord8(u, al, dm)
            # }
            # {
            with horizontal(region[i_start - 1, :]):
                bl, br = xppm.west_edge_iord8plus_0(u, dxa, dm)

            with horizontal(region[i_start, :]):
                bl, br = xppm.west_edge_iord8plus_1(u, dxa, dm)

            with horizontal(region[i_start + 1, :]):
                bl, br = xppm.west_edge_iord8plus_2(u, dm, al)
                bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_end - 1, :]):
                bl, br = xppm.east_edge_iord8plus_0(u, dm, al)
                bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_end, :]):
                bl, br = xppm.east_edge_iord8plus_1(u, dxa, dm)

            with horizontal(region[i_end + 1, :]):
                bl, br = xppm.east_edge_iord8plus_2(u, dxa, dm)

            # Zero corners
            with horizontal(
                region[i_start - 1 : i_start + 1, j_start],
                region[i_start - 1 : i_start + 1, j_end + 1],
                region[i_end : i_end + 2, j_start],
                region[i_end : i_end + 2, j_end + 1],
            ):
                bl = 0.0
                br = 0.0
            # }

        flux = _get_flux(u, courant, rdx, bl, br)


class XTP_U:
    def __init__(self, namelist):
        iord = spec.namelist.hord_mt
        if iord not in (5, 6, 7, 8):
            raise NotImplementedError(
                "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
            )
        self.grid = spec.grid
        self.origin = self.grid.compute_origin()
        self.domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, self.origin, self.domain)
        assert namelist.grid_type < 3
        self.stencil = gtscript.stencil(
            definition=_compute_stencil,
            externals={
                "iord": iord,
                "mord": iord,
                "xt_minmax": False,
                **ax_offsets,
            },
            backend=global_config.get_backend(),
            rebuild=global_config.get_rebuild(),
        )

    def __call__(self, c: FloatField, u: FloatField, flux: FloatField):
        """
        Compute flux of kinetic energy in x-dir.

        Args:
            c (in): Courant number in flux form
            u (in): x-dir wind on D-grid
            flux (out): Flux of kinetic energy
        """
        self.stencil(
            c,
            u,
            flux,
            self.grid.dx,
            self.grid.dxa,
            self.grid.rdx,
            origin=self.origin,
            domain=self.domain,
        )
