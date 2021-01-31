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
from fv3core.decorators import gtstencil
from fv3core.stencils import xppm, yppm
from fv3core.utils.typing import FloatField


@gtscript.function
def get_flux_u_ord8plus(
    q: FloatField, c: FloatField, rdx: FloatField, bl: FloatField, br: FloatField
):
    b0 = bl + br
    cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
    fx1 = xppm.fx1_fn(cfl, br, b0, bl)
    return q[-1, 0, 0] + fx1 if c > 0.0 else q + fx1


def get_flux_u(
    q: FloatField, c: FloatField, rdx: FloatField, bl: FloatField, br: FloatField
):
    from __externals__ import mord

    b0 = bl + br

    if __INLINED(mord == 5):
        smt5 = bl * br < 0
    else:
        smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[-1, 0, 0]:
        tmp = smt5[-1, 0, 0]
    else:
        tmp = smt5[-1, 0, 0] + smt5[0, 0, 0]

    cfl = c * rdx[-1, 0, 0] if c > 0 else c * rdx
    fx0 = xppm.fx1_fn(cfl, br, b0, bl)
    return xppm.final_flux(c, q, fx0, tmp)


def _compute_stencil(
    c: FloatField,
    u: FloatField,
    flux: FloatField,
    dx: FloatField,
    dxa: FloatField,
    rdx: FloatField,
):
    from __externals__ import i_end, i_start, iord, j_end, j_start, namelist

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

            flux = get_flux_u(u, c, rdx, bl, br)

        else:
            dm = xppm.dm_iord8plus(u)
            al = xppm.al_iord8plus(u, dm)

            assert __INLINED(iord == 8)
            # {
            bl, br = xppm.blbr_iord8(u, al, dm)
            # }

            assert __INLINED(namelist.grid_type < 3)
            # {
            with horizontal(region[i_start - 1, :]):
                bl, br = xppm.west_edge_iord8plus_0(u, dxa, dm)
                # bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_start, :]):
                bl, br = xppm.west_edge_iord8plus_1(u, dxa, dm)
                # bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_start + 1, :]):
                bl, br = xppm.west_edge_iord8plus_2(u, dm, al)
                bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_end - 1, :]):
                bl, br = xppm.east_edge_iord8plus_0(u, dm, al)
                bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_end, :]):
                bl, br = xppm.east_edge_iord8plus_1(u, dxa, dm)
                # bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

            with horizontal(region[i_end + 1, :]):
                bl, br = xppm.east_edge_iord8plus_2(u, dxa, dm)
                # bl, br = yppm.pert_ppm_standard_constraint_fcn(u, bl, br)

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

            flux = get_flux_u_ord8plus(u, c, rdx, bl, br)


def compute(c: FloatField, u: FloatField, v: FloatField, flux: FloatField):
    """
    Compute flux of kinetic energy in x-dir using the PPM method.

    Notes:
        v is passed in here, but is unused in the stencil.

    Args:
        c (in): Courant number
        u (in): x-dir wind on D-grid
        v (in): y-dir wind on D-grid
        flux (out): Flux of kinetic energy
    """
    grid = spec.grid
    iord = spec.namelist.hord_mt
    if iord not in (5, 6, 7, 8):
        raise NotImplementedError(
            "Currently xtp_v is only supported for hord_mt == 5,6,7,8"
        )

    stencil = gtstencil(
        definition=_compute_stencil,
        externals={
            "iord": iord,
            "mord": iord,  # Note: no abs here
            "xt_minmax": False,
        },
    )
    stencil(
        c,
        u,
        flux,
        grid.dx,
        grid.dxa,
        grid.rdx,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(add=(1, 1, 0)),
    )
