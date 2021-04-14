from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, FORWARD, computation, interval, horizontal, region

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK

@gtstencil()
def copy_corners_x_nord(q: FloatField, nord: FloatFieldK):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        if nord > 0:
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q = q[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q = q[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q = q[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q = q[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q = q[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q = q[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q = q[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q = q[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q = q[0, 1, 0]
            with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
                q = q[2, -3, 0]
            with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
                q = q[1, -2, 0]
            with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
                q = q[0, -1, 0]
            with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
                q = q[1, -4, 0]
            with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
                q = q[0, -3, 0]
            with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
                q = q[-1, -2, 0]
            with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
                q = q[0, -5, 0]
            with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
                q = q[-1, -4, 0]
            with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
                q = q[-2, -3, 0]

@gtstencil()
def copy_corners_y_nord(q: FloatField, nord: FloatFieldK):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        if nord > 0:
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q = q[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q = q[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q = q[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q = q[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q = q[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q = q[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q = q[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q = q[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q = q[1, 0, 0]
            with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
                q = q[-3, 2, 0]
            with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
                q = q[-4, 1, 0]
            with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
                q = q[-5, 0, 0]
            with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
                q = q[-2, 1, 0]
            with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
                q = q[-3, 0, 0]
            with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
                q = q[-4, -1, 0]
            with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
                q = q[-1, 0, 0]
            with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
                q = q[-2, -1, 0]
            with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
                q = q[-3, -2, 0]

@gtstencil()
def calc_damp(nord: FloatFieldK, damp_c: FloatFieldK, damp4: FloatFieldK, da_min:float):
    with computation(FORWARD), interval(...):
        damp4 = (damp_c * da_min) ** (nord + 1)

@gtstencil()
def fx_calc_stencil(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField, order: int):
    with computation(PARALLEL), interval(...):
        fx = fx_calculation(q, del6_v, order)

@gtstencil()
def fy_calc_stencil(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField, order: int):
    with computation(PARALLEL), interval(...):
        fy = fy_calculation(q, del6_u, order)

@gtstencil()
def fx_calc_stencil_column(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField, nord: FloatFieldK):
    with computation(PARALLEL), interval(...):
        fx = fx_calculation(q, del6_v, nord + 2)

@gtstencil()
def fy_calc_stencil_column(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField, nord: FloatFieldK):
    with computation(PARALLEL), interval(...):
        fy = fy_calculation(q, del6_u, nord + 2)


@gtscript.function
def fx_calculation(q: FloatField, del6_v: FloatField, order: int):
    fx = del6_v * (q[-1, 0, 0] - q)
    fx = -1.0 * fx if order > 1 else fx
    return fx


@gtscript.function
def fy_calculation(q: FloatField, del6_u: FloatField, order: int):
    fy = del6_u * (q[0, -1, 0] - q)
    fy = fy * -1 if order > 1 else fy
    return fy


# WARNING: untested
@gtstencil()
def fx_firstorder_use_sg(
    q: FloatField,
    sin_sg1: FloatField,
    sin_sg3: FloatField,
    dy: FloatField,
    rdxc: FloatField,
    fx: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = (
            0.5 * (sin_sg3[-1, 0, 0] + sin_sg1) * dy * (q[-1, 0, 0] - q) * rdxc
        )


# WARNING: untested
@gtstencil()
def fy_firstorder_use_sg(
    q: FloatField,
    sin_sg2: FloatField,
    sin_sg4: FloatField,
    dx: FloatField,
    rdyc: FloatField,
    fy: FloatField,
):
    with computation(PARALLEL), interval(...):
        fy[0, 0, 0] = (
            0.5 * (sin_sg4[0, -1, 0] + sin_sg2) * dx * (q[0, -1, 0] - q) * rdyc
        )


@gtstencil()
def d2_highorder_stencil(
    fx: FloatField, fy: FloatField, rarea: FloatFieldIJ, d2: FloatField
):
    with computation(PARALLEL), interval(...):
        d2 = d2_highorder(fx, fy, rarea)


@gtscript.function
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


@gtstencil()
def d2_damp(q: FloatField, d2: FloatField, damp: FloatFieldK, nord: FloatFieldK, nmax:int):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[i_start + nmax - nord: i_end - nmax + nord, j_start + nmax - nord: j_end - nmax + nord]
        ):
            d2[0, 0, 0] = damp * q


@gtstencil()
def d2_damp_interval(q: FloatField, d2: FloatField, damp: FloatFieldK, nmax:int, nord0: int, nord1: int, nord2: int, nord3: int):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(0,1):
        with horizontal(
            region[i_start + nmax - nord0: i_end - nmax + nord0, j_start + nmax - nord0: j_end - nmax + nord0]
        ):
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(1,2):
        with horizontal(
            region[i_start + nmax - nord1: i_end - nmax + nord1, j_start + nmax - nord1: j_end - nmax + nord1]
        ):
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(2,3):
        with horizontal(
            region[i_start + nmax - nord2: i_end - nmax + nord2, j_start + nmax - nord2: j_end - nmax + nord2]
        ):
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(3,None):
        with horizontal(
            region[i_start + nmax - nord3: i_end - nmax + nord3, j_start + nmax - nord3: j_end - nmax + nord3]
        ):
            d2[0, 0, 0] = damp * q

@gtstencil()
def copy_stencil_regional(q_in: FloatField, q_out: FloatField, nmax:int, nord0: int, nord1: int, nord2: int, nord3: int):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(0,1):
        with horizontal(
            region[i_start + nmax - nord0: i_end - nmax + nord0, j_start + nmax - nord0: j_end - nmax + nord0]
        ):
            q_out = q_in
    with computation(PARALLEL), interval(1,2):
        with horizontal(
            region[i_start + nmax - nord1: i_end - nmax + nord1, j_start + nmax - nord1: j_end - nmax + nord1]
        ):
            q_out = q_in
    with computation(PARALLEL), interval(2,3):
        with horizontal(
            region[i_start + nmax - nord2: i_end - nmax + nord2, j_start + nmax - nord2: j_end - nmax + nord2]
        ):
            q_out = q_in
    with computation(PARALLEL), interval(3,None):
        with horizontal(
            region[i_start + nmax - nord3: i_end - nmax + nord3, j_start + nmax - nord3: j_end - nmax + nord3]
        ):
            q_out = q_in

@gtstencil()
def add_diffusive_component(
    fx: FloatField, fx2: FloatField, fy: FloatField, fy2: FloatField
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + fx2
        fy[0, 0, 0] = fy + fy2


@gtstencil()
def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


def compute_delnflux_no_sg(
    q: FloatField,
    fx: FloatField,
    fy: FloatField,
    nord: int,
    damp_c: float,
    kstart: Optional[int] = 0,
    nk: Optional[int] = None,
    d2: Optional["FloatField"] = None,
    mass: Optional["FloatField"] = None,
):
    """
    Del-n damping for fluxes, where n = 2 * nord + 2
    Args:
        q: Field for which to calculate damped fluxes (in)
        fx: x-flux on A-grid (inout)
        fy: y-flux on A-grid (inout)
        nord: Order of divergence damping (in)
        damp_c: damping coefficient (in)
        kstart: k-level to begin computing on (in)
        nk: Number of k-levels to compute on (in)
        d2: A damped copy of the q field (in)
        mass: Mass to weight the diffusive flux by (in)
    """

    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    full_origin = (grid.isd, grid.jsd, kstart)
    if d2 is None:
        d2 = utils.make_storage_from_shape(
            q.shape, full_origin, cache_key="delnflux_d2"
        )
    if damp_c <= 1e-4:
        return fx, fy
    
    damp_3d = utils.make_storage_from_shape((1,1, nk)) # fields must be 3d to assign to them
    calc_damp(nord, damp_c, damp_3d, grid.da_min, origin=(0,0,0), domain=(1,1,nk))
    damp = utils.make_storage_data(damp_3d[0,0,:], (nk,), (0,))

    fx2 = utils.make_storage_from_shape(q.shape, full_origin, cache_key="delnflux_fx2")
    fy2 = utils.make_storage_from_shape(q.shape, full_origin, cache_key="delnflux_fy2")
    diffuse_origin = (grid.is_, grid.js, kstart)
    extended_domain = (grid.nic + 1, grid.njc + 1, nk)

    compute_no_sg(q, fx2, fy2, nord, damp, d2, kstart, nk, mass, conditional_calc=False)

    if mass is None:
        add_diffusive_component(fx, fx2, fy, fy2, origin=diffuse_origin, domain=extended_domain)
    else:
        # TODO: To join these stencils you need to overcompute, making the edges
        # 'wrong', but not actually used, separating now for comparison sanity.

        # diffusive_damp(fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin,
        # domain=(grid.nic + 1, grid.njc + 1, nk))
        diffusive_damp(
            fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin, domain=extended_domain
        )
    return fx, fy


def compute_no_sg(
        q,
        fx2,
        fy2,
        nord,
        damp_c,
        d2,
        kstart=0,
        nk=None,
        mass=None,
        conditional_calc=True,
        column_check=False,
):
    if (conditional_calc==True) and (column_check==False):
        if damp_c[0] <= 1e-5: #dcon_threshold
            raise Exception("damp <= 1e-5 in column_cols is untested")
    if max(nord[:]) > 3:
        raise NotImplementedError("nord > 3 is not implemented")
    nmax = max(nord[:])
    grid = spec.grid
    nord = int(nord)
    i1 = grid.is_ - 1 - nmax
    i2 = grid.ie + 1 + nmax
    j1 = grid.js - 1 - nmax
    j2 = grid.je + 1 + nmax
    if nk is None:
        nk = grid.npz - kstart
    origin_d2 = (i1, j1, kstart)
    domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
    f1_ny = grid.je - grid.js + 1 + 2 * nord
    f1_nx = grid.ie - grid.is_ + 2 + 2 * nord
    fx_origin = (grid.is_ - nord, grid.js - nord, kstart)
    if mass is None:
        d2_damp(q, d2, damp_c, origin=origin_d2, domain=domain_d2)
    else:
        copy_stencil(q, d2, origin=origin_d2, domain=domain_d2)

    copy_corners_x_nord(
        d2, nord, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
    )

    fx_calc_stencil(
        d2, grid.del6_v, fx2, order=1, origin=fx_origin, domain=(f1_nx, f1_ny, nk)
    )

    copy_corners_y_nord(
        d2, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
    )

    fy_calc_stencil(
        d2,
        grid.del6_u,
        fy2,
        order=1,
        origin=fx_origin,
        domain=(f1_nx - 1, f1_ny + 1, nk),
    )

    if nord > 0:
        for n in range(nord):
            nt = nord - 1 - n
            nt_origin = (grid.is_ - nt - 1, grid.js - nt - 1, kstart)
            nt_ny = grid.je - grid.js + 3 + 2 * nt
            nt_nx = grid.ie - grid.is_ + 3 + 2 * nt
            d2_highorder_stencil(
                fx2, fy2, grid.rarea, d2, origin=nt_origin, domain=(nt_nx, nt_ny, nk)
            )
            corners.copy_corners_x_stencil(
                d2, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
            )
            nt_origin = (grid.is_ - nt, grid.js - nt, kstart)
            fx_calc_stencil_column(
                d2,
                grid.del6_v,
                fx2,
                n,
                origin=nt_origin,
                domain=(nt_nx - 1, nt_ny - 2, nk),
            )
            corners.copy_corners_y_stencil(
                d2, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
            )

            fy_calc_stencil(
                d2,
                grid.del6_u,
                fy2,
                n,
                origin=nt_origin,
                domain=(nt_nx - 2, nt_ny - 1, nk),
            )
