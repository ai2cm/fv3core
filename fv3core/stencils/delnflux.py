from typing import Optional

import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
import fv3core.utils.corners as corners
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy
from fv3core.utils.typing import FloatField


@gtstencil()
def fx_calc_stencil(q: FloatField, del6_v: FloatField, fx: FloatField, order: int):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = del6_v * (q[-1, 0, 0] - q)
        fx[0, 0, 0] = -1.0 * fx if order > 1 else fx


@gtstencil()
def fy_calc_stencil(q: FloatField, del6_u: FloatField, fy: FloatField, order: int):
    with computation(PARALLEL), interval(...):
        fy[0, 0, 0] = del6_u * (q[0, -1, 0] - q)
        fy[0, 0, 0] = fy * -1 if order > 1 else fy


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
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField, d2: FloatField):
    with computation(PARALLEL), interval(...):
        d2[0, 0, 0] = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea


@gtscript.function
def d2_high_order(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


@gtstencil()
def d2_damp(q: FloatField, d2: FloatField, damp: float):
    with computation(PARALLEL), interval(...):
        d2[0, 0, 0] = damp * q


@gtstencil()
def add_diffusive_component(
    fx: FloatField, fx2: FloatField, fy: FloatField, fy2: FloatField
):
    from __externals__ import local_ie, local_je

    with computation(PARALLEL), interval(...):
        with horizontal(region[:, : local_je + 1]):
            fx[0, 0, 0] = fx + fx2
        with horizontal(region[: local_ie + 1, :]):
            fy[0, 0, 0] = fy + fy2


@gtstencil()
def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: float,
):
    from __externals__ import local_ie, local_je

    with computation(PARALLEL), interval(...):
        with horizontal(region[:, : local_je + 1]):
            fx[0, 0, 0] = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        with horizontal(region[: local_ie + 1, :]):
            fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


@gtstencil()
def diffusive_damp_x(fx: FloatField, fx2: FloatField, mass: FloatField, damp: float):
    with computation(PARALLEL), interval(...):
        fx = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2


@gtstencil()
def diffusive_damp_y(fy: FloatField, fy2: FloatField, mass: FloatField, damp: float):
    with computation(PARALLEL), interval(...):
        fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


def fxy_order(
    q: FloatField,
    del6_u: FloatField,
    del6_v: FloatField,
    fx: FloatField,
    fy: FloatField,
    order: int,
):
    from __externals__ import local_ie, local_is, local_je, local_js, nord

    with computation(PARALLEL), interval(...):
        if __INLINED(nord > 0):
            q = corners.copy_corners_x(q)
        with horizontal(
            region[
                (local_is - nord) : (local_ie + nord + 2),
                (local_js - nord) : (local_je + nord + 1),
            ]
        ):
            fx = fx_calculation(q, del6_v, order)
        if __INLINED(nord > 0):
            q = corners.copy_corners_y(q)
        with horizontal(
            region[
                (local_is - nord) : (local_ie + nord + 1),
                (local_js - nord) : (local_je + nord + 2),
            ]
        ):
            fy = fy_calculation(q, del6_u, order)


def higher_order_compute(
    fx: FloatField,
    fy: FloatField,
    rarea: FloatField,
    d2: FloatField,
    del6_u: FloatField,
    del6_v: FloatField,
    order: int,
):
    from __externals__ import local_ie, local_is, local_je, local_js, nt

    with computation(PARALLEL), interval(...):
        with horizontal(
            region[
                (local_is - nt - 1) : (local_ie + nt + 2),
                (local_js - nt - 1) : (local_je + nt + 2),
            ]
        ):
            d2 = d2_high_order(fx, fy, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - nt) : (local_ie + nt + 2),
                (local_js - nt) : (local_je + nt + 1),
            ]
        ):
            fx = fx_calculation(d2, del6_v, order)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - nt) : (local_ie + nt + 1),
                (local_js - nt) : (local_je + nt + 2),
            ]
        ):
            fy = fy_calculation(d2, del6_u, order)


@gtstencil
def higher_order_compute_unroll3(
    fx: FloatField,
    fy: FloatField,
    rarea: FloatField,
    d2: FloatField,
    del6_u: FloatField,
    del6_v: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):

        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 3) : (local_ie + 3 + 2), (local_js - 3) : (local_je + 3 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 1)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 3) : (local_ie + 3 + 1), (local_js - 3) : (local_je + 3 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 1)

        with horizontal(
            region[
                (local_is - 2 - 1) : (local_ie + 2 + 2),
                (local_js - 2 - 1) : (local_je + 2 + 2),
            ]
        ):
            d2 = d2_high_order(fx, fy, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 2) : (local_ie + 2 + 2), (local_js - 2) : (local_je + 2 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 2)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 2) : (local_ie + 2 + 1), (local_js - 2) : (local_je + 2 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 2)

        with horizontal(
            region[
                (local_is - 1 - 1) : (local_ie + 1 + 2),
                (local_js - 1 - 1) : (local_je + 1 + 2),
            ]
        ):
            d2 = d2_high_order(fx, fy, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 1) : (local_ie + 1 + 2), (local_js - 1) : (local_je + 1 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 3)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 1) : (local_ie + 1 + 1), (local_js - 1) : (local_je + 1 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 3)

        with horizontal(
            region[
                (local_is - 0 - 1) : (local_ie + 0 + 2),
                (local_js - 0 - 1) : (local_je + 0 + 2),
            ]
        ):
            d2 = d2_high_order(fx, fy, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 0) : (local_ie + 0 + 2), (local_js - 0) : (local_je + 0 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 4)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 0) : (local_ie + 0 + 1), (local_js - 0) : (local_je + 0 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 4)


@gtstencil
def higher_order_compute_unroll2(
    fx: FloatField,
    fy: FloatField,
    rarea: FloatField,
    d2: FloatField,
    del6_u: FloatField,
    del6_v: FloatField,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(...):

        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 2) : (local_ie + 2 + 2), (local_js - 2) : (local_je + 2 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 1)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 2) : (local_ie + 2 + 1), (local_js - 2) : (local_je + 2 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 1)

        with horizontal(
            region[
                (local_is - 1 - 1) : (local_ie + 1 + 2),
                (local_js - 1 - 1) : (local_je + 1 + 2),
            ]
        ):
            d2 = d2_high_order(fx, fy, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 1) : (local_ie + 1 + 2), (local_js - 1) : (local_je + 1 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 2)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 1) : (local_ie + 1 + 1), (local_js - 1) : (local_je + 1 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 2)

        with horizontal(
            region[
                (local_is - 0 - 1) : (local_ie + 0 + 2),
                (local_js - 0 - 1) : (local_je + 0 + 2),
            ]
        ):
            d2 = d2_high_order(fx, fy, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - 0) : (local_ie + 0 + 2), (local_js - 0) : (local_je + 0 + 1)
            ]
        ):
            fx = fx_calculation(d2, del6_v, 3)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - 0) : (local_ie + 0 + 1), (local_js - 0) : (local_je + 0 + 2)
            ]
        ):
            fy = fy_calculation(d2, del6_u, 3)


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
        q: Tracer field (in)
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
        d2 = utils.make_storage_from_shape(q.shape, full_origin)
    if damp_c <= 1e-4:
        return fx, fy
    damp = (damp_c * grid.da_min) ** (nord + 1)
    fx2 = utils.make_storage_from_shape(q.shape, full_origin)
    fy2 = utils.make_storage_from_shape(q.shape, full_origin)
    diffuse_origin = (grid.is_, grid.js, kstart)
    extended_domain = (grid.nic + 1, grid.njc + 1, nk)

    # compute_no_sg_multi_loop(q, fx2, fy2, nord, damp, d2, kstart, nk, mass)
    # compute_no_sg_merge_loop(q, fx2, fy2, nord, damp, d2, kstart, nk, mass)
    compute_no_sg_unroll(q, fx2, fy2, nord, damp, d2, kstart, nk, mass)

    if mass is None:
        add_diffusive_component(
            fx, fx2, fy, fy2, origin=diffuse_origin, domain=extended_domain
        )
    else:
        diffusive_damp(
            fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin, domain=extended_domain
        )
    return fx, fy


def compute_no_sg_multi_loop(
    q: FloatField,
    fx: FloatField,
    fy: FloatField,
    nord: int,
    damp_c: float,
    d2: FloatField,
    kstart: Optional[int] = 0,
    nk: Optional[int] = None,
    mass: Optional["FloatField"] = None,
):
    """
    Calculate deln-fluxes in a loop over nord, with multiple stencil calls in the loop.
    We'll apply diffusion later.
    Args:
        q: Tracer field (in)
        fx: x-flux on A-grid (inout)
        fy: y-flux on A-grid (inout)
        nord: Order of divergence damping (in)
        damp_c: damping coefficient (in)
        d2: A damped copy of the q field (in)
        kstart: k-level to begin computing on (in)
        nk: Number of k-levels to compute on (in)
        mass: Mass to weight the diffusive flux by (in)
    """
    grid = spec.grid
    i1 = grid.is_ - 1 - nord
    i2 = grid.ie + 1 + nord
    j1 = grid.js - 1 - nord
    j2 = grid.je + 1 + nord
    if nk is None:
        nk = grid.npz - kstart
    origin_d2 = (i1, j1, kstart)
    domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
    fxy_stencil = gtstencil(definition=fxy_order, externals={"nord": nord})
    if mass is None:
        d2_damp(q, d2, damp_c, origin=origin_d2, domain=domain_d2)
    else:
        d2 = copy(q, origin=origin_d2, domain=domain_d2)

    fxy_stencil(
        d2,
        grid.del6_u,
        grid.del6_v,
        fx,
        fy,
        order=1,
        origin=(grid.isd, grid.jsd, kstart),
        domain=(grid.nid, grid.njd, nk),
    )

    if nord > 0:
        for n in range(nord):
            nt = nord - 1 - n
            nt_origin_extended = (grid.is_ - nt - 1, grid.js - nt - 1, kstart)
            nt_ny = grid.je - grid.js + 3 + 2 * nt
            nt_nx = grid.ie - grid.is_ + 3 + 2 * nt
            nt_origin = (grid.is_ - nt, grid.js - nt, kstart)
            d2_highorder(
                fx,
                fy,
                grid.rarea,
                d2,
                origin=nt_origin_extended,
                domain=(nt_nx, nt_ny, nk),
            )
            corners.copy_corners_x_stencil(
                d2, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
            )
            fx_calc_stencil(
                d2,
                grid.del6_v,
                fx,
                order=2 + n,
                origin=nt_origin,
                domain=(nt_nx - 1, nt_ny - 2, nk),
            )
            corners.copy_corners_y_stencil(
                d2, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
            )

            fy_calc_stencil(
                d2,
                grid.del6_u,
                fy,
                order=2 + n,
                origin=nt_origin,
                domain=(nt_nx - 2, nt_ny - 1, nk),
            )


def compute_no_sg_merge_loop(
    q: FloatField,
    fx: FloatField,
    fy: FloatField,
    nord: int,
    damp_c: float,
    d2: FloatField,
    kstart: Optional[int] = 0,
    nk: Optional[int] = None,
    mass: Optional["FloatField"] = None,
):
    """
    Calculate deln-fluxes in a loop over nord, with one stencil defined in the loop.
    We'll apply diffusion later.
    Args:
        q: Tracer field (in)
        fx: x-flux on A-grid (inout)
        fy: y-flux on A-grid (inout)
        nord: Order of divergence damping (in)
        damp_c: damping coefficient (in)
        d2: A damped copy of the q field (in)
        kstart: k-level to begin computing on (in)
        nk: Number of k-levels to compute on (in)
        mass: Mass to weight the diffusive flux by (in)
    """
    grid = spec.grid
    i1 = grid.is_ - 1 - nord
    i2 = grid.ie + 1 + nord
    j1 = grid.js - 1 - nord
    j2 = grid.je + 1 + nord
    if nk is None:
        nk = grid.npz - kstart
    origin_d2 = (i1, j1, kstart)
    domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
    fxy_stencil = gtstencil(definition=fxy_order, externals={"nord": nord})
    if mass is None:
        d2_damp(q, d2, damp_c, origin=origin_d2, domain=domain_d2)
    else:
        d2 = copy(q, origin=origin_d2, domain=domain_d2)

    fxy_stencil(
        d2,
        grid.del6_u,
        grid.del6_v,
        fx,
        fy,
        order=1,
        origin=(grid.isd, grid.jsd, kstart),
        domain=(grid.nid, grid.njd, nk),
    )

    if nord > 0:
        for n in range(nord):
            nt = nord - 1 - n
            looped_stencil = gtstencil(
                definition=higher_order_compute, externals={"nt": nt}
            )
            looped_stencil(
                fx,
                fy,
                grid.rarea,
                d2,
                grid.del6_u,
                grid.del6_v,
                order=2 + n,
                origin=(grid.isd, grid.jsd, kstart),
                domain=(grid.nid, grid.njd, nk),
            )


def compute_no_sg_unroll(
    q: FloatField,
    fx: FloatField,
    fy: FloatField,
    nord: int,
    damp_c: float,
    d2: FloatField,
    kstart: Optional[int] = 0,
    nk: Optional[int] = None,
    mass: Optional["FloatField"] = None,
):
    """
    Calculate deln-fluxes by unrolling the loop over nord.
    We'll apply diffusion later.
    Args:
        q: Tracer field (in)
        fx: x-flux on A-grid (inout)
        fy: y-flux on A-grid (inout)
        nord: Order of divergence damping (in)
        damp_c: damping coefficient (in)
        d2: A damped copy of the q field (in)
        kstart: k-level to begin computing on (in)
        nk: Number of k-levels to compute on (in)
        mass: Mass to weight the diffusive flux by (in)
    """
    grid = spec.grid
    i1 = grid.is_ - 1 - nord
    i2 = grid.ie + 1 + nord
    j1 = grid.js - 1 - nord
    j2 = grid.je + 1 + nord
    if nk is None:
        nk = grid.npz - kstart
    origin_d2 = (i1, j1, kstart)
    domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
    if mass is None:
        d2_damp(q, d2, damp_c, origin=origin_d2, domain=domain_d2)
    else:
        d2 = copy(q, origin=origin_d2, domain=domain_d2)

    if nord > 0:
        if nord == 3:  # 2, 1, 0
            higher_order_compute_unroll3(
                fx,
                fy,
                grid.rarea,
                d2,
                grid.del6_u,
                grid.del6_v,
                origin=(grid.isd, grid.jsd, kstart),
                domain=(grid.nid, grid.njd, nk),
            )
        elif nord == 2:  # 1, 0
            higher_order_compute_unroll2(
                fx,
                fy,
                grid.rarea,
                d2,
                grid.del6_u,
                grid.del6_v,
                origin=(grid.isd, grid.jsd, kstart),
                domain=(grid.nid, grid.njd, nk),
            )
        else:
            raise NotImplementedError("nord is currently limited to 3, 2, or 0")
    else:
        fxy_stencil = gtstencil(definition=fxy_order, externals={"nord": nord})
        fxy_stencil(
            d2,
            grid.del6_u,
            grid.del6_v,
            fx,
            fy,
            order=1,
            origin=(grid.isd, grid.jsd, kstart),
            domain=(grid.nid, grid.njd, nk),
        )
