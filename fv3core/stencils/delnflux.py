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
def fx2_order(q: FloatField, del6_v: FloatField, fx2: FloatField, order: int):
    with computation(PARALLEL), interval(...):
        fx2[0, 0, 0] = del6_v * (q[-1, 0, 0] - q)
        fx2[0, 0, 0] = -1.0 * fx2 if order > 1 else fx2


@gtstencil()
def fy2_order(q: FloatField, del6_u: FloatField, fy2: FloatField, order: int):
    with computation(PARALLEL), interval(...):
        fy2[0, 0, 0] = del6_u * (q[0, -1, 0] - q)
        fy2[0, 0, 0] = fy2 * -1 if order > 1 else fy2


@gtscript.function
def fx2_func(q: FloatField, del6_v: FloatField, order: int):
    fx2 = del6_v * (q[-1, 0, 0] - q)
    fx2 = -1.0 * fx2 if order > 1 else fx2
    return fx2


@gtscript.function
def fy2_func(q: FloatField, del6_u: FloatField, order: int):
    fy2 = del6_u * (q[0, -1, 0] - q)
    fy2 = fy2 * -1 if order > 1 else fy2
    return fy2


# WARNING: untested
@gtstencil()
def fx2_firstorder_use_sg(
    q: FloatField,
    sin_sg1: FloatField,
    sin_sg3: FloatField,
    dy: FloatField,
    rdxc: FloatField,
    fx2: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx2[0, 0, 0] = (
            0.5 * (sin_sg3[-1, 0, 0] + sin_sg1) * dy * (q[-1, 0, 0] - q) * rdxc
        )


# WARNING: untested
@gtstencil()
def fy2_firstorder_use_sg(
    q: FloatField,
    sin_sg2: FloatField,
    sin_sg4: FloatField,
    dx: FloatField,
    rdyc: FloatField,
    fy2: FloatField,
):
    with computation(PARALLEL), interval(...):
        fy2[0, 0, 0] = (
            0.5 * (sin_sg4[0, -1, 0] + sin_sg2) * dx * (q[0, -1, 0] - q) * rdyc
        )


@gtstencil()
def d2_highorder(fx2: FloatField, fy2: FloatField, rarea: FloatField, d2: FloatField):
    with computation(PARALLEL), interval(...):
        d2[0, 0, 0] = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea


@gtscript.function
def d2_high_order(fx2: FloatField, fy2: FloatField, rarea: FloatField):
    d2 = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
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
    fx2: FloatField,
    fy2: FloatField,
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
            fx2 = fx2_func(q, del6_v, order)
        if __INLINED(nord > 0):
            q = corners.copy_corners_y(q)
        with horizontal(
            region[
                (local_is - nord) : (local_ie + nord + 1),
                (local_js - nord) : (local_je + nord + 2),
            ]
        ):
            fy2 = fy2_func(q, del6_u, order)


def higher_order_compute(
    fx2: FloatField,
    fy2: FloatField,
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
            d2 = d2_high_order(fx2, fy2, rarea)
        d2 = corners.copy_corners_x(d2)
        with horizontal(
            region[
                (local_is - nt) : (local_ie + nt + 2),
                (local_js - nt) : (local_je + nt + 1),
            ]
        ):
            fx2 = fx2_func(d2, del6_v, order)
        d2 = corners.copy_corners_y(d2)
        with horizontal(
            region[
                (local_is - nt) : (local_ie + nt + 1),
                (local_js - nt) : (local_je + nt + 2),
            ]
        ):
            fy2 = fy2_func(d2, del6_u, order)


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

    compute_no_sg(q, fx2, fy2, nord, damp, d2, kstart, nk, mass)

    if mass is None:
        add_diffusive_component(
            fx, fx2, fy, fy2, origin=diffuse_origin, domain=extended_domain
        )
    else:
        diffusive_damp(
            fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin, domain=extended_domain
        )
    return fx, fy


def compute_no_sg(
    q: FloatField,
    fx2: FloatField,
    fy2: FloatField,
    nord: int,
    damp_c: float,
    d2: FloatField,
    kstart: Optional[int] = 0,
    nk: Optional[int] = None,
    mass: Optional["FloatField"] = None,
):
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
        fx2,
        fy2,
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
                fx2,
                fy2,
                grid.rarea,
                d2,
                grid.del6_u,
                grid.del6_v,
                order=2 + n,
                origin=(grid.isd, grid.jsd, kstart),
                domain=(grid.nid, grid.njd, nk),
            )
